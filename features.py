"""Causal prosodic feature extraction for end-of-turn (EOT) detection.

HARD RULE: every feature computed here for a pause at `pause_start` uses
ONLY samples/frames whose time range lies fully inside [0, pause_start].
`pause_end` is never touched by any feature (it is only ever used as a
training LABEL boundary / by the scorer, never as model input).

The features are deliberately language-agnostic (pitch, energy, rhythm,
voicing, turn-context) — no ASR, no lexical content, no language-specific
tuning — because the hidden test set is *mostly Hindi* while we are only
given labelled English + Hindi dev data. Prosodic cues for turn-taking
(falling vs. level/rising pitch, final lengthening, energy decay,
trailing-off voicing) are known to transfer reasonably well across
languages, which is the whole bet of this approach.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10
PITCH_FRAME_MS = 40
FMIN, FMAX = 60.0, 400.0
VOICING_THRESH = 0.30
WINDOW_S = 1.5  # how much trailing context we look at for the "final" stats


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def _frame_starts(n_samples, sr, frame_ms, hop_ms):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if n_samples < fl:
        return np.empty(0, dtype=int), fl, hp
    n = 1 + (n_samples - fl) // hp
    return hp * np.arange(n), fl, hp


def _autocorr_f0(frame, sr, fmin=FMIN, fmax=FMAX, voicing_thresh=VOICING_THRESH):
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


class TurnAudio:
    """Precomputes energy + F0 contours ONCE per file (whole-file, causal-safe
    because every downstream feature masks to frames ending <= pause_start).
    """

    def __init__(self, x, sr):
        self.x = x
        self.sr = sr

        starts_e, fl_e, hp_e = _frame_starts(len(x), sr, FRAME_MS, HOP_MS)
        self.e_hop = hp_e
        self.e_frame_end_s = (starts_e + fl_e) / sr  # time each energy frame becomes available
        if len(starts_e):
            fr = x[starts_e[:, None] + np.arange(fl_e)[None, :]]
            rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
            self.energy_db = 20 * np.log10(rms + 1e-12)
        else:
            self.energy_db = np.empty(0, dtype=np.float32)

        starts_f, fl_f, hp_f = _frame_starts(len(x), sr, PITCH_FRAME_MS, HOP_MS)
        self.f_frame_end_s = (starts_f + fl_f) / sr
        if len(starts_f):
            f0 = np.array([_autocorr_f0(x[s:s + fl_f], sr) for s in starts_f], dtype=np.float32)
        else:
            f0 = np.empty(0, dtype=np.float32)
        self.f0 = f0

    def causal_energy(self, t):
        """Energy-frame values fully available by time t."""
        idx = np.searchsorted(self.e_frame_end_s, t, side="right")
        return self.energy_db[:idx]

    def causal_f0(self, t):
        idx = np.searchsorted(self.f_frame_end_s, t, side="right")
        return self.f0[:idx], self.f_frame_end_s[:idx]


def _voiced_runs(f0, hop_ms=HOP_MS):
    """Return list of (start_frame, end_frame) contiguous voiced (f0>0) runs."""
    voiced = f0 > 0
    runs = []
    i = 0
    n = len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def extract_features(turn: TurnAudio, pause_start, pause_index, prior_pause_durs):
    """Feature vector for ONE pause. Only uses turn audio up to pause_start."""
    sr = turn.sr
    feats = {}

    # ---- energy features (causal) ----
    e_all = turn.causal_energy(pause_start)
    if len(e_all) == 0:
        e_all = np.array([-80.0], dtype=np.float32)
    win_frames = max(1, int(WINDOW_S * 1000 / HOP_MS))
    e_win = e_all[-win_frames:]
    tail_n = max(1, int(300 / HOP_MS))       # last ~300ms
    head_n = max(1, int(300 / HOP_MS))       # ~300ms, one "window" back from tail
    e_tail = e_win[-tail_n:]
    e_head = e_win[-(tail_n + head_n):-tail_n] if len(e_win) > tail_n else e_tail

    feats["energy_tail_mean"] = float(np.mean(e_tail))
    feats["energy_decay"] = float(np.mean(e_head) - np.mean(e_tail))  # + => energy dropped into pause
    if len(e_win) >= 2:
        tt = np.arange(len(e_win))
        slope = np.polyfit(tt, e_win, 1)[0]
    else:
        slope = 0.0
    feats["energy_slope"] = float(slope)
    feats["energy_window_std"] = float(np.std(e_win))

    # ---- pitch features (causal) ----
    f0_all, f0_t = turn.causal_f0(pause_start)
    if len(f0_all) == 0:
        f0_all = np.array([0.0], dtype=np.float32)
        f0_t = np.array([0.0], dtype=np.float32)
    f0_win_mask = f0_t >= max(0.0, pause_start - WINDOW_S)
    f0_win = f0_all[f0_win_mask]
    voiced_all = f0_all[f0_all > 0]
    voiced_win = f0_win[f0_win > 0]

    turn_median_pitch = float(np.median(voiced_all)) if len(voiced_all) else 0.0
    final_pitch = float(np.mean(voiced_win[-3:])) if len(voiced_win) >= 1 else 0.0
    feats["final_pitch_norm"] = (final_pitch - turn_median_pitch) if turn_median_pitch > 0 else 0.0
    feats["voicing_ratio_win"] = float(np.mean(f0_win > 0)) if len(f0_win) else 0.0

    if len(voiced_win) >= 3:
        idxv = np.where(f0_win > 0)[0]
        vv = f0_win[idxv]
        slope_p = np.polyfit(np.arange(len(vv)), vv, 1)[0]
    else:
        slope_p = 0.0
    feats["pitch_slope_final"] = float(slope_p)

    # ---- rhythm / final lengthening (causal) ----
    runs = _voiced_runs(f0_all)
    hop_s = HOP_MS / 1000.0
    run_lens = [(b - a) * hop_s for a, b in runs]
    if runs and runs[-1][1] >= len(f0_all) - 2:
        # a voiced run that extends right up to the pause boundary => final lengthening candidate
        final_run_len = run_lens[-1]
    else:
        final_run_len = run_lens[-1] if run_lens else 0.0
    avg_run_len = float(np.mean(run_lens[:-1])) if len(run_lens) > 1 else (run_lens[0] if run_lens else 0.0)
    feats["final_run_len"] = float(final_run_len)
    feats["final_run_vs_avg"] = float(final_run_len - avg_run_len)
    feats["n_voiced_runs"] = float(len(runs))
    feats["speaking_rate"] = float(len(runs) / max(pause_start, 0.3))

    # ---- turn-context features (fully causal: prior pauses already ended) ----
    feats["pause_index"] = float(pause_index)
    feats["pause_start_time"] = float(pause_start)
    feats["n_prior_pauses"] = float(len(prior_pause_durs))
    if prior_pause_durs:
        feats["mean_prior_pause_dur"] = float(np.mean(prior_pause_durs))
        feats["max_prior_pause_dur"] = float(np.max(prior_pause_durs))
    else:
        feats["mean_prior_pause_dur"] = 0.0
        feats["max_prior_pause_dur"] = 0.0

    return feats


FEATURE_NAMES = [
    "energy_tail_mean", "energy_decay", "energy_slope", "energy_window_std",
    "final_pitch_norm", "voicing_ratio_win", "pitch_slope_final",
    "final_run_len", "final_run_vs_avg", "n_voiced_runs", "speaking_rate",
    "pause_index", "pause_start_time", "n_prior_pauses",
    "mean_prior_pause_dur", "max_prior_pause_dur",
]


def featurize_labels_df(rows, data_dir):
    """rows: list of dict rows from labels.csv (already sorted is NOT assumed).
    Returns X (n, len(FEATURE_NAMES)), keys (turn_id, pause_index) list.
    Groups by turn_id, processes pauses within a turn in pause_index order so
    `prior_pause_durs` is built causally.
    """
    import os
    from collections import defaultdict

    by_turn = defaultdict(list)
    for r in rows:
        by_turn[r["turn_id"]].append(r)

    X, keys, y = [], [], []
    cache = {}
    for turn_id, trows in by_turn.items():
        trows = sorted(trows, key=lambda r: int(r["pause_index"]))
        audio_file = trows[0]["audio_file"]
        path = os.path.join(data_dir, audio_file)
        if path not in cache:
            x, sr = load_wav(path)
            cache[path] = TurnAudio(x, sr)
        turn = cache[path]

        prior_durs = []
        for r in trows:
            ps = float(r["pause_start"])
            feats = extract_features(turn, ps, int(r["pause_index"]), list(prior_durs))
            X.append([feats[n] for n in FEATURE_NAMES])
            keys.append((r["turn_id"], r["pause_index"]))
            if "label" in r and r["label"] is not None:
                y.append(1 if r["label"] == "eot" else 0)
            # this pause has now fully happened -> its duration is causal knowledge
            # for any LATER pause in the same turn
            prior_durs.append(float(r["pause_end"]) - ps)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64) if y else None
    return X, keys, y

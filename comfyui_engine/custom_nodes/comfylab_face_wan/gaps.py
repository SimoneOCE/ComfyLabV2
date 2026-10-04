"""Bridges short detection gaps in a tracker's paste weights (both engines).

H3FaceTrackCrop's per-frame paste weight is "was a face detected this frame",
gaussian-smoothed (window 11) per shot, so the composite fades out where the
face turns away. On small faces the detector also just misses a frame or
two now and then. Each miss dips the weight to ~0.4-0.8 for a few frames,
blending the redrawn face with the original underneath: a ghost that pulses
in and out (Alpha Timber family shot, cba2f8ef: the daughter missed frames
268, 311-312 and 314).

A gap of at most MAX_GAP frames with a detection on both sides, inside one
shot, is treated as detected: the tracker interpolated the box across it,
and a face that's there on both sides of a 1/3-second gap is still there.
Weights are only ever raised, never lowered, so the pack's fades at real
dropouts (turned away, left the shot) and at shot edges are unchanged.
"""

import numpy as np

MAX_GAP = 8          # frames (1/3 s at 24fps)
WINDOW = 11          # the pack's weight smoothing: max(9, smooth_window // 2) with smooth_window 21, made odd


def _smooth(vals, window=WINDOW):
    """The pack's gaussian _smooth (sigma = window / 6, reflected edges)."""
    window = min(int(window), len(vals))
    if window % 2 == 0:
        window += 1
    if window < 3 or len(vals) < 3:
        return vals.astype(np.float64)
    pad = window // 2
    padded = np.pad(vals.astype(np.float64), pad, mode="reflect")
    x = np.arange(window, dtype=np.float64) - pad
    kernel = np.exp(-(x ** 2) / (2.0 * max(window / 6.0, 0.5) ** 2))
    return np.convolve(padded, kernel / kernel.sum(), mode="valid")[: len(vals)]


def bridge(transform, max_gap=MAX_GAP):
    """Returns (transform, frames bridged). The transform is a copy with raised
    weights if any gap was bridged, else the same object."""
    detected = transform.get("detected")
    weights = transform.get("weights")
    if not detected or not weights or len(detected) != len(weights):
        return transform, 0
    n = len(detected)
    det = np.array(detected, dtype=bool)
    absent = np.array(transform.get("absent") or [False] * n, dtype=bool)
    filled = det & ~absent
    segs = [(int(a), min(int(b), n)) for a, b in (transform.get("segments") or [(0, n)])]
    bridged = 0
    for a, b in segs:
        k = a
        while k < b:
            if filled[k]:
                k += 1
                continue
            end = k
            while end < b and not filled[end]:
                end += 1
            # [k, end) is a gap; bridge it only if short, inside the shot, with
            # a detection on both sides and no frame where the person is absent.
            if (k > a and end < b and end - k <= max_gap
                    and not absent[k:end].any() and det[k - 1] and det[end]):
                filled[k:end] = True
                bridged += end - k
            k = end
    if not bridged:
        return transform, 0
    smooth = np.empty(n, dtype=np.float64)
    smooth[:] = filled.astype(np.float64)
    for a, b in segs:
        if b > a:
            smooth[a:b] = _smooth(filled[a:b].astype(np.float64))
    raised = np.maximum(np.array(weights, dtype=np.float64), np.clip(smooth, 0.0, 1.0))
    out = dict(transform)
    out["weights"] = [float(w) for w in raised]
    return out, bridged

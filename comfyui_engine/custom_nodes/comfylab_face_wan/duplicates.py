"""Stops two trackers redrawing the same face (both engines).

The people count is "most small faces on screen together in any one shot",
so a brief false detection can make it one too high. The extra person's
tracker then has no face of its own and latches onto a real one, which gets
redrawn and pasted twice (job 3b73aab9: Lord Farquaad, both "people"
175 frames, face 27-111px, identical tracks).

Rule, per later person against every earlier one, on the source frames where
BOTH trackers actually detected a face: same face that frame if the face
centres are closer than half a face height (two different people side by
side are at least a face width apart). Same face on more than half of those
frames -> the later person is dropped: not redrawn, nothing pasted. Fewer ->
kept, but nothing is pasted on the frames where it sat on the other face.

Crop boxes are (x, y, w, h) in source pixels, centred on the face; the face
height is h / crop_factor (see ComfyUI-H3-FaceRefine's H3FaceTrackCrop).
"""

import numpy as np

SAME_FACE = 0.5      # centres closer than this many face heights = the same face
DROP_SHARE = 0.5     # same face on more than this share of shared frames = duplicate


def _faces_by_source(transform):
    """{source frame: (cx, cy, face height, crop index)} for detected frames."""
    boxes = transform.get("boxes") or []
    n = len(boxes)
    source = list(transform.get("source") or range(n))
    detected = list(transform.get("detected") or [True] * n)
    absent = list(transform.get("absent") or [False] * n)
    cf = float(transform.get("crop_factor", 3.0)) or 3.0
    out = {}
    for k in range(min(n, len(source))):
        if not detected[k] or absent[k]:
            continue
        x, y, w, h = (float(v) for v in boxes[k])
        out[int(source[k])] = (x + w / 2.0, y + h / 2.0, h / cf, k)
    return out


def find_duplicates(transforms):
    """transforms: {person: transform}. Returns (dropped persons,
    {person: crop indices to leave unpasted}, report lines)."""
    order = sorted(transforms)
    faces = {i: _faces_by_source(transforms[i]) for i in order}
    dropped, unpasted, report = set(), {}, []
    for pos, j in enumerate(order):
        for i in order[:pos]:
            if i in dropped:
                continue
            shared = sorted(set(faces[i]) & set(faces[j]))
            if not shared:
                continue
            same = []
            for f in shared:
                ax, ay, ah, _ = faces[i][f]
                bx, by, bh, kj = faces[j][f]
                if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < SAME_FACE * min(ah, bh):
                    same.append(kj)
            if not same:
                continue
            share = len(same) / len(shared)
            if share > DROP_SHARE:
                dropped.add(j)
                report.append(f"person {j} follows person {i}'s face on {len(same)}/{len(shared)} "
                              f"shared frames ({share:.0%}) - duplicate, not redrawn")
                break
            unpasted.setdefault(j, set()).update(same)
            report.append(f"person {j} sits on person {i}'s face on {len(same)}/{len(shared)} "
                          f"shared frames - nothing pasted on those")
    return dropped, {j: sorted(v) for j, v in unpasted.items() if j not in dropped}, report


def apply(work):
    """work: [(person, crops, strength, transform)] as both redraw nodes build
    it. Returns (work without dropped persons, strengths zeroed where a kept
    person sat on another's face, {dropped person: transform with zero paste
    weight}, report lines)."""
    dropped, unpasted, report = find_duplicates({i: t for i, _, _, t in work})
    kept, silenced = [], {}
    for i, crops, strength, transform in work:
        if i in dropped:
            t = dict(transform)
            t["weights"] = [0.0] * len(t.get("weights") or transform.get("boxes") or [])
            silenced[i] = t
            continue
        if i in unpasted:
            strength = np.array(strength, dtype=np.float64, copy=True)
            idx = [k for k in unpasted[i] if k < len(strength)]
            strength[idx] = 0.0
        kept.append((i, crops, strength, transform))
    return kept, silenced, report

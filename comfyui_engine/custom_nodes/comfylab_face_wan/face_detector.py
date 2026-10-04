"""Runs the face detector at 1280px instead of its 640px default.

ComfyUI-H3-FaceRefine's H3 Load Video + Face Select calls the detector
(face_yolov8m) with ultralytics' defaults, which shrink every frame to 640px
wide first. Our frames are 1344 wide, so a 23px face reaches the detector at
~11px and is missed now and then - the smallest face in a shot flickers back
to its unfixed self on those frames (the girl in cba2f8ef's last shot). 960
(~16px) cut her missed frames from 6 to 1 at no measurable time cost
(face finding 8.5-13s vs 8.6-12s at 640), but also picked up a 16px
"face" seen on ~5 frames that the refine counts as a 4th person. 1280
(~22px, near her real size) at the user's request (2026-10-04) - watch for
more such extra faces.

Done by wrapping the pack's _load_detector (a module global, so H3FaceSelect
picks it up) and setting imgsz in the returned model's overrides, which
ultralytics merges into every predict call. The pack's code is unchanged.
"""

import logging

DETECTOR_IMGSZ = 1280


def install(pack):
    load_detector = getattr(pack, "_load_detector", None)
    if load_detector is None or getattr(load_detector, "_comfylab_imgsz", False):
        return

    def sized_load_detector(name, *a, **k):
        model = load_detector(name, *a, **k)
        overrides = getattr(model, "overrides", None)
        if isinstance(overrides, dict) and overrides.get("imgsz") != DETECTOR_IMGSZ:
            overrides["imgsz"] = DETECTOR_IMGSZ
            print(f"[ComfyLabFaceDetector] {name}: detecting at {DETECTOR_IMGSZ}px", flush=True)
            logging.info(f"[ComfyLabFaceDetector] {name}: detecting at {DETECTOR_IMGSZ}px")
        return model

    sized_load_detector._comfylab_imgsz = True
    pack._load_detector = sized_load_detector

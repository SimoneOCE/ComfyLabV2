"""Runs the face detector at 960px instead of its 640px default.

ComfyUI-H3-FaceRefine's H3 Load Video + Face Select calls the detector
(face_yolov8m) with ultralytics' defaults, which shrink every frame to 640px
wide first. Our frames are 1344 wide, so a 23px face reaches the detector at
~11px and is missed now and then - the smallest face in a shot flickers back
to its unfixed self on those frames (the girl in cba2f8ef's last shot). At
960 it's ~16px, where this detector is reliable. Detection runs once per
refine; the step goes from ~10s to ~15-20s for a 15s clip.

Done by wrapping the pack's _load_detector (a module global, so H3FaceSelect
picks it up) and setting imgsz in the returned model's overrides, which
ultralytics merges into every predict call. The pack's code is unchanged.
"""

import logging

DETECTOR_IMGSZ = 960


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

"""Environment probe for the vision frontend (§54).

`kinemimic vision-doctor` prints, in one screen, what this machine can
run today: GPU availability, optional vision dependencies, and the
recommended backend/mode combination. No silent fallbacks — if a backend
needs CUDA and CUDA is absent, the doctor says so explicitly.
"""

from __future__ import annotations


def _probe() -> dict:
    import importlib.util

    def has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    info = {
        "torch": has("torch"),
        "cuda": False,
        "cv2": has("cv2"),
        "ultralytics": has("ultralytics"),
        "rfdetr": has("rfdetr"),
        "scipy": has("scipy"),
        "device": "cpu",
    }
    if info["torch"]:
        try:
            import torch
            info["cuda"] = bool(torch.cuda.is_available())
            if info["cuda"]:
                info["device"] = torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info["device"] = "apple-mps"
        except Exception:
            pass
    return info


def run_doctor() -> None:
    i = _probe()
    line = lambda k, v: print(f"  {k:<14} {v}")

    print("KineMimic vision doctor")
    print("=" * 56)
    line("torch", "installed" if i["torch"] else "NOT installed")
    line("CUDA", "available — modern detectors will use GPU"
         if i["cuda"] else
         "not available — detectors run on CPU (slow for large models)")
    line("device", i["device"])
    line("opencv", "installed" if i["cv2"] else "NOT installed")
    line("ultralytics", "installed — YoloBackend ready"
         if i["ultralytics"] else "not installed (pip install ultralytics)")
    line("rfdetr", "installed — benchmark candidate ready"
         if i["rfdetr"] else "not installed (optional; pip install rfdetr)")
    line("scipy", "installed" if i["scipy"] else "NOT installed")
    print("=" * 56)

    legacy_ok = i["cv2"]
    yolo_ok = i["ultralytics"]

    print("recommended:")
    if legacy_ok:
        print("  quick screening (CPU) : kinemimic vision VIDEO --id ID "
              "--mode fast --detector legacy")
    if yolo_ok and i["cuda"]:
        print("  research data (GPU)   : kinemimic vision VIDEO --id ID "
              "--mode accurate --detector yolo --tile 1024")
        print("                          (use KineMimic-trained weights — "
              "docs/VISION_DATASET.md)")
    if yolo_ok and not i["cuda"]:
        print("  yolo on CPU           : works but slow; prefer legacy for "
              "long videos, yolo for short clips")
    if not yolo_ok and i["cuda"] is False:
        pass
    print("  note: stock COCO weights cannot detect ants — quality gate "
          "will reject such runs; see docs/VISION_DATASET.md")
    if legacy_ok:
        print("  benchmark             : kinemimic vision-benchmark")

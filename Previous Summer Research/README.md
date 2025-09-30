# 🎱 PoolVision — Billiards Ball Detection & Table Coordinates

**PoolVision** is a Python-based computer vision system for **detecting pool balls** in a live camera stream and converting their image locations to **real‑world table coordinates (X, Y)** using a calibrated planar homography.

> Senior Design — Fall 2025 (Clemson University)

---

## ✨ What it does

* **Detects balls** in the video stream (Ultralytics YOLO; trained on billiards classes).
* **Converts pixels → table units** (mm or inches) via homography from a one-time table calibration.
* **Identifies ball numbers** (1–15) + **cue ball**, with class/ID overlays and confidences.
* **Outputs structured data** (CSV/JSON + optional UDP socket) with time-stamped ball positions.
* Optional: **simple tracking**/ID smoothing to stabilize detections across frames.

---

## 🗂 Repository Map

```
autocalibration.py             # Auto-calibration from a single photo using AprilTags at 4 corners
manual_calibration.py          # Manual calibration: click 4 known table corners (recommended)
calibration_board_generator.py # Generates printable AprilTag corner markers for the table
detect_cameras.py              # Lists available camera indices
graphical_interface.py         # Experimental GUI (preview, model swap, capture calibration image)
main.py                        # Core pipeline (run after calibration)
models/                        # Put your YOLO *.pt model(s) here
```

---

## 📦 Installation

**Python 3.10+** recommended. Create/activate a virtual environment, then:

```bash
pip install ultralytics opencv-python numpy pillow
```

For AprilTag detection (choose **one** package that works best on your platform):

```bash
# Option A (RobotPy bindings)
pip install robotpy-apriltag

# Option B (native bindings)
pip install apriltag

# Option C (Pupil Labs)
pip install pupil-apriltags
```

> On Windows, device indices can look unusual (e.g., 700/1400). Use `python detect_cameras.py` to list cameras, then set `CAM_INDEX` in `main.py`.

---

## 🚀 Quick Start

### 1) Calibrate the Table (do this once per camera setup)

You have **two** options. In both, you’ll define the table’s real‑world rectangle so we can map pixels → table coordinates.

**Option A — AprilTag corners (auto)**

1. Print four square AprilTags (family `tag36h11`) sized **60–100 mm** (or **2.5–4 in**) using `calibration_board_generator.py`.
2. Tape one tag **flush** to each **inner table corner** (top‑left ID **0**, top‑right **1**, bottom‑right **2**, bottom‑left **3**). Keep tags flat and visible.
3. Take a clear top‑down-ish photo where all 4 tags are in view.
4. Run:

   ```bash
   python autocalibration.py
   ```

   This computes the planar homography and writes `H.npy` (+ `H_inv.npy`).

**Option B — Manual 4‑click (recommended if you can see all four table corners)**

1. Capture a **PNG** snapshot of the table (avoid JPEG scaling artifacts).
2. Run:

   ```bash
   python manual_calibration.py
   ```
3. Click the table’s inner‑cushion corners **in order**: **top‑left → top‑right → bottom‑right → bottom‑left**.
   Shortcuts: **`u`** undo, **`r`** reset, **`s`**/**Enter** save. Outputs `H.npy` (+ `H_inv.npy`).
4. In the prompt (or config), enter the **real table size** (e.g., 8‑ft table playing field ≈ **88 in × 44 in**). Units are up to you; pick **mm** or **in** and stay consistent.

> If your camera or table moves, **re‑run calibration**. Mount the camera securely (overhead preferred) to avoid drift.

### 2) Run the Pipeline

```bash
python main.py
```

* Loads `H.npy`, your YOLO model, and streams detections with **class/ball‑number, confidence, and (X, Y)** in table units.
* Writes a rolling **`runs/latest/positions.csv`** (timestamp, id, class, x, y, conf, frame).
* (Optional) Emits a lightweight **UDP** message per frame for downstream consumers (enable in config).

---

## 🔧 Configure (edit in `main.py`)

* **Camera**: `CAM_INDEX = ...` (discover with `detect_cameras.py`).
* **Model**: `YOLO_MODEL_PATH = "models/billiards_model.pt"`.
* **Labels**: `ALLOWED_LABELS` (e.g., `{"cue","1","2",...,"15"}`) or `None` for all.
* **Thresholds**: `CONF_THRESH = 0.35`, `IOU_THRESH = 0.45` (tune per lighting/camera).
* **Homography**: `H_FILE = "H.npy"` (produced by calibration).
* **Units**: `UNITS = "mm"` or `"in"`.
* **Output**: `WRITE_CSV = True`; `CSV_PATH = "runs/latest/positions.csv"`; optional `UDP_ENABLE`, `UDP_HOST`, `UDP_PORT`.
* **Smoothing** (optional): enable a simple tracker or EMA on centroids to reduce jitter.

---

## 🖼 Optional GUI

This is experimental — for quick peeks only. Prefer `main.py` for production.

```bash
python graphical_interface.py
```

* Preview camera, switch YOLO models, generate/save AprilTag markers, capture a calibration image.

---

## 🧪 Troubleshooting

* **Wrong coordinates**

  * Ensure `H.npy` matches the current camera pose and table.
  * Verify click order (manual) or tag IDs/visibility (auto).
  * Confirm your **table dimensions** and **units** are correct/consistent.
* **Missed/mislabelled balls**

  * Increase lighting and reduce glare (matte lights, polarizing filter can help).
  * Raise `CONF_THRESH` for precision or lower it to catch faint detections.
  * Retrain/fine‑tune YOLO on your exact cloth color/lighting if needed.
* **Blurry frames / motion blur**

  * Use higher shutter speed / more light. Mount the camera rigidly.
* **Camera won’t open**

  * Run `python detect_cameras.py` and select a valid index. Close other apps using the camera.

---

## 🤝 Contributing

Pull requests welcome! For bug reports, include:

* OS, Python version, and pip package versions
* Steps to reproduce (short screen recording helps)
* Logs and relevant config snippets

---

## 📜 License

Add a license (e.g., **MIT** or **Apache‑2.0**) in `LICENSE`.
Without a license, *all rights are reserved by default*.

---

## ⚠️ Notes

* Mount the camera securely and re‑calibrate after any movement.
* If you switch units (mm/in), also update any downstream tooling that consumes the CSV or UDP messages.

---

## Acknowledgments

* Clemson University — Senior Design (Fall 2025)

---

## Contact

Questions or issues: **[ejo@clemson.edu](mailto:ejo@clemson.edu)**
Bug reports: please use your project’s GitHub Issues.

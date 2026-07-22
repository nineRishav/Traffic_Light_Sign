# 🚦 Traffic Light & Sign Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Supported-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

<div align="center">
  <img src="assets/basler_demo.gif" alt="Traffic Light Detection Demo" width="100%">
</div>

---

## 🔬 Tested on S2TLD (Small Traffic Light Dataset)

The pipeline was validated against images from the **S2TLD Dataset** (SJTU), which features real-world urban scenes with small, distant, and heavily occluded traffic lights — one of the hardest benchmarks for traffic light detection.

<div align="center">
  <img src="assets/s2tld_demo.gif" alt="S2TLD Dataset Detection Demo" width="100%">
  <p><i>Animated inference on S2TLD frames. The model correctly identifies light states despite the small pixel footprint and busy urban background.</i></p>
</div>

| 🔴 Red | 🟢 Green | 🟡 Yellow |
|:------:|:--------:|:---------:|
| <img src="assets/s2tld_red.jpg" width="300"> | <img src="assets/s2tld_green.jpg" width="300"> | <img src="assets/s2tld_yellow.jpg" width="300"> |

---

Real-time and offline detection of traffic light states (Red / Yellow / Green / Off) and traffic sign speed limits (via YOLO + OCR). 

Originally developed as vehicle-testing prototypes coupled to a Basler industrial camera and ROS, this repository is a refined, hardware-agnostic version. It has no ROS or camera-SDK dependencies—the scripts seamlessly accept a **plain image**, a **video file**, or a **webcam index** as input.

---

## 🧠 How It Works (The Pipeline)

### 🚥 Traffic Light Detection Pipeline
Running raw YOLO detection frame-by-frame on video is notoriously jittery. Objects get partially occluded, confidence scores fluctuate, and false positives appear for a split second. This pipeline solves that by implementing an **IoU-based Tracker (`LightTracker`)**.

```mermaid
graph TD
    A[Video Frame / Image] --> B[YOLOv8 Inference]
    B -->|Bounding Boxes & Confidences| C{Cross-Class NMS}
    C -->|Filter overlapping states| D[IoU LightTracker]
    D -->|Match with previous frames| E{Detected for >3 frames?}
    E -- Yes --> F[Commit New State]
    E -- No --> G[Hold Previous State]
    F --> H[Draw Colored Bounding Box]
    G --> H
```
*The system accumulates "evidence" over time (e.g., 3 consecutive frames) before changing the state of a traffic light, mimicking how a human brain processes continuous video to result in a buttery-smooth output!*

### 🛑 Traffic Sign & OCR Pipeline
This pipeline combines YOLO for sign detection with EasyOCR and intensive preprocessing for reading speed limits.

```mermaid
graph TD
    A[Video Frame / Image] --> B[YOLOv8 Inference]
    B -->|Detected 'Speed Limit' Signs| C[Crop Sign Image]
    C --> D[OCR Preprocessing]
    D -->|Upscale, CLAHE, Denoise, Deskew| E[EasyOCR Inference]
    E -->|Read Digits| F{Whitelist Validation}
    F -- "Valid Speed (20, 40, 60...)" --> G[Evidence Accumulator]
    F -- "Invalid / Noise" --> H[Discard Reading]
    G -->|Confirmed across frames?| I[Draw Speed Limit Bounding Box]
```

---

## ✨ Features

### 🚥 Traffic Light Detection (`traffic_light_detection.py`)
- **State Recognition:** YOLO-based 4-class traffic light state detection (Red, Yellow, Green, Off).
- **Cross-Class NMS:** Class-agnostic Non-Maximum Suppression ensures only one state can be reported per physical signal—preventing impossible states (like red and green simultaneously).
- **Temporal Tracking (`LightTracker`):** A lightweight IoU-based tracker provides temporal confirmation drastically reducing single-frame flickering or misclassification.
- **Dynamic Output Saving:** When saving outputs, it automatically creates unique run directories (e.g., `output_name_1`) containing the full `.mp4` video, an `animated .gif` summary, and a `frames/` folder containing every individual annotated frame.

### 🛑 Traffic Sign & Speed Limit OCR (`traffic_sign_speed_detection.py`)
- **Sign Detection & OCR:** YOLO-based traffic sign detection coupled with an EasyOCR pipeline to read speed limit digits.
- **Robust Preprocessing:** Full OCR prep-pipeline including upscaling, CLAHE contrast enhancement, adaptive thresholding, denoising, perspective correction, and deskewing.
- **Real-World Validation:** Whitelist validation against actual legal speed values (e.g., `20/30/40/60/70/80`).
- **Evidence Accumulation:** Speed readings are temporally tracked and only committed once `--min-agree` OCR readings agree across multiple frames. Unconfirmed readings expire after `--expire-frames`.
- **Graceful Degradation:** OCR is entirely optional. If `easyocr` isn't installed, the script falls back to bounding-box sign detection only.

---

## 🛠️ Installation

It is recommended to use a Conda environment to manage dependencies cleanly.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Traffic-Light-And-Sign-Detection.git
cd Traffic-Light-And-Sign-Detection

# 2. Create and activate a Conda environment
conda create -n traffic_light python=3.11
conda activate traffic_light

# 3. Install dependencies
pip install -r requirements.txt
```

### 🧠 Model Weights Setup
Place your trained YOLO weights (`.pt` files) in the `models/` directory or a `weights/` directory:
- `traffic_light.pt` — 4-class traffic light model.
- `traffic_sign.pt` — Traffic sign model (must include a `speed limit-m` class).

*(Note: Model weights are not committed to this repo due to size constraints. Distribute them via GitHub Releases or Git LFS).*

---

## 🚀 Usage

### 🚦 Traffic Light Detection

```bash
# Single image (Default Resolution is 832, Conf is 0.50)
python src/traffic_light_detection.py --source examples/frame.jpg --model weights/traffic_light.pt --show

# Video file, save annotated output (Creates a unique folder with video, frames, and a GIF)
python src/traffic_light_detection.py --source examples/drive.mp4 --model weights/traffic_light.pt --save output_light

# Live Webcam
python src/traffic_light_detection.py --source 0 --model weights/traffic_light.pt --show
```

### 🛑 Traffic Sign / Speed-Limit OCR

```bash
# Single image
python src/traffic_sign_speed_detection.py --source examples/frame.jpg --sign-model weights/traffic_sign.pt --show

# Video file, save annotated output
python src/traffic_sign_speed_detection.py --source examples/drive.mp4 --sign-model weights/traffic_sign.pt --save output_sign
```

### ⚙️ Key CLI Arguments (Both Scripts)
- `--conf`: Detection confidence threshold (Default: `0.50`). Lower this if you want to detect partially occluded lights, but beware of false positives!
- `--imgsz`: Inference resolution (Default: `832`).
- `--device`: Target compute device (`cpu`, `0` for CUDA, etc.).
- `--save`: Base name/path for the output directory.
- `--show`: Display the live annotated output in an OpenCV window.
- `--confirm-frames`: (Traffic Light) Consecutive frames a state must persist before acceptance (Default: `3`).
- `--min-agree`, `--expire-frames`, `--no-ocr`: (Traffic Sign) Fine-tuning for the OCR evidence accumulation.

---

## 🏗️ Repository Structure

```text
Traffic-Light-And-Sign-Detection/
├── traffic_light_detection.py       # Main script for traffic lights
├── traffic_sign_speed_detection.py  # Main script for traffic signs
├── requirements.txt
├── README.md
├── changes.md                       # Internal changelog
├── data/                            # Sample input images/videos
└── weights/                         # Directory for your trained .pt weights
```

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

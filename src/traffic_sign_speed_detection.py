#!/usr/bin/env python3
"""
traffic_sign_speed_detection.py

Traffic sign detection + speed-limit OCR using YOLO and EasyOCR.

Redesigned from the original Basler + ROS-coupled prototype:
  - No ROS, no cv_bridge, no pypylon/Basler dependency. Input is a plain
    image file, a video file, or a webcam index -- anything OpenCV's
    VideoCapture (or imread) can open.
  - Processes every detected speed-limit box per frame instead of only
    the first one found.
  - Adds track-based OCR evidence accumulation: each physical sign is
    followed across frames (simple IoU tracking) and a reading is only
    committed once enough consecutive/aggregate readings agree, instead
    of trusting a single OCR call.
  - Persisted readings now expire after a configurable number of frames
    without reconfirmation, instead of being shown forever.

Usage:
    python traffic_sign_speed_detection.py --source path/to/image.jpg
    python traffic_sign_speed_detection.py --source path/to/video.mp4 --save out.mp4
    python traffic_sign_speed_detection.py --source 0   # webcam
"""

import argparse
import collections
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VALID_SPEEDS = {"20", "30", "40", "60", "70", "80"}
SPEED_LIMIT_CLASS_NAME = "speed limit-m"

CLASS_NAMES_SIGN = [
    "traffic signal-c", "u-turn prohibited-m", "stop-m", "round about-c", "bus stop-I",
    "informatory board-I", "t-Junction-c", "right Reverse Bend-c", "right Hand Curve-c",
    "no Entry-m", "give way-m", "pedestrian Crossing-c", "no Parking-m", "speed limit-m",
    "compulsary turn left ahead-m", "restrictions ends-m", "horn prohibited-m",
    "side road left-c", "side road right-c", "school-zone-c", "speed bump-c",
    "right turn prohibited-m", "left hand curve-c", "u-turn", "children at play-c",
    "y-junction-c", "parking lot bike-I", "parking lot car-I", "parking lot cycles-I",
    "parking lot-I", "hump or rough road-c", "chevron sign right-w",
    "national highway route marker-I", "chevron sign left-w", "gap in median-c",
    "school ahead-c", "cross road-c", "merging traffic from left-c", "go slow-c",
    "overtaking prohibited-m", "narrow road ahead-c", "staggered intersection-c",
    "men at work-c", "two way-c", "accident prone zone-c", "compulsory keep left-m",
    "truck prohibited-m", "left turn prohibited-m", "direction sign-I",
    "no stopping or no standing-m", "one way-m", "guarded level crossing-c",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Traffic sign detection + speed-limit OCR.")
    ap.add_argument("--source", required=True,
                     help="Path to an image, path to a video file, or a webcam index (e.g. 0).")
    ap.add_argument("--sign-model", default="models/traffic_sign.pt", help="Path to the YOLO sign-detector .pt weights.")
    ap.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.5, help="Detection IoU threshold for NMS.")
    ap.add_argument("--device", default=None, help="Torch device, e.g. 'cpu', '0', 'cuda'.")
    ap.add_argument("--min-agree", type=int, default=2,
                     help="Minimum agreeing OCR readings on a track before it's committed/displayed.")
    ap.add_argument("--iou-match", type=float, default=0.3, help="IoU threshold used to match sign tracks across frames.")
    ap.add_argument("--expire-frames", type=int, default=90,
                     help="Frames a committed reading is kept after its track disappears (~3s at 30fps).")
    ap.add_argument("--no-ocr", action="store_true", help="Disable OCR even if EasyOCR is installed.")
    ap.add_argument("--save", default=None, help="Optional output path for annotated image/video.")
    ap.add_argument("--show", action="store_true", help="Display the annotated output in a window.")
    return ap.parse_args()


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)


# ---------------------------------------------------------------------------
# OCR preprocessing pipeline (unchanged in spirit from the original script)
# ---------------------------------------------------------------------------

def upscale_and_preprocess(cropped_sign):
    upscaled = cv2.resize(cropped_sign, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.medianBlur(thresh, 3)
    return denoised


def correct_perspective(image):
    """Approximate perspective correction. Falls back to the raw image if no
    clear quadrilateral is found; this is not a full corner-detection
    homography, only a light stabilizing step ahead of deskewing."""
    h, w = image.shape[:2]
    pts_src = np.array([[10, 10], [w - 10, 10], [w - 10, h - 10], [10, h - 10]], dtype="float32")
    pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(image, M, (w, h))


def deskew_image(binary_image):
    coords = np.column_stack(np.where(binary_image > 0))
    if len(coords) == 0:
        return binary_image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = binary_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(binary_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def validate_speed(detected_text):
    cleaned_text = detected_text.replace("O", "0").replace("o", "0")
    digits = "".join(filter(str.isdigit, cleaned_text))
    return digits if digits in VALID_SPEEDS else "NA"


def read_speed_sign(cropped_sign, ocr_reader):
    rotated = cv2.rotate(cropped_sign, cv2.ROTATE_90_COUNTERCLOCKWISE)
    corrected = correct_perspective(rotated)
    preprocessed = upscale_and_preprocess(corrected)
    deskewed = deskew_image(preprocessed)
    ocr_results = ocr_reader.readtext(deskewed)
    if not ocr_results:
        return "NA", 0.0
    text, conf = ocr_results[0][1], ocr_results[0][2]
    return validate_speed(text), conf


# ---------------------------------------------------------------------------
# Track-based evidence accumulation
# ---------------------------------------------------------------------------

class SignTrack:
    __slots__ = ("track_id", "box", "misses", "readings", "committed_speed", "ttl")

    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box
        self.misses = 0
        self.readings = collections.Counter()
        self.committed_speed = None
        self.ttl = 0

    def add_reading(self, speed, min_agree, expire_frames):
        if speed != "NA":
            self.readings[speed] += 1
            top_speed, top_count = self.readings.most_common(1)[0]
            if top_count >= min_agree:
                self.committed_speed = top_speed
        if self.committed_speed is not None:
            self.ttl = expire_frames


class SignTracker:
    def __init__(self, iou_match=0.3, min_agree=2, expire_frames=90, max_misses=10):
        self.tracks = []
        self._next_id = 0
        self.iou_match = iou_match
        self.min_agree = min_agree
        self.expire_frames = expire_frames
        self.max_misses = max_misses

    def match_or_create(self, box):
        best_iou, best_track = 0.0, None
        for t in self.tracks:
            score = iou(t.box, box)
            if score > best_iou:
                best_iou, best_track = score, t
        if best_track is not None and best_iou >= self.iou_match:
            best_track.box = box
            best_track.misses = 0
            return best_track
        t = SignTrack(self._next_id, box)
        self._next_id += 1
        self.tracks.append(t)
        return t

    def tick(self):
        for t in self.tracks:
            t.misses += 1
            if t.ttl > 0:
                t.ttl -= 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

    def active_display_speed(self):
        """Best current committed speed across all live/recently-seen tracks."""
        best = None
        for t in self.tracks:
            if t.committed_speed is not None and t.ttl > 0:
                best = t.committed_speed
        return best


def annotate_box(img, x1, y1, x2, y2, label, color=(255, 0, 255)):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def process_frame(model, ocr_reader, tracker, frame, args):
    results = model.predict(source=frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False)

    seen_track_ids = set()
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = CLASS_NAMES_SIGN[cls] if cls < len(CLASS_NAMES_SIGN) else "unknown"

            if class_name != SPEED_LIMIT_CLASS_NAME:
                continue

            track = tracker.match_or_create((x1, y1, x2, y2))
            seen_track_ids.add(track.track_id)

            speed_label = "reading..."
            if ocr_reader is not None:
                margin = 20
                x1_m = max(0, x1 - margin)
                y1_m = max(0, y1 - margin)
                x2_m = min(frame.shape[1], x2 + margin)
                y2_m = min(frame.shape[0], y2 + margin)
                cropped = frame[y1_m:y2_m, x1_m:x2_m]
                if cropped.size > 0:
                    speed, conf = read_speed_sign(cropped, ocr_reader)
                    track.add_reading(speed, args.min_agree, args.expire_frames)
                    if track.committed_speed:
                        speed_label = f"{track.committed_speed} km/h"

            annotate_box(frame, x1, y1, x2, y2, f"#{track.track_id} {speed_label}")

    tracker.tick()
    display_speed = tracker.active_display_speed()

    label = f"Maintain Speed Limit of {display_speed}" if display_speed else "Maintain Speed Limit of NA"
    cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    return frame


def get_unique_save_path(save_path):
    if not save_path:
        return save_path
    base, ext = os.path.splitext(save_path)
    counter = 1
    new_path = f"{base}_{counter}{ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    return new_path


def main():
    args = parse_args()
    if args.save:
        args.save = get_unique_save_path(args.save)

    if not os.path.exists(args.sign_model):
        print(f"WARNING: model weights not found at '{args.sign_model}'. "
              f"Point --sign-model at your trained sign-detector .pt file.", file=sys.stderr)

    model = YOLO(args.sign_model)

    ocr_reader = None
    if not args.no_ocr:
        if OCR_AVAILABLE:
            ocr_reader = easyocr.Reader(["en"], gpu=(args.device not in (None, "cpu")))
            print("Speed-limit OCR initialized.")
        else:
            print("WARNING: EasyOCR not installed; running detection only, OCR disabled.", file=sys.stderr)

    tracker = SignTracker(iou_match=args.iou_match, min_agree=args.min_agree, expire_frames=args.expire_frames)

    source = args.source
    is_image = os.path.splitext(str(source))[1].lower() in IMAGE_EXTS if os.path.exists(str(source)) else False

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        out = process_frame(model, ocr_reader, tracker, frame, args)
        if args.save:
            cv2.imwrite(args.save, out)
            print(f"Saved annotated image to {args.save}")
        if args.show:
            cv2.imshow("Speed Sign OCR", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            out = process_frame(model, ocr_reader, tracker, frame, args)

            if args.save:
                if writer is None:
                    h, w = out.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
                writer.write(out)

            if args.show:
                cv2.imshow("Speed Sign OCR", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if args.save:
        print(f"Saved annotated video to {args.save}")


if __name__ == "__main__":
    main()

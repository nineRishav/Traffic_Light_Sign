#!/usr/bin/env python3
"""
traffic_light_detection.py

Real-time / offline traffic light state detection using YOLO.

Redesigned from the original Basler + ROS-coupled prototype:
  - No ROS, no pypylon/Basler dependency. Input is a plain image file,
    a video file, or a webcam index -- anything OpenCV's VideoCapture
    (or imread) can open.
  - Adds class-agnostic ("cross-class") NMS so only one light state can
    win per physical signal location, matching the real-world constraint
    that a traffic light shows exactly one state at a time.
  - Adds a lightweight IoU tracker with temporal confirmation: a light's
    displayed state only changes after it has been seen for N consecutive
    frames, instead of trusting a single frame's classification instantly.

Usage:
    python traffic_light_detection.py --source path/to/image.jpg
    python traffic_light_detection.py --source path/to/video.mp4 --save out.mp4
    python traffic_light_detection.py --source 0                 # webcam
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import imageio
from ultralytics import YOLO

ALL_CLASSES = ["Red-Stop", "Yellow", "Green-Go", "Off"]
CLASS_COLORS = {
    "Red-Stop": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Green-Go": (0, 255, 0),
    "Off": (0, 0, 0),
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    ap = argparse.ArgumentParser(description="Traffic light state detection (YOLO).")
    ap.add_argument("--source", required=True,
                     help="Path to an image, path to a video file, or a webcam index (e.g. 0).")
    ap.add_argument("--model", default="models/traffic_light.pt", help="Path to the YOLO .pt weights.")
    ap.add_argument("--conf", type=float, default=0.50, help="Confidence threshold.")
    ap.add_argument("--imgsz", type=int, default=832, help="Inference size (square).")
    ap.add_argument("--confirm-frames", type=int, default=3,
                     help="Consecutive frames a state must persist on a track before it's accepted.")
    ap.add_argument("--iou-match", type=float, default=0.3, help="IoU threshold used to match tracks across frames.")
    ap.add_argument("--device", default=None, help="Torch device, e.g. 'cpu', '0'. Default: let Ultralytics choose.")
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


class LightTrack:
    """A single physical traffic light followed across frames."""
    __slots__ = ("track_id", "box", "candidate_class", "candidate_streak",
                 "confirmed_class", "misses")

    def __init__(self, track_id, box, cls_name):
        self.track_id = track_id
        self.box = box
        self.candidate_class = cls_name
        self.candidate_streak = 1
        self.confirmed_class = cls_name  # first observation is shown immediately
        self.misses = 0

    def update(self, box, cls_name, confirm_frames):
        self.box = box
        self.misses = 0
        if cls_name == self.candidate_class:
            self.candidate_streak += 1
        else:
            self.candidate_class = cls_name
            self.candidate_streak = 1
        if self.candidate_streak >= confirm_frames:
            self.confirmed_class = self.candidate_class


class LightTracker:
    """Minimal IoU-based multi-object tracker for temporal state confirmation."""

    def __init__(self, iou_match=0.3, confirm_frames=3, max_misses=5):
        self.tracks = []
        self._next_id = 0
        self.iou_match = iou_match
        self.confirm_frames = confirm_frames
        self.max_misses = max_misses

    def step(self, detections):
        """detections: list of (box_xyxy, cls_name, conf). Returns list of tracks with confirmed state."""
        unmatched_dets = list(range(len(detections)))
        for track in self.tracks:
            best_iou, best_j = 0.0, -1
            for j in unmatched_dets:
                box, cls_name, conf = detections[j]
                score = iou(track.box, box)
                if score > best_iou:
                    best_iou, best_j = score, j
            if best_j != -1 and best_iou >= self.iou_match:
                box, cls_name, conf = detections[best_j]
                track.update(box, cls_name, self.confirm_frames)
                unmatched_dets.remove(best_j)
            else:
                track.misses += 1

        for j in unmatched_dets:
            box, cls_name, conf = detections[j]
            self.tracks.append(LightTrack(self._next_id, box, cls_name))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks


def annotate_box(img, x1, y1, x2, y2, cls_name, conf, color):
    label = f"{cls_name} ({conf:.2f})"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_inference(model, frame, imgsz, conf_thres, device):
    """Runs YOLO with class-agnostic NMS so only one state wins per light location,
    and returns detections rescaled to the original frame size."""
    h, w = frame.shape[:2]
    results = model.predict(
        frame, imgsz=imgsz, conf=conf_thres, agnostic_nms=True,
        device=device, verbose=False,
    )[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls < len(ALL_CLASSES):
            cls_name = ALL_CLASSES[cls]
        elif hasattr(model, "names") and model.names and cls in model.names:
            cls_name = model.names[cls]
        else:
            cls_name = f"class_{cls}"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(((x1, y1, x2, y2), cls_name, conf))
    return detections


def process_frame(model, frame, tracker, args):
    detections = run_inference(model, frame, args.imgsz, args.conf, args.device)
    tracks = tracker.step(detections)

    det_by_box = {d[0]: (d[1], d[2]) for d in detections}
    for track in tracks:
        if track.misses > 0:
            continue  # not seen this frame, nothing to draw
        cls_name = track.confirmed_class
        conf = det_by_box.get(track.box, (cls_name, 0.0))[1]
        
        cls_lower = cls_name.lower()
        if "red" in cls_lower:
            color = (0, 0, 255)
        elif "yellow" in cls_lower:
            color = (0, 255, 255)
        elif "green" in cls_lower:
            color = (0, 255, 0)
        elif "off" in cls_lower:
            color = (0, 0, 0)
        else:
            color = CLASS_COLORS.get(cls_name, (128, 128, 128))
            
        x1, y1, x2, y2 = track.box
        annotate_box(frame, x1, y1, x2, y2, cls_name, conf, color)

    cv2.putText(frame, "Traffic Light Detections", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return frame


def setup_run_dir(save_path):
    if not save_path:
        return None, None, None, None
    parent_dir = os.path.dirname(save_path) or "."
    filename = os.path.basename(save_path)
    base_name, ext = os.path.splitext(filename)
    if not ext:
        ext = ".mp4"
    
    counter = 1
    run_dir = os.path.join(parent_dir, f"{base_name}_{counter}")
    while os.path.exists(run_dir):
        counter += 1
        run_dir = os.path.join(parent_dir, f"{base_name}_{counter}")
        
    os.makedirs(run_dir, exist_ok=True)
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    video_path = os.path.join(run_dir, base_name + ext)
    return run_dir, video_path, frames_dir, base_name


def main():
    args = parse_args()

    run_dir, video_path, frames_dir, base_name = None, None, None, None
    if args.save:
        run_dir, video_path, frames_dir, base_name = setup_run_dir(args.save)

    if not os.path.exists(args.model):
        print(f"WARNING: model weights not found at '{args.model}'. "
              f"Point --model at your trained traffic-light .pt file.", file=sys.stderr)

    model = YOLO(args.model)
    tracker = LightTracker(iou_match=args.iou_match, confirm_frames=args.confirm_frames)

    source = args.source
    is_image = os.path.splitext(str(source))[1].lower() in IMAGE_EXTS if os.path.exists(str(source)) else False
    is_dir   = os.path.isdir(str(source))

    # --- Directory of images mode ---
    if is_dir:
        image_paths = sorted([
            os.path.join(source, f) for f in os.listdir(source)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])
        if not image_paths:
            raise FileNotFoundError(f"No images found in directory: {source}")
        out_dir = None
        if args.save:
            out_dir = args.save
            os.makedirs(out_dir, exist_ok=True)
            print(f"Saving annotated images to: {out_dir}")
        total = len(image_paths)
        for idx, img_path in enumerate(image_paths, 1):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[{idx}/{total}] SKIP (unreadable): {os.path.basename(img_path)}")
                continue
            tracker_single = LightTracker(
                iou_match=args.iou_match,
                confirm_frames=1  # single image: confirm immediately
            )
            out = process_frame(model, frame, tracker_single, args)
            if out_dir:
                save_name = os.path.splitext(os.path.basename(img_path))[0] + ".jpg"
                cv2.imwrite(os.path.join(out_dir, save_name), out)
            if idx % 20 == 0 or idx == total:
                print(f"  [{idx}/{total}] processed.")
            if args.show:
                cv2.imshow("Traffic Light", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if args.show:
            cv2.destroyAllWindows()
        print(f"\nDone. {total} images processed.")
        return

    # --- Single image mode ---
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        out = process_frame(model, frame, tracker, args)
        if args.save:
            image_filename = f"{base_name}.jpg"
            image_path = os.path.join(run_dir, image_filename)
            cv2.imwrite(image_path, out)
            print(f"Saved annotated image to {image_path}")
        if args.show:
            cv2.imshow("Traffic Light", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Video file or webcam index
    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    frame_count = 0
    gif_frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            out = process_frame(model, frame, tracker, args)

            if args.save:
                if writer is None:
                    h, w = out.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                writer.write(out)
                
                # Save each individual frame
                frame_save_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_save_path, out)
                frame_count += 1
                
                # Accumulate every 3rd frame for the GIF (resized to 1024 width for high quality)
                if frame_count % 3 == 0:
                    gif_h = int(h * (1024.0 / w))
                    resized_out = cv2.resize(out, (1024, gif_h))
                    gif_frames.append(cv2.cvtColor(resized_out, cv2.COLOR_BGR2RGB))

            if args.show:
                cv2.imshow("Traffic Light", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if args.save:
        print(f"Saved annotated video to {video_path}")
        print(f"Saved {frame_count} frames to {frames_dir}")
        if gif_frames:
            gif_path = os.path.join(run_dir, base_name + ".gif")
            
            # Safely calculate fps, falling back to 25 if cv2 returns 0
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            raw_fps = raw_fps if raw_fps > 0 else 25.0
            
            target_fps = max(1.0, raw_fps / 3.0)
            
            # Save GIF, using duration instead of fps to be safe with newer imageio
            duration_ms = int(1000.0 / target_fps)
            imageio.mimsave(gif_path, gif_frames, duration=duration_ms, loop=0)
            print(f"Saved animated GIF to {gif_path}")


if __name__ == "__main__":
    main()

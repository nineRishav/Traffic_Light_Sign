"""
S2TLD Evaluation Script
Runs the traffic light YOLO model over the S2TLD JPEGImages and compares
predictions against the Pascal VOC XML annotations.

Metrics reported per class and overall:
  - TP, FP, FN counts
  - Precision, Recall, F1
  - Detection accuracy (% images with at least one correct class match)

IoU threshold for a TP match: 0.30 (lenient, since S2TLD lights are tiny)
"""

import os
import sys
import xml.etree.ElementTree as ET
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# --- Config ---
IOU_THRESH = 0.30
CONF_THRESH = 0.25  # lower conf to catch small lights
IMG_SIZE    = 832

def normalize_class(cls_name):
    """Map YOLO model class names to S2TLD ground truth class names."""
    c = cls_name.lower().strip()
    if "red" in c:    return "red"
    if "green" in c:  return "green"
    if "yellow" in c: return "yellow"
    if "off" in c:    return "off"
    if "wait" in c:   return "wait_on"
    return c


def parse_voc_xml(xml_path):
    """Returns list of (class_name, x1,y1,x2,y2) from a Pascal VOC XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bb = obj.find("bndbox")
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        objects.append((name, x1, y1, x2, y2))
    return objects


def iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)


def run_yolo(model, img_path, imgsz, conf_thres, device):
    """Run inference and return list of (class_name_normalized, x1,y1,x2,y2, conf)."""
    import cv2
    frame = cv2.imread(str(img_path))
    if frame is None:
        return []
    results = model.predict(
        frame, imgsz=imgsz, conf=conf_thres,
        agnostic_nms=True, device=device, verbose=False
    )[0]
    detections = []
    for box in results.boxes:
        cls_idx = int(box.cls[0])
        cls_raw = model.names.get(cls_idx, f"class_{cls_idx}")
        cls_name = normalize_class(cls_raw)  # normalize to S2TLD names
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((cls_name, x1, y1, x2, y2, conf_val))
    return detections


def evaluate(model_path, jpeg_dir, ann_dir, imgsz, conf_thres, device, out_csv):
    from ultralytics import YOLO

    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)

    # Iterate over real images (non-zero size) and find matching annotation
    image_files = sorted([
        f for f in Path(jpeg_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and f.stat().st_size > 0
    ])
    print(f"Found {len(image_files)} real images in {jpeg_dir}\n")

    # Per-class counters
    classes = {"red", "yellow", "green", "off", "wait_on"}
    tp_cls = defaultdict(int)
    fp_cls = defaultdict(int)
    fn_cls = defaultdict(int)

    per_image_results = []

    for img_path in image_files:
        # Find matching annotation
        xml_path = Path(ann_dir) / (img_path.stem + ".xml")
        if not xml_path.exists():
            continue

        gt_objects = parse_voc_xml(xml_path)
        pred_objects = run_yolo(model, img_path, imgsz, conf_thres, device)

        # Match predictions to GTs with IoU
        matched_gt  = set()
        matched_pred = set()

        for pi, (p_cls, px1, py1, px2, py2, p_conf) in enumerate(pred_objects):
            best_iou   = IOU_THRESH
            best_gi    = -1
            for gi, (g_cls, gx1, gy1, gx2, gy2) in enumerate(gt_objects):
                if gi in matched_gt:
                    continue
                if g_cls != p_cls:
                    continue
                score = iou([px1, py1, px2, py2], [gx1, gy1, gx2, gy2])
                if score >= best_iou:
                    best_iou = score
                    best_gi  = gi
            if best_gi >= 0:
                tp_cls[p_cls] += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                fp_cls[p_cls] += 1

        # Unmatched GTs are FNs
        for gi, (g_cls, *_) in enumerate(gt_objects):
            if gi not in matched_gt:
                fn_cls[g_cls] += 1

        # Image-level: correct if dominant GT class predicted at least once
        gt_classes_present  = {o[0] for o in gt_objects}
        pred_classes_present = {o[0] for o in pred_objects}
        img_correct = bool(gt_classes_present & pred_classes_present)

        n_gt   = len(gt_objects)
        n_pred = len(pred_objects)
        n_tp   = sum(1 for pi in matched_pred for _ in [pi])

        per_image_results.append({
            "image":   img_path.name,
            "n_gt":    n_gt,
            "n_pred":  n_pred,
            "n_tp":    len(matched_pred),
            "gt_cls":  ",".join(sorted(gt_classes_present)),
            "pred_cls":",".join(sorted(pred_classes_present)),
            "correct": img_correct,
        })

    # Print per-class metrics
    print(f"\n{'='*70}")
    print(f"{'Class':<12} {'TP':>6} {'FP':>6} {'FN':>6}  {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"{'-'*70}")

    all_tp = all_fp = all_fn = 0
    for cls in sorted(classes):
        tp = tp_cls[cls]; fp = fp_cls[cls]; fn = fn_cls[cls]
        all_tp += tp; all_fp += fp; all_fn += fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
        print(f"{cls:<12} {tp:>6} {fp:>6} {fn:>6}  {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")

    print(f"{'-'*70}")
    overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_rec  = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    overall_f1   = 2*overall_prec*overall_rec / (overall_prec+overall_rec) if (overall_prec+overall_rec) > 0 else 0
    print(f"{'OVERALL':<12} {all_tp:>6} {all_fp:>6} {all_fn:>6}  {overall_prec:>7.3f} {overall_rec:>7.3f} {overall_f1:>7.3f}")
    print(f"{'='*70}")

    img_acc = sum(r["correct"] for r in per_image_results) / len(per_image_results) if per_image_results else 0
    print(f"\nImages evaluated : {len(per_image_results)}")
    print(f"Image-level acc  : {img_acc*100:.1f}%  (≥1 correct class predicted)")
    print(f"Overall Precision : {overall_prec*100:.1f}%")
    print(f"Overall Recall    : {overall_rec*100:.1f}%")
    print(f"Overall F1        : {overall_f1*100:.1f}%")

    # Save per-image CSV
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_results[0].keys())
            writer.writeheader()
            writer.writerows(per_image_results)
        print(f"\nPer-image results saved to: {out_csv}")

    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate YOLO model on S2TLD dataset")
    p.add_argument("--model",  required=True, help="Path to .pt weights")
    p.add_argument("--images", required=True, help="Path to JPEGImages dir")
    p.add_argument("--anns",   required=True, help="Path to Annotations dir")
    p.add_argument("--imgsz",  type=int, default=IMG_SIZE)
    p.add_argument("--conf",   type=float, default=CONF_THRESH)
    p.add_argument("--device", default="0")
    p.add_argument("--out-csv", default="s2tld_eval_results.csv")
    args = p.parse_args()

    evaluate(
        model_path=args.model,
        jpeg_dir=args.images,
        ann_dir=args.anns,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        device=args.device,
        out_csv=args.out_csv,
    )

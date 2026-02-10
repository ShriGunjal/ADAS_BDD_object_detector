import os
import cv2
import numpy as np


def visualize_and_save(
    img_path,
    img_name,
    gt_boxes,
    pred_boxes,
    class_map,
    save_dir
):
    """
    Visualize GT (green) and Predicted (red) bounding boxes.
    Only class ID is shown on each box.
    A palette mapping (ID -> class name) is shown on the right side.

    Args:
        img_path (str): Path to image directory.
        img_name (str): Image filename.
        gt_boxes (list): List of [cls_id, x1, y1, x2, y2].
        pred_boxes (list): List of [cls_id, x1, y1, x2, y2].
        class_map (dict): {id: class_name}.
        save_dir (str): Directory to save visualization.
    """

    full_path = os.path.join(img_path, img_name)
    if not os.path.exists(full_path):
        print(f"Image not found: {full_path}")
        return

    img = cv2.imread(full_path)
    if img is None:
        print(f"cv2 failed to read image: {full_path}")
        return

    vis_img = img.copy()
    h, w = vis_img.shape[:2]

    thickness = max(2, h // 500)
    font_scale = max(0.5, h / 1500)

    # Draw GT boxes (Green)
    for box in gt_boxes:
        cls_id, x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(
            vis_img,
            str(cls_id),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA
        )

    # Draw Pred boxes (Red)
    for box in pred_boxes:
        cls_id, x1, y1, x2, y2 , cls_conf= map(int, box)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        cv2.putText(
            vis_img,
            str(cls_id),
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA
        )

    # ---------- Add Palette Area ----------
    palette_width = 300
    palette = 255 * np.ones((h, palette_width, 3), dtype="uint8")

    y_offset = 40
    for cls_id, cls_name in class_map.items():
        text = f"{cls_id} : {cls_name}"
        cv2.putText(
            palette,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        y_offset += 30

    # Combine image + palette
    combined = cv2.hconcat([vis_img, palette])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.splitext(img_name)[0] + "_vis.png")

    cv2.imwrite(save_path, combined)
    print(f"Saved visualization to {save_path}")



def visualize_detection_results(
    img_path,
    img_name,
    detection_results,
    class_map,
    save_dir
):
    """
    Visualize TP, FP, FN and matched GT boxes.

    Colors:
        TP -> Green
        FP -> Red
        FN -> Yellow
        Matched GT -> White

    Also shows:
        - Color legend
        - Class ID to Class Name mapping
    """

    full_path = os.path.join(img_path, img_name)
    if not os.path.exists(full_path):
        print(f"Image not found: {full_path}")
        return

    img = cv2.imread(full_path)
    if img is None:
        print(f"Failed to load image: {full_path}")
        return

    vis_img = img.copy()
    h, w = vis_img.shape[:2]

    thickness = max(2, h // 500)
    font_scale = max(0.5, h / 1500)

    # =========================
    # Draw Predictions
    # =========================
    for pred in detection_results["predictions"]:

        x1, y1, x2, y2 = map(int, pred["box"])
        cls_id = pred["cls_id"]

        if pred["status"] == "TP":
            color = (0, 255, 0)      # Green
        else:
            color = (0, 0, 255)      # Red

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            vis_img,
            str(cls_id),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    # =========================
    # Draw Ground Truths
    # =========================
    for gt in detection_results["ground_truths"]:

        x1, y1, x2, y2 = map(int, gt["box"])
        cls_id = gt["cls_id"]

        if gt["status"] == "FN":
            color = (0, 255, 255)    # Yellow
        else:
            color = (255, 255, 255)  # White

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            vis_img,
            str(cls_id),
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    # =========================
    # Create Palette Panel
    # =========================
    palette_width = 350
    palette = 255 * np.ones((h, palette_width, 3), dtype=np.uint8)

    y_offset = 30
    line_spacing = 30
    
    tp = detection_results["summary"]["tp"]
    fp = detection_results["summary"]["fp"]
    fn = detection_results["summary"]["fn"]


    # ----- Legend -----
    legend_items = [
        ("TP (Prediction): "+str(tp), (0, 255, 0)),
        ("FP (Prediction): "+str(fp), (0, 0, 255)),
        ("FN (Missed GT): "+str(fn), (0, 255, 255)),
        ("Matched GT: closeness with TP", (255, 255, 255)),
    ]

    cv2.putText(palette, "Legend:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_offset += line_spacing

    for text, color in legend_items:
        cv2.rectangle(palette, (10, y_offset - 15),
                      (30, y_offset - 5), color, -1)
        cv2.putText(palette, text, (40, y_offset - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_offset += line_spacing

    y_offset += 20

    # ----- Class Map -----
    cv2.putText(palette, "Class ID Mapping:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_offset += line_spacing

    for cls_id, cls_name in class_map.items():
        text = f"{cls_id} : {cls_name}"
        cv2.putText(palette, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_offset += line_spacing

    # Combine image + palette
    combined = cv2.hconcat([vis_img, palette])

    # =========================
    # Save With Summary in Name
    # =========================
    base_name = os.path.splitext(img_name)[0]
    save_name = f"{base_name}_TP{tp}_FP{fp}_FN{fn}.png"
    save_path = os.path.join(save_dir, save_name)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_path, combined)

    print(f"Saved evaluation visualization to {save_path}")




def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        float: IoU
    """

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0, (x2_1 - x1_1)) * max(0, (y2_1 - y1_1))
    area2 = max(0, (x2_2 - x1_2)) * max(0, (y2_2 - y1_2))

    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


def match_detections(gt_boxes, pred_boxes, class_map, iou_threshold=0.5):
    """
    Perform greedy matching between GT and predictions.

    Args:
        gt_boxes: list of [cls_id, x1, y1, x2, y2]
        pred_boxes: list of [cls_id, x1, y1, x2, y2, confidence]
        class_map: dict {cls_id: class_name}
        iou_threshold: float

    Returns:
        dict containing per-box mapping and summary
    """

    results = {
        "predictions": [],
        "ground_truths": [],
        "summary": {"tp": 0, "fp": 0, "fn": 0}
    }

    # ----------------------------
    # Sanitize Inputs
    # ----------------------------
    gt_boxes = [
        g for g in gt_boxes
        if isinstance(g, (list, tuple)) and len(g) >= 5
    ]

    pred_boxes = [
        p for p in pred_boxes
        if isinstance(p, (list, tuple)) and len(p) >= 6
    ]

    # ----------------------------
    # Case 1: No GT and No Predictions
    # ----------------------------
    if not gt_boxes and not pred_boxes:
        return results

    # ----------------------------
    # Case 2: No Predictions
    # ----------------------------
    if not pred_boxes:
        for gt in gt_boxes:
            cls_id = gt[0]
            results["ground_truths"].append({
                "cls_id": cls_id,
                "class_name": class_map.get(cls_id, "unknown"),
                "box": gt[1:5],
                "status": "FN"
            })
        results["summary"]["fn"] = len(gt_boxes)
        return results

    # ----------------------------
    # Case 3: No Ground Truth
    # ----------------------------
    if not gt_boxes:
        for pred in pred_boxes:
            cls_id = pred[0]
            results["predictions"].append({
                "cls_id": cls_id,
                "class_name": class_map.get(cls_id, "unknown"),
                "box": pred[1:5],
                "confidence": pred[5],
                "status": "FP",
                "matched_gt_idx": None,
                "iou": 0.0
            })
        results["summary"]["fp"] = len(pred_boxes)
        return results

    # ----------------------------
    # Normal Matching Case
    # ----------------------------
    matched_gt = set()

    # Sort predictions by confidence (descending)
    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[5], reverse=True)

    # -------- Process Predictions --------
    for pred in pred_boxes_sorted:
        pred_cls = pred[0]
        pred_class_name = class_map.get(pred_cls, "unknown")

        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            if gt[0] != pred_cls:
                continue

            iou = compute_iou(pred[1:5], gt[1:5])

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold:
            # True Positive
            results["predictions"].append({
                "cls_id": pred_cls,
                "class_name": pred_class_name,
                "box": pred[1:5],
                "confidence": pred[5],
                "status": "TP",
                "matched_gt_idx": best_gt_idx,
                "iou": best_iou
            })

            matched_gt.add(best_gt_idx)
            results["summary"]["tp"] += 1

        else:
            # False Positive
            results["predictions"].append({
                "cls_id": pred_cls,
                "class_name": pred_class_name,
                "box": pred[1:5],
                "confidence": pred[5],
                "status": "FP",
                "matched_gt_idx": None,
                "iou": best_iou
            })

            results["summary"]["fp"] += 1

    # -------- Process Ground Truths --------
    for idx, gt in enumerate(gt_boxes):
        cls_id = gt[0]

        if idx in matched_gt:
            results["ground_truths"].append({
                "cls_id": cls_id,
                "class_name": class_map.get(cls_id, "unknown"),
                "box": gt[1:5],
                "status": "matched"
            })
        else:
            results["ground_truths"].append({
                "cls_id": cls_id,
                "class_name": class_map.get(cls_id, "unknown"),
                "box": gt[1:5],
                "status": "FN"
            })
            results["summary"]["fn"] += 1

    return results


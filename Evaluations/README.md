# Evaluations - BDD100k Validation Dataset Analysis

This folder contains comprehensive evaluation results and analysis for the YOLOv7 model fine-tuned on BDD100k dataset.

## 📋 Contents

### Files & Folders

| Item | Description |
|------|-------------|
| **evaluations.ipynb** | Main Jupyter notebook for running evaluation on BDD100k validation data |
| **eval_utils.py** | Utility functions for evaluation metrics computation and visualization |
| **predicted_labels/** | YOLOv7 model predictions on validation images (YOLO format .txt files) |
| **visualize_TP_FP_FN/** | Visualization images showing True Positives, False Positives, and False Negatives |
| **eval_vis/** | Additional evaluation visualizations |
| **eval_vis_tpfpfn/** | TP/FP/FN visualizations (alternative storage location) |
| **valdata_analysis_results.json** | Complete evaluation results saved as JSON |

---

## 🚀 Running Evaluations

### Prerequisites
- BDD100k validation dataset images in `/workspace/data_store/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/val/`
- BDD100k labels CSV file: `/workspace/data_store/BDD_Image_labels_train_val.csv` 
- Pre-trained model weights: `best.pt` (download from [OneDrive Link](https://1drv.ms/f/c/a5d589f3070f80d5/IgCu09bxT8LvTYdishSeDayOAeiA2ZwybnP2ol8qKFhT1h4?e=avzndy))

### Step 1: Generate Predictions

Run inference on validation data with specified thresholds:

```bash
cd /workspace/yolov7
python detect.py \
  --weights runs/train/exp/weights/best.pt \
  --conf 0.25 \
  --img-size 640 \
  --source /workspace/data_store/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/val/ \
  --save-txt \
  --save-conf
```

**Parameters:**
- `--conf 0.25`: Confidence threshold for detections
- `--save-txt`: Save predictions in YOLO format
- `--save-conf`: Include confidence scores in output files

Copy generated labels from `runs/detect/expX/labels/` to `predicted_labels/` folder.

### Step 2: Run Evaluation Notebook

```bash
cd /workspace/Evaluations
jupyter notebook evaluations.ipynb
```

Execute all cells to compute evaluation metrics and generate visualizations.

---

## 📊 Evaluation Configuration

The evaluation uses the following parameters to match predictions with ground truth:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Confidence Threshold** | 0.25 | Model prediction confidence cutoff |
| **IoU Threshold** | 0.5 | Intersection over Union for match detection |
| **NMS Threshold** | 0.45 | Non-Maximum Suppression overlap threshold |
| **Validation Split** | BDD100k Val | ~10,000 images with diverse scenarios |

---

## 📈 Metrics Computed

The evaluation notebook computes three types of performance metrics:

### 1. Overall Performance Metrics
```
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
- Total TP, FP, FN counts
```

### 2. Per-Class Metrics
Individual performance breakdown for each of the 10 BDD100k classes:
- **Person**: Pedestrians
- **Rider**: Cyclists and motorcycle riders  
- **Car**: Standard vehicles
- **Bus**: Public transport
- **Train**: Rail vehicles
- **Truck**: Large commercial vehicles
- **Motorcycle**: Motorcycles
- **Bike**: Bicycles
- **Traffic Light**: Traffic signal lights
- **Traffic Sign**: Road signs

For each class: Precision, Recall, TP, FP, FN counts

### 3. Size-based Recall Analysis
Performance across different object sizes:
- **Small** (<32×32 pixels): Distant objects, small signs
- **Medium** (32×96 pixels): Typical vehicles and pedestrians
- **Large** (>96×96 pixels): Close-up vehicles and infrastructure

---

## 🎨 Visualization: TP/FP/FN Analysis

The `visualize_TP_FP_FN/` folder contains annotated images showing detection performance:

**Color Code:**
- 🟢 **Green Boxes**: True Positives (TP) - Correctly detected objects
- 🔴 **Red Boxes**: False Positives (FP) - Incorrect predictions
- 🟡 **Yellow Boxes**: False Negatives (FN) - Missed ground truth objects
- ⚪ **White Boxes**: Ground Truth (GT) boxes that matched with TP boxes - shown for reference to visualize closeness with predicted TP boxes

**File Naming Convention:**
```
{image_id}_TP{count}_FP{count}_FN{count}.png
```

Example: `b1d0a191-03dcecc2_TP32_FP10_FN6.png`
- 32 True Positives
- 10 False Positives  
- 6 False Negatives

---

## 📁 How Evaluation Works

### 1. Data Preparation
- Reads ground truth from BDD100k labels CSV
- Loads YOLOv7 predictions from `.txt` files
- Converts predictions (normalized coordinates) to pixel coordinates

### 2. Matching Strategy
For each image, the notebook:
- Matches predictions to ground truth using IoU threshold (0.5)
- Assigns class IDs and calculates overlap
- Classifies each detection as TP, FP, or FN

### 3. Metrics Aggregation
- **TP**: Prediction matched ground truth with IoU ≥ 0.5
- **FP**: Prediction with no matching ground truth (IoU < 0.5)
- **FN**: Ground truth with no matching prediction

### 4. Visualization
- Draws bounding boxes for TP (green), FP (red), FN (yellow)
- Saves annotated images to `visualize_TP_FP_FN/`

---

## 📊 Key Performance Insights

After running evaluation, check:

1. **Overall Metrics**

2. **Per-Class Analysis**

3. **Size Analysis**
   
---

## 🔍 Interpreting Visualizations

When viewing `visualize_TP_FP_FN/` images:

**Good Performance Indicators:**
- Mostly green boxes (high TP rate)
- Few red boxes (low FP rate)
- Minimal yellow boxes (low FN rate)
- Consistent detection across different scenarios

**Areas for Improvement:**
- Multiple red boxes in same scene → detector too sensitive
- Many yellow boxes → missing important objects
- Class-specific false alarms → tune confidence threshold
- Size-specific misses → model scaling issues

---

## 💡 Next Steps

After reviewing evaluation results:

1. **Fine-tune Model**
   - Adjust confidence threshold if needed
   - Reweight loss for underperforming classes
   - Augment training data for challenging scenarios

2. **Analyze Failure Cases**
   - Identify common patterns in false positives
   - Study false negative patterns
   - Check for dataset issues or label errors

3. **Optimize for Deployment**
   - Balance precision/recall based on application needs
   - Consider real-time inference requirements
   - Test on edge hardware if applicable

---

## 📚 Related Files

- **Main README**: [../README.md](../README.md) - Project overview and setup instructions
- **Training Results**: [../snap_ui/](../snap_ui/) - Training curves and metrics
- **Training Configuration**: [../yolov7/cfg/training/yolov7bdd.yaml](../yolov7/cfg/training/yolov7bdd.yaml)
- **Dataset Configuration**: [../yolov7/data/bdd100k.yaml](../yolov7/data/bdd100k.yaml)

---

## 🎯 Quick Reference

**Run Full Evaluation Pipeline:**
```bash
# Step 1: Generate predictions (from yolov7 folder)
cd /workspace/yolov7
python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 \
  --img-size 640 --source /workspace/data_store/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/val/ \
  --save-txt --save-conf

# Step 2: Copy predictions to Evaluations folder
cp runs/detect/exp*/labels/*.txt /workspace/Evaluations/predicted_labels/

# Step 3: Run evaluation
cd /workspace/Evaluations
jupyter notebook evaluations.ipynb
# Execute all cells to compute metrics and visualizations
```

---

## 📝 Evaluation Utils Reference

**Key functions in `eval_utils.py`:**

- `match_detections()`: Match predictions to ground truth, compute TP/FP/FN
- `compute_iou()`: Calculate Intersection over Union
- `visualize_detection_results()`: Draw and save annotated images
- `compute_overall_metrics_from_json()`: Aggregate metrics from results
- `compute_per_class_metrics_from_json()`: Per-class performance breakdown
- `compute_size_based_recall_from_json()`: Size-wise recall analysis

---

**For questions or issues, refer to the main [README.md](../README.md) or contact the development team.**

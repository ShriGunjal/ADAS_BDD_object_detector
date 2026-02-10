# ADAS_BDD_object_detector

Automated Driving Assistance System (ADAS) object detector using YOLOv7 trained on BDD100k dataset.

## Project Overview

This project demonstrates an end-to-end implementation of an advanced object detection system specifically tuned for autonomous driving scenarios. The system leverages the YOLOv7 architecture and is trained on the BDD100k (Berkeley DeepDrive) dataset, which contains diverse real-world driving scenarios.

## Key Achievements

✅ **Model Training**: Successfully trained YOLOv7 on BDD100k dataset for 10 epochs with optimized hyperparameters
- Achieved high precision and recall on diverse driving scenarios
- Transfer learning from COCO pre-trained weights for faster convergence

✅ **Custom Data Loader**: Modified and optimized the data loader for BDD100k dataset
- Efficient batch processing and augmentation pipeline
- Support for multiple object classes (vehicles, pedestrians, cyclists, traffic signs, etc.)
- Proper handling of BDD100k label format and image preprocessing

✅ **Model Performance Testing**: Comprehensive evaluation on test set
- Precision-Recall analysis across confidence thresholds
- Per-class performance metrics via confusion matrix
- F1-score optimization for balanced precision-recall trade-off

✅ **Visualization & Results**: Complete analysis and visualization of training results
- Training/validation loss curves over epochs
- Performance curves (P, R, F1) at different confidence levels
- Real-world detection results on BDD100k test images
- Confusion matrix for detailed class-wise analysis

## Prerequisites

- Docker with NVIDIA GPU support (nvidia-docker or Docker with GPU runtime)
- Git
- CUDA-capable GPU
- Sufficient disk space for data_store and models


## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd ADAS_BDD_object_detector
```

### 2. Build Docker Image

Navigate to the directory containing the Dockerfile and build the image:

```bash
docker build -t odetector-bdd:latest .
```

Expected build time: 10-15 minutes (varies based on internet speed and GPU)

### 3. Start Docker Container (with port mapping)

After `cd ADAS_BDD_object_detector`, run:

```bash
docker run --name odetector -it --gpus all \
  -v ./data_store/:/workspace/data_store/ \
  -v ./yolov7/:/workspace/yolov7/ \
  -v ./Evaluations/:/workspace/Evaluations/ \
  -v ./export_models/:/workspace/export_models/ \
  -p 8888:8888 \
  --hostname localhost \
  --shm-size=64g \
  odetector-bdd:latest
```

This starts an interactive bash terminal inside the container. You can now run Python scripts directly.

**Note:** Use relative paths and ensure you're in the `ADAS_BDD_object_detector` directory.

### 4. Data Analysis (First Step - Using Jupyter Notebook)

Before training, analyze your dataset using the provided Jupyter notebook:

```bash
docker exec -it odetector jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

Open your browser and navigate to `http://localhost:8888`, then open:
- **`data_store/DataVis.ipynb`** - Data visualization and analysis notebook

This notebook provides:
- Dataset overview and statistics
- Label distribution analysis
- Image quality checks
- Data preprocessing verification
- BDD100k dataset exploration

**Recommendation:** Complete this analysis first to ensure data integrity before proceeding with training.

### 5. Download Pre-trained Weights and Models (Optional)

Pre-trained YOLOv7 weights and models trained on BDD dataset are available at:

**[OneDrive Link - Pre-trained Weights & Trained Models](https://1drv.ms/f/c/a5d589f3070f80d5/IgCu09bxT8LvTYdishSeDayOAeiA2ZwybnP2ol8qKFhT1h4?e=avzndy)**

Download and extract:
- `yolov7_pretrained_coco80.pt` - YOLOv7 pre-trained on COCO dataset (place in `/workspace/yolov7/`)
- `best.pt` - Best model trained on BDD100k dataset (place in `/workspace/yolov7/runs/train/exp/weights/`)
- `last.pt` - Last checkpoint from BDD100k training (place in `/workspace/yolov7/runs/train/exp/weights/`)
- `results/` - Training results and metrics folder

### 6. Train the Model (inside container terminal)

Ensure your data is organized in `data_store/` and configuration is set up, then run:

```bash
cd /workspace/yolov7
python train.py --img 640 --batch 16 --epochs 10 --data /workspace/yolov7/data/bdd100k.yaml --cfg /workspace/yolov7/cfg/training/yolov7bdd.yaml --weights yolov7_pretrained_coco80.pt --device 0
```

**Training parameters:**
- `--img 640` : Image size 640x640
- `--batch 16` : Batch size 16
- `--epochs 10` : Number of training epochs
- `--data bdd100k.yaml` : Dataset configuration
- `--cfg /yolov7/cfg/training/yolov7bdd.yaml` : YOLOv7-BDD architecture
- `--weights yolov7_pretrained_coco80.pt` : Pre-trained YOLOv7 on COCO dataset
- `--device 0` : GPU device 0

Training output will be saved to `runs/train/exp/`

### 7. Training Results and Metrics

Training results and performance metrics are available in the `snap_ui/` folder:

**Overall Training Results:**
![Training Results](snap_ui/results.png)

**Performance Curves:**

![Precision-Recall Curve](snap_ui/PR_curve.png)
*Precision-Recall Curve - Overall PR performance*

![F1-Score Curve](snap_ui/F1_curve.png)
*F1-Score Curve - F1 score vs confidence threshold*

![Precision Curve](snap_ui/P_curve.png)
*Precision Curve - Precision vs confidence threshold*

![Recall Curve](snap_ui/R_curve.png)
*Recall Curve - Recall vs confidence threshold*

![Confusion Matrix](snap_ui/confusion_matrix.png)
*Confusion Matrix - Detailed class-wise performance*

**Sample Detection Results:**

Detection results on test BDD100k images showing ADAS object detection capabilities:
- Classes detected: vehicles, buses, pedestrians, cyclists, traffic signs, traffic lights
- Real-world driving scenarios from the BDD100k dataset

Example detections:
![Detection Sample 1](snap_ui/detections_results/cad7fdff-d9946f73.jpg)
![Detection Sample 2](snap_ui/detections_results/caf56f7a-1407df75.jpg)

### 8. Comprehensive Validation Results & Metrics on BDD100k Val Data

Complete evaluation metrics available in the [Evaluations/evaluations.ipynb](Evaluations/evaluations.ipynb) notebook with detailed performance analysis.

#### Evaluation Configuration

The validation uses carefully tuned inference and evaluation parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Confidence Threshold** | 0.25 | Detection confidence cutoff |
| **IoU Threshold** | 0.5 | Intersection over Union for matching predictions to ground truth |
| **NMS Threshold** | 0.45 | Non-Maximum Suppression overlap threshold |
| **Validation Data** | BDD100k Validation Split | Diverse real-world driving scenarios |
| **Fine-tuned Classes** | 10 classes | Person, Rider, Car, Bus, Train, Truck, Motorcycle, Bike, Traffic Light, Traffic Sign |

👉 **For detailed metrics, run the evaluation notebook:**
```bash
cd /workspace/Evaluations
jupyter notebook evaluations.ipynb
```

The notebook computes:
- **Overall Metrics**: Precision, Recall, F1-Score, TP/FP/FN counts
- **Class-wise Performance**: Per-class metrics for all 10 object categories
- **Size-wise Recall**: Performance across small, medium, and large objects

---

#### Detection Performance Visualization

Visual analysis of True Positives (TP), False Positives (FP), and False Negatives (FN) across validation images:

**Color Legend:**
- 🟢 **Green Boxes (TP)**: Correctly detected objects
- 🔴 **Red Boxes (FP)**: Predicted but incorrect detections  
- 🟡 **Yellow Boxes (FN)**: Missed ground truth objects
- ⚪ **White Boxes (GT)**: Ground truth boxes that matched with TP boxes - shown for reference to visualize closeness with predicted TP boxes

**Representative Validation Results:**

##### Urban Complex Scenes

![Evaluation Result 1](Evaluations/visualize_TP_FP_FN/b1d0a191-03dcecc2_TP32_FP10_FN6.png)
*Complex urban environment with multiple vehicles, pedestrians, and traffic infrastructure*

![Evaluation Result 2](Evaluations/visualize_TP_FP_FN/b1d0a191-2ed2269e_TP26_FP10_FN8.png)
*Densely populated scene with diverse object classes and occlusions*

##### Mixed Traffic Scenarios

![Evaluation Result 3](Evaluations/visualize_TP_FP_FN/b1ceb32e-51852abe_TP12_FP4_FN2.png)
*Balanced traffic scene with vehicles, pedestrians, and cyclists*

![Evaluation Result 4](Evaluations/visualize_TP_FP_FN/b1cd1e94-549d0bfe_TP12_FP2_FN1.png)
*High precision detection with minimal false alerts*

##### Varied Scenarios

![Evaluation Result 5](Evaluations/visualize_TP_FP_FN/b1f4491b-09593e90_TP4_FP2_FN0.png)
*Sparse traffic with complete object detection*

![Evaluation Result 6](Evaluations/visualize_TP_FP_FN/b3e08585-cb03a2fb_TP10_FP2_FN1.png)
*Intersection scene with multiple vehicle and pedestrian interactions*

---

#### Performance Analysis by Object Type

The evaluation notebook provides detailed analysis for:

1. **Overall Detection Performance** - Aggregate precision, recall, and F1-score metrics
2. **Per-Class Metrics** - Individual performance for:
   - Person, Rider, Car, Bus, Train, Truck
   - Motorcycle, Bike, Traffic Light, Traffic Sign
3. **Size-based Performance** - Recall analysis for:
   - Small objects (<32×32 pixels)
   - Medium objects (32×96 pixels)  
   - Large objects (>96×96 pixels)

**🔍 Navigate to the evaluation notebook to see:**
- Precision/Recall values for each class
- True Positive, False Positive, False Negative counts
- Per-size performance breakdown
- Confusion matrix analysis

### 9. Run Inference on Test Data (inside container terminal)

```bash
cd /workspace/yolov7
python detect.py --weights runs/train/exp/weights/best.pt --source /workspace/data_store/test_data/
```

Or use the pre-trained best model from OneDrive:

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source /workspace/data_store/test_data/
```

### 10. Jupyter Notebook for Results Visualization (Optional)

From another terminal on your host machine (keep the container running):

```bash
docker exec -it odetector jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

Then open browser: `http://localhost:8888`

### 11. Stopping and Restarting

Stop container:
```bash
docker stop odetector
```

Restart container:
```bash
docker start -i odetector
```

Attach to running container:
```bash
docker exec -it odetector /bin/bash
```

## Project Implementation Details

### Model Training
- **Architecture**: YOLOv7 with YOLOv7-BDD custom configuration
- **Dataset**: BDD100k - Berkeley DeepDrive 100K dataset
- **Training Duration**: 10 epochs with batch size 16
- **Base Weights**: YOLOv7 pre-trained on COCO dataset (80 classes)
- **Transfer Learning**: Leveraging COCO pre-training for faster convergence on BDD classes

### Custom Data Loader
The BDD100k data loader has been modified to:
- Parse BDD100k specific label format and structure
- Handle diverse driving scenarios (weather, time of day, occlusion)
- Implement efficient data augmentation pipeline
- Support multiple object classes: vehicles, buses, pedestrians, cyclists, motorcycles, traffic signs, traffic lights
- Proper annotation preprocessing for YOLOv7 training format

### Model Performance
The trained model achieves strong performance metrics:
- **High Precision**: Minimizes false positives in real-world driving
- **High Recall**: Detects objects even in challenging conditions
- **F1-Score Optimization**: Balanced performance across all object classes
- **Robust Detection**: Works across different weather conditions, time of day, and occlusion scenarios

### Results and Visualization
Complete analysis available in `snap_ui/` folder:
- **Training curves**: Loss convergence and metric evolution
- **Performance analysis**: Precision, Recall, F1-score at various confidence thresholds
- **Class-wise performance**: Confusion matrix showing per-class accuracy
- **Real-world detections**: Sample images showing detection in actual driving scenarios

## Detected Object Classes
- Vehicles (cars, trucks)
- Buses
- Pedestrians
- Cyclists
- Motorcycles
- Traffic Signs
- Traffic Lights

## Next Steps
- Deploy model for real-time inference
- Integrate with vehicle systems for ADAS functionality
- Further fine-tune on specific driving scenarios
- Optimize model size for edge deployment

## License

This project builds upon YOLOv7 and uses the BDD100k dataset. Please refer to respective licenses.

## References

- [YOLOv7: Trainable state-of-the-art object detector](https://github.com/WongKinYiu/yolov7)
- [Berkeley DeepDrive Dataset](https://bdd-data.berkeley.edu/)
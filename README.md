# ADAS_BDD_object_detector

Automated Driving Assistance System (ADAS) object detector using YOLOv7 trained on BDD100k dataset.

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

**Performance Curves:**
- **Precision-Recall Curve** (`snap_ui/PR_curve.png`) - Overall PR performance
- **F1-Score Curve** (`snap_ui/F1_curve.png`) - F1 score vs confidence threshold
- **Precision Curve** (`snap_ui/P_curve.png`) - Precision vs confidence threshold
- **Recall Curve** (`snap_ui/R_curve.png`) - Recall vs confidence threshold
- **Confusion Matrix** (`snap_ui/confusion_matrix.png`) - Detailed class-wise performance
- **Training Results** (`snap_ui/results.png`) - Training/validation loss and metrics over epochs

**Sample Detection Results:**
Detection results on test images are available in `snap_ui/detections_results/`:
- Example detections showing ADAS object detection capabilities
- Classes detected: vehicles, pedestrians, traffic signs, cyclists, etc.

### 8. Run Inference on Test Data (inside container terminal)

```bash
cd /workspace/yolov7
python detect.py --weights runs/train/exp/weights/best.pt --source /workspace/data_store/test_data/
```

Or use the pre-trained best model from OneDrive:

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source /workspace/data_store/test_data/
```

### 9. Jupyter Notebook for Results Visualization (Optional)

From another terminal on your host machine (keep the container running):

```bash
docker exec -it odetector jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

Then open browser: `http://localhost:8888`

### 10. Stopping and Restarting

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

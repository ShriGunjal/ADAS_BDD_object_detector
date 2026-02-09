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

### 4. Run Python Scripts (inside container terminal)

```bash
cd /workspace/yolov7
python train.py --img 640 --batch 32 --epochs 100 --data data/bdd100k.yaml
python detect.py --weights runs/train/exp/weights/best.pt --source /workspace/data_store/test_data/
```

### 5. Run Jupyter Notebook (when needed)

From another terminal on your host machine (keep the container running):

```bash
docker exec -it odetector jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

Then open browser: `http://localhost:8888`

### 6. Stopping and Restarting

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

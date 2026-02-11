
## Export to ONNX and TensorRT 
Exported onnx, trt models and visual results can be found at:
**[OneDrive Link - onnx, trt models and visual results](https://1drv.ms/f/c/a5d589f3070f80d5/IgDpXcuy5pXDQZ1R0DKATz5CAUKwDejOKmtBpO4y7XQkFZI?e=W7EuE4)**

To export the trained YOLOv7 model to ONNX and build a TensorRT engine, and then run inference with the TensorRT engine, perform the following steps from the repository root.

1. Change into the `yolov7` directory:

```bash
cd yolov7
```

2. Export to ONNX (example command used for this project):

```bash
python export.py --weights runs/train/exp7/weights/best.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.5 --conf-thres 0.25 --img-size 640 640
```

This produces `runs/train/exp7/weights/best.onnx`.

3. Clone the TensorRT helper repository and build a TRT engine (example using `tensorrt-python`):

```bash
git clone https://github.com/Linaom1214/tensorrt-python.git
cd tensorrt-python
```

4. Convert ONNX to TensorRT engine (FP16 example):

```bash
python ./tensorrt-python/export.py -o ../yolov7/runs/train/exp7/weights/best.onnx -e best.trt -p fp16
```

Note: adjust paths if you ran the ONNX export to a different location. The resulting engine `best.trt` will be created in the current working directory.

5. Run TensorRT inference using the project's `infer_tensorrt.py` (assumes a TRT-based inference script is available):

```bash
# from repository root
cd yolov7
#modify trt model path and test_data paths then run
python infer_tensorrt.py
```

6. Output images from TensorRT inference should be saved to a folder (for example `trt_outputs/`). Add those images to the README by placing them under the repository and embedding like:

![TRT output example](export_models/trt_outputs/b1ca2e5d-84cf9134.jpg)
![TRT output example 2](export_models/trt_outputs/b2bdb7b6-d34fab57.jpg)

import cv2
import torch
import time
import random
import numpy as np
import tensorrt as trt
from pathlib import Path


# ==========================================================
# LOAD ENGINE
# ==========================================================
def load_trt_engine(engine_path, device):

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    bindings = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = tuple(engine.get_tensor_shape(name))

        # Fix dynamic batch
        if shape[0] == -1:
            shape = (1,) + shape[1:]

        tensor = torch.empty(size=shape,
                             dtype=torch.from_numpy(
                                 np.empty([], dtype=dtype)).dtype,
                             device=device)

        bindings[name] = tensor

    return engine, context, bindings


# ==========================================================
# LETTERBOX
# ==========================================================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):

    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


# ==========================================================
# PREPROCESS
# ==========================================================
def preprocess(image_path, device):

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig = img_rgb.copy()

    img_resized, ratio, dwdh = letterbox(img_rgb)

    img_resized = img_resized.transpose((2, 0, 1))
    img_resized = np.expand_dims(img_resized, 0)
    img_resized = np.ascontiguousarray(img_resized).astype(np.float32)

    tensor = torch.from_numpy(img_resized).to(device)
    tensor /= 255.0

    return orig, tensor, ratio, dwdh


# ==========================================================
# POSTPROCESS
# ==========================================================
def postprocess(boxes, ratio, dwdh, img_shape):

    dwdh = torch.tensor(dwdh * 2, device=boxes.device)
    boxes -= dwdh
    boxes /= ratio

    boxes[:, 0::2].clamp_(0, img_shape[1])
    boxes[:, 1::2].clamp_(0, img_shape[0])

    return boxes.round().int()


# ==========================================================
# INFERENCE (Modern API)
# ==========================================================
def infer(engine, context, bindings, input_tensor):

    start_total = time.perf_counter()

    input_name = engine.get_tensor_name(0)

    # Set dynamic input shape
    context.set_input_shape(input_name, tuple(input_tensor.shape))

    # Bind input
    context.set_tensor_address(input_name, int(input_tensor.data_ptr()))

    # Bind outputs
    for name in bindings:
        if name != input_name:
            context.set_tensor_address(name, int(bindings[name].data_ptr()))

    # Execute
    start_infer = time.perf_counter()
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_infer

    nums = bindings["num_dets"].clone()
    boxes = bindings["det_boxes"].clone()
    scores = bindings["det_scores"].clone()
    classes = bindings["det_classes"].clone()

    num = int(nums[0][0])

    boxes = boxes[0, :num]
    scores = scores[0, :num]
    classes = classes[0, :num]

    total_time = time.perf_counter() - start_total

    return boxes, scores, classes, inference_time, total_time


# ==========================================================
# VISUALIZATION
# ==========================================================
def visualize(image, boxes, scores, classes, class_names):

    colors = {name: [random.randint(0, 255) for _ in range(3)]
              for name in class_names}

    for box, score, cl in zip(boxes, scores, classes):

        class_id = int(cl)
        name = class_names[class_id]
        color = colors[name]

        label = f"{name} {float(score):.3f}"

        cv2.rectangle(image,
                      box[:2].tolist(),
                      box[2:].tolist(),
                      color, 2)

        cv2.putText(image,
                    label,
                    (int(box[0]), int(box[1]) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    return image


# ==========================================================
# RUN FOLDER
# ==========================================================
def run_folder(folder_path,
               engine, context,
               bindings,
               device,
               class_names,
               save_dir="trt_outputs"):

    folder_path = Path(folder_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    image_paths = list(folder_path.glob("*.jpg")) + \
                  list(folder_path.glob("*.png"))

    for img_path in image_paths:

        print(f"Processing {img_path.name}")

        orig_img, tensor, ratio, dwdh = preprocess(str(img_path), device)

        boxes, scores, classes, infer_time, total_time = infer(
            engine, context, bindings, tensor
        )

        boxes = postprocess(boxes, ratio, dwdh, orig_img.shape)

        vis_img = visualize(
            orig_img.copy(),
            boxes, scores, classes,
            class_names
        )

        output_path = save_dir / img_path.name
        output_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), output_bgr)

        print(f"Inference: {infer_time:.4f}s | Total: {total_time:.4f}s")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    device = torch.device("cuda:0")
    engine_path = "/yolov7/best.trt"

    engine, context, bindings = load_trt_engine(engine_path, device)

    class_names = [
        'person','rider','car','bus','train',
        'truck','bike','motor','traffic light','traffic sign'
    ]

    run_folder(
        folder_path="/yolov7/data/test_data/",
        engine=engine,
        context=context,
        bindings=bindings,
        device=device,
        class_names=class_names
    )

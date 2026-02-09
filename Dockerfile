# -----------------------------------------------------------------------------
# Base Image
# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/

# -----------------------------------------------------------------------------
# System deps (NO libgl needed for headless)
# -----------------------------------------------------------------------------
RUN apt update && apt install -y \
    zip \
    htop \
    screen \
    git \
    wget \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Install Python deps EXCEPT OpenCV
# -----------------------------------------------------------------------------
COPY requirements.txt /workspace/

RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Force clean OpenCV install (CRITICAL)
# -----------------------------------------------------------------------------
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2* && \
    pip install --no-cache-dir opencv-python-headless==4.9.0.80

# -----------------------------------------------------------------------------
# Copy source
# -----------------------------------------------------------------------------
COPY yolov7/ /workspace/yolov7/

EXPOSE 8888

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["/bin/bash"]

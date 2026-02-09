# Automated News Livestream Intelligence System

FROM python:3.10-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ============================================================
# Stage 1: Install system dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    cmake \
    build-essential \
    git \
    curl \
    # FFmpeg for audio processing
    ffmpeg \
    # OpenCV dependencies
    libopencv-dev \
    libgl1 \
    libglib2.0-0 \
    # PulseAudio for audio capture
    pulseaudio \
    pulseaudio-utils \
    # Virtual display
    xvfb \
    xauth \
    x11-utils \
    # Chromium browser dependencies (for Patchright)
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcb-dri3-0 \
    libxshmfence1 \
    fonts-liberation \
    fonts-noto-cjk \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================================
# Stage 2: Setup working directory
# ============================================================
WORKDIR /app

COPY pyproject.toml ./

# ============================================================
# Stage 3: Install Python dependencies
# ============================================================
# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install the project dependencies
RUN pip install --no-cache-dir \
    patchright>=1.47.0 \
    opencv-python>=4.10.0 \
    pillow>=10.4.0 \
    numpy>=1.26.4 \
    ultralytics>=8.0.0 \
    easyocr>=1.7.0 \
    pywhispercpp>=1.2.0 \
    llama-cpp-python>=0.2.90 \
    pyyaml>=6.0.2 \
    requests>=2.32.0 \
    tqdm>=4.66.0

# Install Patchright browser (Chromium)
RUN python -m patchright install chromium

# ============================================================
# Stage 4: Download AI Models
# ============================================================

# Create models directory
RUN mkdir -p ./models

# Download YOLO model (~6MB)
RUN curl -L -o ./models/yolov8n_finetuned.pt \
    "https://huggingface.co/AzrilFahmiardi/yt-news-ribbon-yolov8n-detector/resolve/main/yolov8n_finetuned.pt"

# Download Whisper model (~142MB)
RUN curl -L -o ./models/ggml-base.bin \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"

# Download Qwen LLM (~1GB)
RUN curl -L -o ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Pre-download EasyOCR models (Indonesian + English, ~189MB)
RUN python -c "import easyocr; print('Pre-downloading EasyOCR models...'); reader = easyocr.Reader(['id', 'en'], gpu=False, verbose=True); print('EasyOCR models downloaded')"

# Verify all models
RUN echo "=== Downloaded Models ===" && ls -lh ./models/

# ============================================================
# Stage 5: Copy application code 
# ============================================================
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY main.py ./
COPY Makefile ./

# Create output directories
RUN mkdir -p output/segments output/logs output/debug

# ============================================================
# Stage 6: Setup entrypoint
# ============================================================
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Environment variables for runtime
ENV DISPLAY=:99
ENV PULSE_SERVER=unix:/tmp/pulse/native

# Expose health check port 
EXPOSE 8080

# Default command
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "main.py"]

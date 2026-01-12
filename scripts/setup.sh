#!/bin/bash

# Setup script for Automated News Livestream Intelligence System
# For Ubuntu 22.04

set -e

echo "Setting up Automated News Livestream Intelligence System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu
if [ ! -f /etc/lsb-release ]; then
    echo -e "${RED}This script is designed for Ubuntu${NC}"
    exit 1
fi

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
sudo apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libopencv-dev \
    cmake \
    build-essential \
    git \
    curl

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo -e "${GREEN}uv already installed${NC}"
fi

# Create virtual environment and install dependencies
echo -e "${YELLOW}Creating virtual environment and installing Python dependencies...${NC}"
uv venv
source .venv/bin/activate
uv pip install -e .

# Install patchright browser binaries
echo -e "${YELLOW}Installing patchright browser binaries...${NC}"
python -m patchright install chromium

# Verify installations
echo -e "${YELLOW}Verifying installations...${NC}"

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}FFmpeg installed: $(ffmpeg -version | head -n 1 | cut -d' ' -f3)${NC}"
else
    echo -e "${RED}FFmpeg not found${NC}"
fi

# Check Python packages
python -c "import patchright; print('✓ patchright installed')" 2>/dev/null || echo -e "${RED}✗ patchright not found${NC}"
python -c "import cv2; print('✓ opencv-python installed')" 2>/dev/null || echo -e "${RED}✗ opencv-python not found${NC}"
python -c "import easyocr; print('✓ easyocr installed')" 2>/dev/null || echo -e "${RED}✗ easyocr not found${NC}"
python -c "import ultralytics; print('✓ ultralytics (YOLO) installed')" 2>/dev/null || echo -e "${RED}✗ ultralytics not found${NC}"
python -c "import pywhispercpp; print('✓ pywhispercpp installed')" 2>/dev/null || echo -e "${RED}✗ pywhispercpp not found${NC}"
python -c "import llama_cpp; print('✓ llama-cpp-python installed')" 2>/dev/null || echo -e "${RED}✗ llama-cpp-python not found${NC}"

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Download AI models: bash scripts/download_models.sh"
echo "3. Run the system: python main.py"
echo ""
echo "For more information, see README.md"

#!/bin/bash

# Download AI models for the system

set -e

MODELS_DIR="./models"
CONFIG_FILE="./config/models.yaml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "Downloading AI models..."

# Create models directory if not exists
mkdir -p "$MODELS_DIR"

# Function to download file with progress
download_model() {
    local url=$1
    local output=$2
    local name=$3
    
    if [ -f "$output" ]; then
        echo -e "${GREEN}$name already exists, skipping${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}Downloading $name...${NC}"
    curl -L --progress-bar "$url" -o "$output"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Downloaded $name${NC}"
    else
        echo -e "${RED}Failed to download $name${NC}"
        return 1
    fi
}

# Download Whisper model (base)
echo ""
echo "=== Whisper Model ==="
download_model \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" \
    "$MODELS_DIR/ggml-base.bin" \
    "Whisper Base (142MB)"

# Download LLM (Qwen2.5-1.5B)
echo ""
echo "=== LLM Model ==="
echo -e "${YELLOW}This is a large file (~1GB), it may take a while...${NC}"
download_model \
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    "$MODELS_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    "Qwen2.5-1.5B-Instruct Q4_K_M (1GB)"

# Check total size
echo ""
echo "=== Summary ==="
TOTAL_SIZE=$(du -sh "$MODELS_DIR" | cut -f1)
echo -e "${GREEN}All models downloaded successfully${NC}"
echo -e "Total size: $TOTAL_SIZE"
echo ""
echo "Models location: $MODELS_DIR"
ls -lh "$MODELS_DIR"

echo ""
echo -e "${GREEN}Model download complete!${NC}"
echo "You can now run: python main.py"

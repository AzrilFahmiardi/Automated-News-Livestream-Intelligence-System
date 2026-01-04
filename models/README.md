# AI Models Directory

This directory stores downloaded AI models for the system.

## Required Models

### 1. LLM (Large Language Model)
**Recommended**: Qwen2.5-1.5B-Instruct-Q4_K_M
- **Size**: ~1GB
- **Purpose**: Reasoning and structured data extraction
- **Format**: GGUF

### 2. Whisper (Speech-to-Text)
**Recommended**: Whisper Base
- **Size**: ~142MB
- **Purpose**: Audio transcription (Bahasa Indonesia)
- **Format**: GGML binary

## Download

```bash
bash scripts/download_models.sh
```

## Manual Download

If automatic download fails:

### LLM Model
```bash
cd models
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### Whisper Model
```bash
cd models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

## Model Selection

See `config/models.yaml` for alternative model options.

### Smaller Models (if storage limited)
- **LLM**: Llama-3.2-1B (~800MB)
- **Whisper**: tiny (~75MB)

### Larger Models (if accuracy needed)
- **LLM**: Gemma-2-2B (~1.4GB)
- **Whisper**: small (~466MB)

## Verification

```bash
ls -lh models/
```

Expected output:
```
-rw-r--r-- 1 user user 1.0G Jan  4 19:00 qwen2.5-1.5b-instruct-q4_k_m.gguf
-rw-r--r-- 1 user user 142M Jan  4 19:05 ggml-base.bin
```

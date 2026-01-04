# Automated News Livestream Intelligence System

---

## Features

- **Browser Automation** - Patchright-python untuk stealth YouTube livestream capture
- **Speech-to-Text** - Whisper.cpp untuk transkrip audio Bahasa Indonesia
- **OCR Processing** - Tesseract untuk extract lower-third/ribbon text
- **LLM Reasoning** - Llama.cpp untuk ekstraksi data terstruktur
- **Segment Detection** - Heuristic-based boundary detection
- **Structured Output** - JSON per segmen dengan multi-actor support
- **CPU-only** - Tidak butuh GPU
- **Offline** - Semua processing lokal
- **Low Storage** - Total model ~1.2GB

---


## Project Structure

```
.
├── config/
│   ├── settings.yaml       # Main configuration
│   └── models.yaml         # Model download URLs
├── src/
│   ├── browser/            # Patchright automation
│   ├── video/              # OCR processing
│   ├── audio/              # Whisper speech-to-text
│   ├── llm/                # Llama reasoning
│   ├── segment/            # Segment detection
│   ├── pipeline/           # Main orchestrator
│   └── utils.py            # Utilities
├── models/                 # Downloaded AI models
├── output/
│   ├── segments/           # JSON outputs
│   └── logs/               # System logs
├── scripts/
│   ├── setup.sh            # Installation script
│   └── download_models.sh  # Model downloader
├── main.py                 # Entry point
├── pyproject.toml          # Dependencies (uv)
└── README.md
```

---

## Output Format

Each news segment produces a JSON file with this structure:

```json
{
  "segment": {
    "channel": "KompasTV",
    "segment_id": "kompastv_20260104_193210",
    "start_time": "2026-01-04T19:32:10",
    "end_time": "2026-01-04T19:36:40",
    "duration_sec": 270
  },
  "content": {
    "actors": [
      {
        "name": "Joko Widodo",
        "role": "Presiden Republik Indonesia",
        "source": ["ribbon", "speech"],
        "confidence": 0.96
      }
    ],
    "summary": {
      "short": "Peresmian proyek nasional di Jawa Tengah.",
      "full": "Presiden Joko Widodo meresmikan proyek infrastruktur..."
    },
    "topics": ["infrastruktur", "pemerintahan"]
  },
  "raw": {
    "speech_text": "Full transcription...",
    "ribbon_texts": [
      {
        "time": "2026-01-04T19:32:15",
        "text": "JOKO WIDODO | PRESIDEN RI",
        "confidence": 0.95
      }
    ]
  }
}
```
---

## Quick Start

### 1. Install
```bash
make install
# or
bash scripts/setup.sh && bash scripts/download_models.sh
```

### 2. Activate
```bash
source .venv/bin/activate
```

### 3. Run
```bash
python main.py
# or
make run
```

---

### Key Settings in `config/settings.yaml`

```yaml
channels:
  - name: "KompasTV"
    url: "https://www.youtube.com/watch?v=..."
    enabled: true

browser:
  headless: false  # MUST be false for anti-bot

video:
  fps_sample_rate: 0.5  # Sample every 2 seconds

audio:
  chunk_duration: 30
  whisper_model: "base"

llm:
  model_path: "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
  temperature: 0.3

segment:
  min_duration: 30
  max_duration: 600
```

---

## Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| LLM | Qwen2.5-1.5B-Instruct-Q4_K_M | ~1GB | Reasoning & extraction |
| Speech-to-Text | Whisper Base | ~142MB | Audio transcription |
| OCR | Tesseract | ~50MB | Lower-third text |

**Total**: ~1.2GB

---

## Credits

Built with:
- [patchright-python](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright) - Browser automation
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Speech recognition
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Text recognition

---

# Automated News Livestream Intelligence System

<p align="center">
  <img src="assets/banner.jpg" alt="Automated News Livestream Intelligence System" width="800"/>
</p>

An automated system for capturing, analyzing, and extracting structured information from Indonesian news livestreams using computer vision, speech recognition, and large language models.

This system monitors YouTube news livestreams and automatically extracts structured data including speaker information, topics, and summaries. It operates entirely offline on CPU-only hardware using lightweight AI models.


### Key Capabilities

- Automated capture of YouTube livestreams using stealth browser automation
- Real-time ribbon text detection and extraction using fine-tuned YOLOv8n and OCR
- Indonesian speech-to-text transcription
- LLM-based extraction of structured information
- Automatic segment detection and boundary identification
- JSON-formatted output with multi-actor support


## Architecture

The system consists of six main components organized in a processing pipeline:

### Components

**1. Browser Automation** (`src/browser/`) - Captures livestream frames using Patchright with anti-detection

**2. Vision Processing** (`src/vision/`) - Detects ribbons with fine-tuned YOLOv8n and extracts text via EasyOCR

**3. Audio Processing** (`src/audio/`) - Transcribes audio using Whisper.cpp (Indonesian language)

**4. LLM Reasoning** (`src/llm/`) - Extracts structured data using Qwen2.5-1.5B-Instruct via llama.cpp

**5. Segment Detection** (`src/segment/`) - Identifies segment boundaries based on content changes

**6. Pipeline Orchestration** (`src/pipeline/`) - Coordinates all components with async operations

### Project Structure

```
.
├── config/
│   └── settings.yaml           # System configuration
├── src/
│   ├── browser/                # Browser automation module
│   ├── vision/                 # YOLO + OCR processing
│   ├── audio/                  # Speech-to-text module
│   ├── llm/                    # LLM reasoning module
│   ├── segment/                # Segment detection
│   ├── pipeline/               # Orchestration logic
│   └── utils.py                # Shared utilities
├── models/                     # AI model files
├── output/
│   ├── segments/               # Generated JSON files
│   └── logs/                   # System logs
├── scripts/
│   ├── setup.sh                # System setup script
│   └── download_models.sh      # Model download script
├── main.py                     # Application entry point
├── Makefile                    # Build automation
└── pyproject.toml              # Python dependencies
```

## Installation


### Setup Instructions

1. Clone the repository and navigate to the project directory

2. Run the installation script:
```bash
make install
```
Alternatively, run the setup steps manually:
```bash
bash scripts/setup.sh
bash scripts/download_models.sh
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate
```

## Usage

### Basic Usage

Start the system with default configuration:
```bash
python main.py
```

Or using Make:
```bash
make run
```

### Command Line Options

```bash
# Specify configuration file
python main.py --config ./config/settings.yaml

# Process specific channel
python main.py --channel KompasTV

# Run in debug mode (no segmentation, continuous monitoring)
python main.py --debug
```

### Operating Modes

**Production Mode** (default) - Auto-segmentation with JSON output per segment

**Debug Mode** (`--debug`) - Continuous monitoring without segmentation, saves intermediate artifacts

## Configuration

The main configuration file is `config/settings.yaml`. Key settings include:

### Channel Configuration
```yaml
channels:
  - name: "KompasTV"
    url: "https://www.youtube.com/watch?v=..."
    enabled: true
    resolution: "1080p"
```

### Processing Parameters
```yaml
video:
  fps_sample_rate: 0.5          # Sample every 2 seconds
  frame_diff_threshold: 30

yolo_ribbon:
  conf_threshold: 0.3           # Detection confidence
  stability_frames: 2           # Frames before accepting detection
  text_similarity_threshold: 0.85

audio:
  chunk_duration: 30            # Seconds per chunk
  whisper_model: "base"
  language: "id"

llm:
  model_path: "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
  temperature: 0.3
  max_tokens: 1024
  n_threads: 4

segment:
  min_duration: 30              # Minimum segment length (seconds)
  max_duration: 600             # Maximum segment length (seconds)
  idle_threshold: 60            # Idle time before segment end
```

## Output Format

Each processed segment generates a JSON file in `output/segments/` with the following structure:

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
    "title": "Peresmian Infrastruktur Nasional",
    "actors": [
      {
        "name": "Joko Widodo",
        "role": "Presiden Republik Indonesia",
        "source": ["ribbon", "speech"],
        "confidence": 0.96
      }
    ],
    "summary": {
      "short": "Ringkasan singkat dari segmen berita",
      "full": "Ringkasan lengkap dengan detail dan konteks dari transkrip audio dan informasi ribbon"
    },
    "topics": ["infrastruktur", "pemerintahan", "pembangunan"]
  },
  "raw": {
    "speech_text": "Transkrip lengkap dari audio segmen...",
    "ribbon_texts": [
      {
        "time": "2026-01-04T19:32:15",
        "text": "JOKO WIDODO | PRESIDEN RI",
        "confidence": 0.95
      },
      {
        "time": "2026-01-04T19:33:20",
        "text": "BREAKING NEWS - PERESMIAN PROYEK",
        "confidence": 0.92
      }
    ]
  }
}
```



## AI Models

The system uses the following models:

| Component | Model | Size | Description |
|-----------|-------|------|-------------|
| Object Detection | yt-news-ribbon-yolov8n-detector | 6MB | Custom-trained on Indonesian news broadcast ribbons by [AzrilFahmiardi](https://huggingface.co/AzrilFahmiardi/yt-news-ribbon-yolov8n-detector) |
| Speech Recognition | Whisper Base | 142MB | Indonesian language transcription (ggml format) |
| OCR | EasyOCR | ~50MB | Indonesian and English text extraction |
| Language Model | Qwen2.5-1.5B-Instruct-Q4_K_M | 1GB | Structured information extraction (GGUF Q4_K_M quantized) |




## Acknowledgments

Built with the following open-source projects:

- [Patchright](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright) - Stealth browser automation
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Efficient speech recognition
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference engine
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [yt-news-ribbon-yolov8n-detector](https://huggingface.co/AzrilFahmiardi/yt-news-ribbon-yolov8n-detector) - Object detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Text recognition

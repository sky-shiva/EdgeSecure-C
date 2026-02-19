"# EdgeSecure-C" 
# EdgeSecure 🔒
**Local AI Meeting Suite — 100% On-Device, Zero Cloud**

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Phi-3 Mini Model (4-bit GGUF)
```bash
# Create models folder
mkdir -p models

# Download from HuggingFace (one-time, ~2.3GB)
# Option A — using huggingface-hub
pip install huggingface-hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='microsoft/Phi-3-mini-4k-instruct-gguf',
    filename='Phi-3-mini-4k-instruct-q4.gguf',
    local_dir='models/'
)
"
```

### 3. Run
```bash
cd src
python main.py
```

Press `Ctrl+C` to stop and generate the summary.

---

## How It Works

```
Mic → [every 30s] → .wav chunk → Whisper (int8) → transcript
                                                         ↓
                                    [every 10min] → Phi-3 Mini (4-bit) → chunk summary
                                                         ↓
                                    [meeting ends] → final JSON output
```

**Key design:** Whisper and Phi-3 Mini NEVER run simultaneously.
They take turns, keeping CPU/NPU load minimal so your Zoom/Teams call is unaffected.

---

## Output
Each meeting generates a JSON file in `output/`:
```json
{
  "meeting_date": "2026-02-19",
  "duration": "45 min",
  "full_transcript": "...",
  "action_items": ["Follow up with client by Friday"],
  "decisions_made": ["Budget approved at 50k"],
  "follow_up_email_draft": "Hi team...",
  "processing_mode": "100% Local | Zero Cloud"
}
```

---

## AMD NPU Optimization
To leverage the AMD Ryzen AI NPU for faster inference:

```bash
# For Phi-3 Mini — build llama-cpp with ROCm support
CMAKE_ARGS="-DLLAMA_ROCM=on" pip install llama-cpp-python --force-reinstall

# Then in summarizer.py, change:
n_gpu_layers=0   →   n_gpu_layers=-1
```

---

## Tech Stack
| Component | Tool | Why |
|-----------|------|-----|
| Transcription | faster-whisper (int8) | 60% smaller, minimal accuracy loss |
| Summarization | Phi-3 Mini (4-bit GGUF) | Best quality/speed for local use |
| Inference | llama-cpp-python | Optimized C++ backend, AMD compatible |
| Audio | PyAudio | Cross-platform mic capture |

---

## File Structure
```
EdgeSecure/
├── audio/          # Temp .wav chunks (auto-deleted after transcription)
├── transcripts/    # chunk_XXXX.txt + full_transcript.txt
├── summaries/      # summary_chunk_XXXX.json + final_summary.json
├── output/         # Final meeting JSON (permanent)
├── models/         # Local model files (never leave device)
├── src/
│   ├── main.py         # Entry point
│   ├── pipeline.py     # Coordinator
│   ├── audio_capture.py
│   ├── transcriber.py
│   └── summarizer.py
└── requirements.txt
```
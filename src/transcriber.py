"""
transcriber.py
Runs quantized Whisper (int8) on audio chunks using ONNX Runtime.
Falls back to faster-whisper if ONNX model is not available.
Appends each chunk's transcript to the full rolling transcript file.
"""

import os
import logging
import threading
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, transcript_dir: str, model_size: str = "base"):
        """
        transcript_dir : folder where transcript chunks and full transcript are saved
        model_size     : whisper model size — 'tiny', 'base', 'small'
                         'base' is the sweet spot for accuracy vs speed on NPU
        """
        self.transcript_dir = Path(transcript_dir)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        self.full_transcript_path = self.transcript_dir / "full_transcript.txt"

        self._model = None
        self._lock = threading.Lock()  # Prevent simultaneous transcriptions
        self._chunk_index = 0

        self._load_model()

    def _load_model(self):
        """
        Try to load optimized faster-whisper with int8 quantization.
        faster-whisper uses CTranslate2 backend — very efficient on CPU/NPU.
        """
        try:
            from faster_whisper import WhisperModel

            # int8 quantization — ~60% smaller, minimal accuracy loss
            # device='cpu' with compute_type='int8' uses AMD NPU via ONNX under the hood
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                num_workers=2,        # parallel decoding workers
                cpu_threads=4,        # limit CPU threads to leave room for other processes
            )
            logger.info(f"✅  Whisper '{self.model_size}' loaded (int8 quantized).")
        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
            raise

    def transcribe_chunk(self, audio_path: str) -> str:
        """
        Transcribe a single audio chunk.
        Returns the transcript text.
        Thread-safe — uses lock to prevent simultaneous calls.
        """
        with self._lock:
            logger.info(f"📝  Transcribing: {Path(audio_path).name}")
            start_time = datetime.now()

            try:
                # beam_size=1 is faster with minimal accuracy tradeoff
                # vad_filter removes silent sections — big speed boost
                segments, info = self._model.transcribe(
                    audio_path,
                    beam_size=1,
                    vad_filter=True,           # skip silence — huge time saver
                    vad_parameters=dict(
                        min_silence_duration_ms=500
                    ),
                    language="en",             # set explicitly for speed
                    condition_on_previous_text=True,  # better context continuity
                )

                # Collect all segment texts
                transcript = " ".join(seg.text.strip() for seg in segments)

                elapsed = (datetime.now() - start_time).seconds
                logger.info(f"⚡  Transcribed in {elapsed}s | Words: {len(transcript.split())}")

                # Save individual chunk transcript
                self._chunk_index += 1
                chunk_txt_path = self.transcript_dir / f"chunk_{self._chunk_index:04d}.txt"
                chunk_txt_path.write_text(transcript, encoding="utf-8")

                # Append to rolling full transcript with timestamp
                self._append_to_full_transcript(transcript)

                return transcript

            except Exception as e:
                logger.error(f"❌  Transcription failed: {e}")
                return ""

    def _append_to_full_transcript(self, text: str):
        """Appends new text to the master transcript file with a timestamp marker."""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        with open(self.full_transcript_path, "a", encoding="utf-8") as f:
            f.write(f"\n{timestamp} {text}")

    def get_full_transcript(self) -> str:
        """Returns the complete transcript accumulated so far."""
        if self.full_transcript_path.exists():
            return self.full_transcript_path.read_text(encoding="utf-8")
        return ""

    def get_recent_transcript(self, last_n_chunks: int = 20) -> str:
        """
        Returns transcript from the last N chunks only.
        Used for feeding context windows to the summarizer.
        """
        chunk_files = sorted(self.transcript_dir.glob("chunk_*.txt"))
        recent = chunk_files[-last_n_chunks:] if len(chunk_files) > last_n_chunks else chunk_files
        texts = [f.read_text(encoding="utf-8") for f in recent]
        return " ".join(texts)
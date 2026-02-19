"""
audio_capture.py
Captures microphone/system audio in 30-second chunks and saves them locally.
Deletes chunks after transcription to save disk space.
"""

import pyaudio
import wave
import os
import time
import threading
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Audio settings - optimal for Whisper
SAMPLE_RATE = 16000       # Whisper expects 16kHz
CHANNELS = 1              # Mono is sufficient and lighter
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHUNK_DURATION = 30       # seconds per audio chunk
FRAMES_PER_BUFFER = 1024


class AudioCapture:
    def __init__(self, audio_dir: str, on_chunk_ready=None):
        """
        audio_dir      : folder where .wav chunks are saved
        on_chunk_ready : callback(chunk_path) called when a new chunk is saved
        """
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.on_chunk_ready = on_chunk_ready

        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._recording = False
        self._thread = None
        self._chunk_index = 0

    def _get_chunk_path(self) -> Path:
        self._chunk_index += 1
        return self.audio_dir / f"chunk_{self._chunk_index:04d}.wav"

    def _record_loop(self):
        """Main recording loop — records 30s chunks back to back."""
        frames_per_chunk = SAMPLE_RATE * CHUNK_DURATION

        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )
        logger.info("🎙️  Audio capture started.")

        while self._recording:
            frames = []
            frames_recorded = 0

            # Record one chunk
            while self._recording and frames_recorded < frames_per_chunk:
                data = self._stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                frames.append(data)
                frames_recorded += FRAMES_PER_BUFFER

            if frames:
                chunk_path = self._get_chunk_path()
                self._save_wav(chunk_path, frames)
                logger.info(f"💾  Saved audio chunk: {chunk_path.name}")

                # Notify pipeline that a new chunk is ready
                if self.on_chunk_ready:
                    self.on_chunk_ready(str(chunk_path))

        self._stream.stop_stream()
        self._stream.close()
        logger.info("🛑  Audio capture stopped.")

    def _save_wav(self, path: Path, frames: list):
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self._pa.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

    def start(self):
        if self._recording:
            return
        self._recording = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._recording = False
        if self._thread:
            self._thread.join(timeout=5)
        self._pa.terminate()

    def delete_chunk(self, chunk_path: str):
        """Call this after a chunk has been transcribed — frees disk space."""
        try:
            os.remove(chunk_path)
            logger.debug(f"🗑️  Deleted processed chunk: {chunk_path}")
        except FileNotFoundError:
            pass
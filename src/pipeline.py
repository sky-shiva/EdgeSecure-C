"""
pipeline.py
The brain of EdgeSecure.
Coordinates AudioCapture → Transcriber → Summarizer in a non-overlapping schedule.
Whisper and Phi-3 NEVER run simultaneously — they take turns to keep CPU load low.
"""

import threading
import queue
import logging
import json
import os
from pathlib import Path
from datetime import datetime

from audio_capture import AudioCapture
from transcriber import Transcriber
from summarizer import Summarizer

logger = logging.getLogger(__name__)

# How many chunks to accumulate before triggering a summarization pass
# 20 chunks × 30s = 10 minutes of audio per summarization cycle
CHUNKS_PER_SUMMARY = 20


class Pipeline:
    def __init__(self, base_dir: str = "."):
        base = Path(base_dir)

        # Directories
        self.audio_dir = base / "audio"
        self.transcript_dir = base / "transcripts"
        self.summary_dir = base / "summaries"
        self.output_dir = base / "output"

        for d in [self.audio_dir, self.transcript_dir, self.summary_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Components
        self.capture = AudioCapture(
            audio_dir=str(self.audio_dir),
            on_chunk_ready=self._on_chunk_ready,
        )
        self.transcriber = Transcriber(transcript_dir=str(self.transcript_dir))
        self.summarizer = Summarizer(summary_dir=str(self.summary_dir))

        # State
        self._audio_queue = queue.Queue()   # chunks waiting to be transcribed
        self._chunk_count = 0              # total chunks processed
        self._running = False
        self._worker_thread = None
        self._meeting_start = None

        # Status for UI
        self.status = {
            "internet": "Offline",
            "ai_processing": "Idle",
            "chunks_processed": 0,
            "words_transcribed": 0,
            "last_action": "Ready",
        }

    # ─────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ─────────────────────────────────────────────

    def start_meeting(self):
        """Start recording and processing."""
        if self._running:
            logger.warning("Pipeline already running.")
            return

        self._running = True
        self._meeting_start = datetime.now()

        # Start background worker (processes audio queue)
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True
        )
        self._worker_thread.start()

        # Start audio capture
        self.capture.start()

        logger.info("🚀  EdgeSecure pipeline started.")
        self._update_status("ai_processing", "Listening")

    def stop_meeting(self) -> dict:
        """
        Stop recording, wait for pending chunks to finish,
        generate final summary, and return the output JSON.
        """
        logger.info("⏹️   Stopping pipeline...")
        self._update_status("ai_processing", "Finalizing")

        # Stop new audio from coming in
        self.capture.stop()
        self._running = False

        # Wait for the queue to drain
        self._audio_queue.join()

        # Generate final summary
        full_transcript = self.transcriber.get_full_transcript()
        final_summary = self.summarizer.generate_final_summary(full_transcript)

        # Build output JSON
        output = self._build_output(full_transcript, final_summary)
        self._save_output(output)

        self._update_status("ai_processing", "Complete")
        logger.info("✅  Meeting processing complete.")
        return output

    def get_live_transcript(self) -> str:
        """For UI — returns the rolling transcript accumulated so far."""
        return self.transcriber.get_full_transcript()

    def get_status(self) -> dict:
        return self.status

    # ─────────────────────────────────────────────
    # INTERNAL LOGIC
    # ─────────────────────────────────────────────

    def _on_chunk_ready(self, chunk_path: str):
        """Called by AudioCapture when a 30s chunk is saved. Adds to queue."""
        self._audio_queue.put(chunk_path)
        logger.debug(f"📥  Queued chunk: {Path(chunk_path).name}")

    def _worker_loop(self):
        """
        Main worker — processes one chunk at a time.
        KEY DESIGN: Whisper runs FIRST, then Phi-3 runs in the gap.
        They NEVER overlap.
        """
        while self._running or not self._audio_queue.empty():
            try:
                # Block for up to 2 seconds waiting for next chunk
                chunk_path = self._audio_queue.get(timeout=2)
            except queue.Empty:
                continue

            # ── Step 1: Transcribe (Whisper) ──
            self._update_status("ai_processing", "Transcribing")
            transcript = self.transcriber.transcribe_chunk(chunk_path)

            # Clean up raw audio immediately to save disk space
            self.capture.delete_chunk(chunk_path)

            self._chunk_count += 1
            self._update_status("chunks_processed", self._chunk_count)
            self._update_status(
                "words_transcribed",
                self.status["words_transcribed"] + len(transcript.split())
            )

            # Mark queue task as done
            self._audio_queue.task_done()

            # ── Step 2: Summarize every N chunks (Phi-3) ──
            # Only runs when Whisper is idle (queue is empty or we've hit the interval)
            if self._chunk_count % CHUNKS_PER_SUMMARY == 0:
                self._update_status("ai_processing", "Summarizing")
                recent = self.transcriber.get_recent_transcript(last_n_chunks=CHUNKS_PER_SUMMARY)
                self.summarizer.summarize_segment(recent)
                self._update_status("ai_processing", "Listening")

        logger.info("🏁  Worker loop finished.")

    def _build_output(self, full_transcript: str, final_summary: dict) -> dict:
        duration_minutes = int(
            (datetime.now() - self._meeting_start).seconds / 60
        ) if self._meeting_start else 0

        return {
            "meeting_date": datetime.now().strftime("%Y-%m-%d"),
            "meeting_time": self._meeting_start.strftime("%H:%M:%S") if self._meeting_start else "",
            "duration": f"{duration_minutes} min",
            "full_transcript": full_transcript,
            "action_items": final_summary.get("action_items", []),
            "decisions_made": final_summary.get("decisions_made", []),
            "key_discussion_points": final_summary.get("key_discussion_points", []),
            "follow_up_email_draft": final_summary.get("follow_up_email_draft", ""),
            "processing_mode": "100% Local | Zero Cloud",
            "words_transcribed": self.status["words_transcribed"],
            "chunks_processed": self._chunk_count,
        }

    def _save_output(self, output: dict):
        filename = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        logger.info(f"💾  Output saved: {output_path}")

    def _update_status(self, key: str, value):
        self.status[key] = value
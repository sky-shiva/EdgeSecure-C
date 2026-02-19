"""
summarizer.py
Runs Phi-3 Mini (4-bit quantized) locally for summarization.
Uses llama-cpp-python backend — extremely efficient on CPU/AMD NPU.
Processes transcript in chunks to handle long meetings within context limits.
"""

import logging
import threading
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Context window safe limit — Phi-3 Mini has 4k-128k context depending on version
# We cap at 3000 words per chunk to stay well within limits and keep it fast
MAX_WORDS_PER_CHUNK = 3000

SUMMARY_SYSTEM_PROMPT = """You are a professional meeting assistant. Your job is to analyze meeting transcripts and extract structured information accurately.

Rules:
- Only include information that was EXPLICITLY mentioned in the transcript
- Never invent, assume, or hallucinate details
- If something is unclear, mark it with [UNCLEAR] 
- Be concise and professional
- Output valid JSON only"""

CHUNK_SUMMARY_PROMPT = """Analyze this meeting transcript segment and extract:
1. Key discussion points
2. Any decisions made
3. Any action items mentioned (with owner if stated)

Transcript:
{transcript}

Respond in this JSON format:
{{
  "key_points": ["point 1", "point 2"],
  "decisions": ["decision 1"],
  "action_items": ["action 1 - Owner: name or UNASSIGNED"]
}}"""

FINAL_SUMMARY_PROMPT = """You have these partial summaries from different segments of a meeting:

{partial_summaries}

Combine them into one final comprehensive meeting summary.

Respond in this JSON format:
{{
  "action_items": ["item 1", "item 2"],
  "decisions_made": ["decision 1"],
  "key_discussion_points": ["point 1", "point 2"],
  "follow_up_email_draft": "Professional email draft here summarizing the meeting outcomes"
}}"""


class Summarizer:
    def __init__(self, summary_dir: str, model_path: str = None):
        """
        summary_dir : folder where summaries are saved
        model_path  : path to GGUF model file (Phi-3 Mini 4-bit)
                      Downloads automatically if not provided
        """
        self.summary_dir = Path(summary_dir)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path

        self._model = None
        self._lock = threading.Lock()
        self._partial_summaries = []
        self._summary_index = 0

        self._load_model()

    def _load_model(self):
        """Load Phi-3 Mini via llama-cpp-python with 4-bit quantization."""
        try:
            from llama_cpp import Llama

            # If no model path, use a default GGUF path
            # Download: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
            model_path = self.model_path or "models/phi-3-mini-4k-instruct-q4.gguf"

            self._model = Llama(
                model_path=model_path,
                n_ctx=4096,          # context window
                n_threads=4,         # CPU threads — leave headroom for Whisper
                n_gpu_layers=0,      # set to -1 if you have GPU/NPU support via ROCm
                verbose=False,
                use_mmap=True,       # memory-map model file — faster load, less RAM
                use_mlock=False,
            )
            logger.info("✅  Phi-3 Mini loaded (4-bit quantized).")
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            raise
        except Exception as e:
            logger.error(f"❌  Failed to load Phi-3 Mini: {e}")
            raise

    def _run_inference(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Run inference on the local model.
        Returns raw text output.
        """
        with self._lock:
            response = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.1,      # low temp = more deterministic, less hallucination
                top_p=0.9,
                stop=["```", "###"],  # stop tokens
                echo=False,
            )
            return response["choices"][0]["text"].strip()

    def _chunk_transcript(self, transcript: str) -> list:
        """Split transcript into word-count-limited chunks."""
        words = transcript.split()
        chunks = []
        for i in range(0, len(words), MAX_WORDS_PER_CHUNK):
            chunk = " ".join(words[i:i + MAX_WORDS_PER_CHUNK])
            chunks.append(chunk)
        return chunks

    def _parse_json_safely(self, text: str) -> dict:
        """Extract JSON from model output safely."""
        # Find JSON block in the output
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        # Return empty structure if parsing fails
        return {"key_points": [], "decisions": [], "action_items": []}

    def summarize_segment(self, transcript: str) -> dict:
        """
        Summarize a 10-minute transcript segment.
        Called periodically during the meeting.
        """
        if not transcript.strip():
            return {}

        logger.info(f"🧠  Summarizing segment ({len(transcript.split())} words)...")

        prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\n{CHUNK_SUMMARY_PROMPT.format(transcript=transcript)}"

        raw_output = self._run_inference(prompt, max_tokens=400)
        summary = self._parse_json_safely(raw_output)

        # Store for final compilation
        self._partial_summaries.append(summary)

        # Save to disk
        self._summary_index += 1
        summary_path = self.summary_dir / f"summary_chunk_{self._summary_index:04d}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.info(f"✅  Segment summary saved: {summary_path.name}")
        return summary

    def generate_final_summary(self, full_transcript: str) -> dict:
        """
        Generate the complete meeting summary at the end.
        Uses hierarchical approach — summarizes the partial summaries.
        """
        logger.info("🎯  Generating final meeting summary...")

        # If the transcript fits in one chunk, summarize directly
        chunks = self._chunk_transcript(full_transcript)

        if len(chunks) == 1 and not self._partial_summaries:
            # Short meeting — direct summarization
            partial = [self.summarize_segment(full_transcript)]
        else:
            partial = self._partial_summaries

        # Combine all partial summaries
        partial_text = json.dumps(partial, indent=2)

        prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\n{FINAL_SUMMARY_PROMPT.format(partial_summaries=partial_text)}"
        raw_output = self._run_inference(prompt, max_tokens=800)
        final = self._parse_json_safely(raw_output)

        # Add metadata
        final["generated_at"] = datetime.now().isoformat()
        final["processing_mode"] = "100% Local | Zero Cloud"

        # Save final summary
        final_path = self.summary_dir / "final_summary.json"
        final_path.write_text(json.dumps(final, indent=2), encoding="utf-8")

        logger.info("✅  Final summary complete.")
        return final

    def get_partial_summaries(self) -> list:
        return self._partial_summaries
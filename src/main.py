"""
main.py
EdgeSecure — Entry point.
Starts the pipeline, shows live status in terminal, handles graceful shutdown.
Run: python main.py
"""

import os
import sys
import signal
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import Pipeline

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("edgesecure.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("EdgeSecure")

# ── Banner ──
BANNER = """
╔══════════════════════════════════════════════════════╗
║          EdgeSecure — Local AI Meeting Suite         ║
║    🔒 Internet: OFFLINE  |  🧠 AI: 100% On-Device   ║
╚══════════════════════════════════════════════════════╝
"""


def print_status(pipeline: Pipeline):
    """Print a live status line to the terminal."""
    s = pipeline.get_status()
    print(
        f"\r⚡ Status: {s['ai_processing']:15s} | "
        f"Chunks: {s['chunks_processed']:4d} | "
        f"Words: {s['words_transcribed']:6d} | "
        f"Internet: {s['internet']}   ",
        end="",
        flush=True,
    )


def print_output(output: dict):
    """Pretty-print the final meeting output."""
    print("\n\n" + "═" * 60)
    print("  📋  MEETING SUMMARY")
    print("═" * 60)
    print(f"  Date     : {output['meeting_date']}")
    print(f"  Duration : {output['duration']}")
    print(f"  Words    : {output['words_transcribed']}")
    print(f"  Mode     : {output['processing_mode']}")
    print()

    print("  ✅  ACTION ITEMS:")
    for i, item in enumerate(output.get("action_items", []), 1):
        print(f"     {i}. {item}")

    print("\n  🏛️   DECISIONS MADE:")
    for i, d in enumerate(output.get("decisions_made", []), 1):
        print(f"     {i}. {d}")

    print("\n  💬  KEY DISCUSSION POINTS:")
    for i, p in enumerate(output.get("key_discussion_points", []), 1):
        print(f"     {i}. {p}")

    print("\n  📧  FOLLOW-UP EMAIL DRAFT:")
    print("  " + "-" * 50)
    email = output.get("follow_up_email_draft", "")
    for line in email.split("\n"):
        print(f"  {line}")

    print("═" * 60)
    print(f"\n  Output saved to: output/meeting_{output['meeting_date']}_*.json\n")


def main():
    print(BANNER)

    # Initialize pipeline
    base_dir = Path(__file__).parent.parent  # EdgeSecure root
    pipeline = Pipeline(base_dir=str(base_dir))

    # Graceful shutdown on Ctrl+C
    shutdown_event = {"triggered": False}
    output_data = {}

    def handle_shutdown(sig, frame):
        if not shutdown_event["triggered"]:
            shutdown_event["triggered"] = True
            print("\n\n⏹️   Stopping recording and generating summary...")
            result = pipeline.stop_meeting()
            output_data.update(result)
            print_output(result)
            sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Start
    print("  Press Ctrl+C to stop recording and generate summary.\n")
    pipeline.start_meeting()

    # Status update loop
    try:
        while True:
            print_status(pipeline)
            time.sleep(1)
    except KeyboardInterrupt:
        handle_shutdown(None, None)


if __name__ == "__main__":
    main()
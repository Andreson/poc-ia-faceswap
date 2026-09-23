"""CLI entrypoint for local pipeline validation. See REQ-OPS-004 / REQ-OPS-005.

Usage:
    python cli_swap.py --source-image face.jpg --target-video clip.mp4 --output-video out.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline import VideoFaceSwapperPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local face-swap pipeline runner.")
    parser.add_argument("--source-image", required=True, type=Path)
    parser.add_argument("--target-video", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = os.environ.get("MODELS_DIR")

    pipeline = VideoFaceSwapperPipeline(device=args.device, models_dir=models_dir)
    result = pipeline.process_video(
        source_img_path=args.source_image,
        video_path=args.target_video,
        output_path=args.output_video,
        progress_callback=lambda pct: print(f"progress: {pct}%"),
    )
    print(f"done: {result.output_path}")


if __name__ == "__main__":
    main()

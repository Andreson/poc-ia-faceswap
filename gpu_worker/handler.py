"""RunPod Serverless entrypoint. See .harness/03-ai_pipeline_specification_harness.md REQ-AI-012.

Wraps VideoFaceSwapperPipeline (pipeline.py) for RunPod's `runpod.serverless.start`
runtime. Test locally with:
    python handler.py --test_input test_input.json
"""

from __future__ import annotations

import os
from pathlib import Path

import runpod

from pipeline import VideoFaceSwapperPipeline
from storage import download_from_r2, upload_to_r2

MODELS_DIR = os.environ.get("MODELS_DIR")
DEVICE = os.environ.get("DEVICE", "cuda")

# REQ-AI-010: load once at cold start and fail fast if the Network Volume
# isn't mounted, instead of on every job invocation.
pipeline = VideoFaceSwapperPipeline(device=DEVICE, models_dir=MODELS_DIR)


def handler(event):
    job_input = event["input"]
    job_id = job_input["job_id"]

    try:
        source_path = download_from_r2(job_input["source_image_key"])
        video_path = download_from_r2(job_input["target_video_key"])
        output_path = Path(f"/tmp/{job_id}_output.mp4")

        result = pipeline.process_video(
            source_img_path=source_path,
            video_path=video_path,
            output_path=output_path,
            progress_callback=lambda pct: runpod.serverless.progress_update(
                event, {"progress": pct}
            ),  # REQ-AI-013
        )

        output_key = f"processed/{job_id}/output.mp4"
        upload_to_r2(result.output_path, output_key)
    except Exception as exc:
        # REQ-AI-011: mark FAILED with a descriptive message, don't crash the worker.
        return {"error": str(exc), "job_id": job_id}

    return {"output_video_key": output_key, "job_id": job_id}


runpod.serverless.start({"handler": handler})



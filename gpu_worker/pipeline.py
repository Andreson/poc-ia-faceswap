"""Face-swap pipeline core logic. See .harness/03-ai_pipeline_specification_harness.md.

Implements REQ-AI-001 through REQ-AI-011 for local/CLI use (REQ-OPS-004). The
RunPod handler entrypoint (REQ-AI-012, Phase 4) wraps this same class.
"""

from __future__ import annotations

import gc
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import ffmpeg
import insightface
import torch

from enhancer import CodeFormerRestorer

FRAME_CHECKPOINT_INTERVAL = 100  # REQ-AI-007, REQ-AI-013
MAX_WORKING_HEIGHT = 1080  # REQ-AI-008
SWAPPER_MODEL_FILE = "inswapper_128.onnx"


@dataclass
class ProcessingResult:
    output_path: Path
    had_audio: bool


def _bbox_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _largest_face(faces):
    return max(faces, key=_bbox_area)


class VideoFaceSwapperPipeline:
    """Wraps InsightFace + CodeFormer + FFmpeg. See REQ-AI-001 through REQ-AI-011."""

    def __init__(self, device: str = "cuda", models_dir: Optional[str] = None) -> None:
        # REQ-AI-010: models_dir must be pre-populated; fail fast rather than download.
        if not models_dir:
            raise RuntimeError(
                "MODELS_DIR is not set. Model weights must be pre-populated locally "
                "(or on the RunPod Network Volume in production) — the worker does not "
                "download them at runtime (REQ-AI-010)."
            )
        self.device = device
        self.models_dir = Path(models_dir)
        if not self.models_dir.is_dir():
            raise RuntimeError(f"MODELS_DIR does not exist: {self.models_dir} (REQ-AI-010).")
        self._load_models()

    def _load_models(self) -> None:
        provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"

        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=[provider], root=str(self.models_dir)
        )
        self.face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

        swapper_path = self.models_dir / SWAPPER_MODEL_FILE
        if not swapper_path.exists():
            raise RuntimeError(f"Swap model not found at {swapper_path} (REQ-AI-010).")
        self.swapper = insightface.model_zoo.get_model(str(swapper_path), download=False)

        self.restorer = CodeFormerRestorer(models_dir=self.models_dir, device=self.device)

    def extract_source_embedding(self, source_img_path: Path):
        img = cv2.imread(str(source_img_path))
        if img is None:
            raise ValueError(f"Could not read source image: {source_img_path}")
        faces = self.face_app.get(img)
        if not faces:
            raise ValueError("No face detected in source image.")  # REQ-AI-001
        return _largest_face(faces)  # REQ-AI-002

    def process_video(
        self,
        source_img_path: Path,
        video_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> ProcessingResult:
        source_face = self.extract_source_embedding(source_img_path)  # REQ-AI-001

        work_dir = output_path.parent / f".{output_path.stem}_frames"
        in_frames_dir = work_dir / "in"
        out_frames_dir = work_dir / "out"
        in_frames_dir.mkdir(parents=True, exist_ok=True)
        out_frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            fps, height, had_audio = self._probe_video(video_path)
            self._extract_frames(video_path, in_frames_dir, needs_downscale=height > MAX_WORKING_HEIGHT)

            audio_path = None
            if had_audio:
                audio_path = work_dir / "audio.aac"
                self._extract_audio(video_path, audio_path)

            frame_files = sorted(in_frames_dir.iterdir())
            total = len(frame_files)
            for index, frame_file in enumerate(frame_files, start=1):
                frame = cv2.imread(str(frame_file))
                faces = self.face_app.get(frame)
                if faces:
                    target_face = _largest_face(faces)  # REQ-AI-003
                    frame = self.swapper.get(frame, target_face, source_face, paste_back=True)
                    frame = self.restorer.enhance(frame)
                # else: REQ-AI-004 — leave the frame unmodified.

                cv2.imwrite(str(out_frames_dir / frame_file.name), frame)

                if index % FRAME_CHECKPOINT_INTERVAL == 0 or index == total:
                    gc.collect()  # REQ-AI-007
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    if progress_callback is not None:
                        progress_callback(int(index / total * 100))  # REQ-AI-013

            self._assemble_video(out_frames_dir, fps, audio_path, output_path)  # REQ-AI-005 / REQ-AI-006
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        return ProcessingResult(output_path=output_path, had_audio=had_audio)

    # -- FFmpeg helpers -------------------------------------------------

    def _probe_video(self, video_path: Path) -> tuple[float, int, bool]:
        try:
            info = ffmpeg.probe(str(video_path))
        except ffmpeg.Error as exc:
            raise RuntimeError(f"FFmpeg failed to read {video_path}: {exc.stderr.decode()}") from exc  # REQ-AI-011

        video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
        if video_stream is None:
            raise RuntimeError(f"No video stream found in {video_path}.")
        had_audio = any(s["codec_type"] == "audio" for s in info["streams"])  # REQ-AI-006

        num, _, den = (video_stream.get("avg_frame_rate") or "30/1").partition("/")
        fps = float(num) / float(den) if den and float(den) else 30.0
        return fps, int(video_stream["height"]), had_audio

    def _extract_frames(self, video_path: Path, frames_dir: Path, needs_downscale: bool) -> None:
        stream = ffmpeg.input(str(video_path))
        if needs_downscale:
            stream = stream.filter("scale", -2, MAX_WORKING_HEIGHT)  # REQ-AI-008
        stream = stream.output(str(frames_dir / "%06d.png"), start_number=0)
        try:
            stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as exc:
            raise RuntimeError(f"FFmpeg frame extraction failed: {exc.stderr.decode()}") from exc  # REQ-AI-011

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        try:
            (
                ffmpeg.input(str(video_path))
                .output(str(audio_path), acodec="copy", vn=None)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            raise RuntimeError(f"FFmpeg audio extraction failed: {exc.stderr.decode()}") from exc  # REQ-AI-011

    def _assemble_video(
        self, frames_dir: Path, fps: float, audio_path: Optional[Path], output_path: Path
    ) -> None:
        video_in = ffmpeg.input(str(frames_dir / "%06d.png"), framerate=fps)
        try:
            if audio_path is not None and audio_path.exists():
                audio_in = ffmpeg.input(str(audio_path))
                ffmpeg.output(
                    video_in,
                    audio_in,
                    str(output_path),
                    vcodec="libx264",
                    acodec="aac",
                    pix_fmt="yuv420p",
                    shortest=None,
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            else:
                video_in.output(
                    str(output_path), vcodec="libx264", pix_fmt="yuv420p"
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as exc:
            raise RuntimeError(f"FFmpeg re-assembly failed: {exc.stderr.decode()}") from exc  # REQ-AI-011

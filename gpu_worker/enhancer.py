"""CodeFormer-based facial enhancement. See .harness/03-ai_pipeline_specification_harness.md.

CodeFormer has no PyPI package (see requirements.txt) — this wraps the model
architecture from `basicsr` and the face crop/paste-back helper from `facexlib`,
the same building blocks CodeFormer's own inference script depends on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

CODEFORMER_WEIGHTS = "codeformer.pth"
FIDELITY_WEIGHT = 0.7  # balances identity fidelity vs. restoration strength


class CodeFormerRestorer:
    """Enhances facial resolution in a full BGR frame (REQ-AI: pipeline step 3)."""

    def __init__(self, models_dir: Path, device: str = "cuda") -> None:
        ckpt_path = models_dir / CODEFORMER_WEIGHTS
        if not ckpt_path.exists():
            raise RuntimeError(f"CodeFormer checkpoint not found at {ckpt_path} (REQ-AI-010).")

        self.device = device
        self.net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        self.net.load_state_dict(checkpoint["params_ema"])
        self.net.eval()

        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
            model_rootpath=str(models_dir),
        )

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        self.face_helper.clean_all()
        self.face_helper.read_image(frame)
        self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        self.face_helper.align_warp_face()

        for cropped_face in self.face_helper.cropped_faces:
            face_tensor = _to_tensor(cropped_face).unsqueeze(0).to(self.device)
            try:
                with torch.no_grad():
                    output = self.net(face_tensor, w=FIDELITY_WEIGHT, adain=True)[0]
                    restored_face = _tensor_to_image(output)
            except RuntimeError:
                restored_face = cropped_face
            self.face_helper.add_restored_face(restored_face)

        self.face_helper.get_inverse_affine(None)
        return self.face_helper.paste_faces_to_input_image()


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    tensor.sub_(0.5).div_(0.5)
    return tensor


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.squeeze(0).clamp_(-1, 1)
    tensor = (tensor + 1) / 2
    img = tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    return img.round().clip(0, 255).astype("uint8")

"""
Shared face encoder + age head for demos (CLI / Gradio).
Matches training forward in models/fr.py: backbone(..., return_age=True) -> estimation_network(x_age).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def add_repo_to_path() -> None:
    root = repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def group_labels(age_group: int) -> list[str]:
    """Human-readable bins (same boundaries as common.ops.age2group)."""
    if age_group == 7:
        return [
            "0-9",
            "10-19",
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60+",
        ]
    if age_group == 8:
        return [
            "0-12",
            "13-18",
            "19-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "66+",
        ]
    if age_group == 6:
        return ["0-9", "10-19", "20-29", "30-39", "40-49", "50+"]
    if age_group == 5:
        return ["0-19", "20-29", "30-39", "40-49", "50+"]
    if age_group == 4:
        return ["0-29", "30-39", "40-49", "50+"]
    return [f"group_{i}" for i in range(age_group)]


@dataclass
class AgePrediction:
    age_years: float
    group_id: int
    group_label: str
    group_confidence: float


class FaceEngine:
    def __init__(
        self,
        backbone_name: str = "ir50",
        image_size: int = 112,
        age_group: int = 7,
        backbone_ckpt: str | None = None,
        age_ckpt: str | None = None,
        device: torch.device | None = None,
    ):
        add_repo_to_path()
        from backbone.aifr import backbone_dict, AgeEstimationModule
        from common.ops import load_network, get_dex_age

        self._get_dex_age = get_dex_age
        self.image_size = image_size
        self.age_group = age_group
        self.group_label_list = group_labels(age_group)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        self.backbone = backbone_dict[backbone_name](input_size=image_size).to(device).eval()
        self.age_net = AgeEstimationModule(input_size=image_size, age_group=age_group).to(device).eval()

        if backbone_ckpt:
            sd = load_network(backbone_ckpt)
            self.backbone.load_state_dict(sd, strict=False)
        if age_ckpt:
            sd = load_network(age_ckpt)
            self.age_net.load_state_dict(sd, strict=False)

        self._has_age = bool(age_ckpt)
        self._has_backbone = bool(backbone_ckpt)

        self.tfm = transforms.Compose(
            [
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def _pil_from_any(self, img) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            return Image.fromarray(img, "RGB")
        raise TypeError(f"Unsupported image type: {type(img)}")

    @torch.inference_mode()
    def encode_embedding(self, img) -> torch.Tensor:
        """L2-normalized 512-D embedding (same as verification)."""
        pil = self._pil_from_any(img)
        x = self.tfm(pil).unsqueeze(0).to(self.device)
        emb = self.backbone(x)
        emb = F.normalize(emb, dim=1)
        return emb.squeeze(0)

    @torch.inference_mode()
    def predict_age(self, img) -> AgePrediction | None:
        if not self._has_age:
            return None
        pil = self._pil_from_any(img)
        x = self.tfm(pil).unsqueeze(0).to(self.device)
        _, _, x_age = self.backbone(x, return_age=True)
        age_logits, group_logits = self.age_net(x_age)
        pred_age = self._get_dex_age(age_logits).squeeze(0).item()
        probs = F.softmax(group_logits, dim=1).squeeze(0)
        gid = int(probs.argmax().item())
        conf = float(probs[gid].item())
        label = self.group_label_list[gid] if gid < len(self.group_label_list) else str(gid)
        return AgePrediction(age_years=pred_age, group_id=gid, group_label=label, group_confidence=conf)

    def verify_pair(self, img_a, img_b) -> tuple[float, str]:
        e1 = self.encode_embedding(img_a)
        e2 = self.encode_embedding(img_b)
        sim = float((e1 * e2).sum().item())
        if self._has_backbone:
            return sim, "ok"
        return sim, "random_weights"

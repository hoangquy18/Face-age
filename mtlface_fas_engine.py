"""
FAS (Face Age Synthesis) inference: frozen AIResNet + AgingModule, same as models/fas.py train step.
"""

from __future__ import annotations

import os
import sys
import warnings

import torch
from PIL import Image
from torch import amp
from torch import nn
from torchvision import transforms


def _sanitize_batchnorm2d(module: nn.Module) -> int:
    """
    Checkpoint generator-* can contain corrupt BatchNorm running_mean / running_var (NaN/Inf),
    which makes forward produce NaN and black images after denorm. Repair in-place for inference.
    Returns number of BatchNorm2d layers touched.
    """
    n = 0
    for m in module.modules():
        if not isinstance(m, nn.BatchNorm2d):
            continue
        touched = False
        if m.running_mean is not None and not torch.isfinite(m.running_mean).all():
            m.running_mean.data.copy_(
                torch.nan_to_num(m.running_mean.data, nan=0.0, posinf=0.0, neginf=0.0)
            )
            touched = True
        if m.running_var is not None and not torch.isfinite(m.running_var).all():
            rv = torch.nan_to_num(m.running_var.data, nan=1.0, posinf=1.0, neginf=1.0)
            m.running_var.data.copy_(rv.clamp(min=1e-3))
            touched = True
        if m.weight is not None and not torch.isfinite(m.weight).all():
            m.weight.data.copy_(torch.nan_to_num(m.weight.data, nan=1.0))
            touched = True
        if m.bias is not None and not torch.isfinite(m.bias).all():
            m.bias.data.copy_(torch.nan_to_num(m.bias.data, nan=0.0))
            touched = True
        if touched:
            n += 1
    return n


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def add_repo_to_path() -> None:
    root = repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


class FASEngine:
    """Backbone (shortcuts + x_id/x_age) + AgingModule; checkpoints from joint FR+FAS training."""

    def __init__(
        self,
        backbone_name: str = "ir50",
        image_size: int = 112,
        age_group: int = 7,
        backbone_ckpt: str | None = None,
        generator_ckpt: str | None = None,
        device: torch.device | None = None,
    ):
        add_repo_to_path()
        from backbone.aifr import backbone_dict
        from common.networks import AgingModule
        from common.ops import load_network

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.image_size = image_size
        self.age_group = age_group

        self.backbone = backbone_dict[backbone_name](input_size=image_size).to(device).eval()
        self.generator = AgingModule(age_group=age_group).to(device).eval()

        if backbone_ckpt:
            sd = load_network(backbone_ckpt)
            self.backbone.load_state_dict(sd, strict=False)
        if generator_ckpt:
            sd = load_network(generator_ckpt)
            missing, unexpected = self.generator.load_state_dict(sd, strict=False)
            if missing or unexpected:
                warnings.warn(
                    f"FAS generator load_state_dict: missing={len(missing)} unexpected={len(unexpected)}",
                    stacklevel=2,
                )
            bad_bn = _sanitize_batchnorm2d(self.generator)
            if bad_bn:
                warnings.warn(
                    f"FAS generator: repaired non-finite values in {bad_bn} BatchNorm2d layer(s). "
                    "Your checkpoint may be from an unstable training step; consider re-saving or training longer.",
                    stacklevel=2,
                )

        self._ready = bool(backbone_ckpt and generator_ckpt)

        self.tfm = transforms.Compose(
            [
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    @property
    def ready(self) -> bool:
        return self._ready

    @torch.inference_mode()
    def synthesize(
        self,
        img: Image.Image,
        target_group_id: int,
        *,
        residual_scale: float = 1.0,
        generator_train_mode: bool = False,
    ) -> Image.Image:
        if not self._ready:
            raise RuntimeError("FASEngine: load backbone + generator checkpoints first.")
        pil = img.convert("RGB")
        x = self.tfm(pil).unsqueeze(0).to(self.device)
        cond = torch.tensor([int(target_group_id)], device=self.device, dtype=torch.long)

        use_cuda_amp = self.device.type == "cuda"
        prev_train = self.generator.training
        self.generator.train(generator_train_mode)

        def forward_backbone():
            return self.backbone(x, return_shortcuts=True)

        def forward_gen(x_1, x_2, x_3, x_4, x_5, x_id, x_age):
            return self.generator(
                x,
                x_1,
                x_2,
                x_3,
                x_4,
                x_5,
                x_id,
                x_age,
                condition=cond,
                residual_scale=float(residual_scale),
            )

        try:
            if use_cuda_amp:
                with amp.autocast("cuda", enabled=True):
                    x_1, x_2, x_3, x_4, x_5, x_id, x_age = forward_backbone()
                x_1 = x_1.float()
                x_2 = x_2.float()
                x_3 = x_3.float()
                x_4 = x_4.float()
                x_5 = x_5.float()
                x_id = x_id.float()
                x_age = x_age.float()
                with amp.autocast("cuda", enabled=True):
                    y = forward_gen(x_1, x_2, x_3, x_4, x_5, x_id, x_age)
            else:
                x_1, x_2, x_3, x_4, x_5, x_id, x_age = forward_backbone()
                x_1 = x_1.float()
                x_2 = x_2.float()
                x_3 = x_3.float()
                x_4 = x_4.float()
                x_5 = x_5.float()
                x_id = x_id.float()
                x_age = x_age.float()
                y = forward_gen(x_1, x_2, x_3, x_4, x_5, x_id, x_age)
        finally:
            self.generator.train(prev_train)

        y = y.float().squeeze(0).cpu().clamp(-1.0, 1.0)
        y = y * 0.5 + 0.5
        y = y.clamp(0.0, 1.0)
        return transforms.functional.to_pil_image(y)

    @torch.inference_mode()
    def mean_abs_pixel_diff_between_groups(
        self,
        img: Image.Image,
        group_a: int,
        group_b: int,
        *,
        residual_scale: float = 1.0,
        generator_train_mode: bool = False,
    ) -> float:
        """
        One backbone forward, two generator forwards (different `condition`).
        Returns mean |pixel_a - pixel_b| on 0–255 scale (inputs are in [0,1] before scaling).
        """
        if not self._ready:
            raise RuntimeError("FASEngine: load backbone + generator checkpoints first.")
        if int(group_a) == int(group_b):
            return 0.0
        pil = img.convert("RGB")
        x = self.tfm(pil).unsqueeze(0).to(self.device)
        use_cuda_amp = self.device.type == "cuda"
        prev_train = self.generator.training
        self.generator.train(generator_train_mode)

        def forward_backbone():
            return self.backbone(x, return_shortcuts=True)

        def run_gen(cond_id: int):
            cond = torch.tensor([int(cond_id)], device=self.device, dtype=torch.long)
            return self.generator(
                x,
                x_1,
                x_2,
                x_3,
                x_4,
                x_5,
                x_id,
                x_age,
                condition=cond,
                residual_scale=float(residual_scale),
            )

        try:
            if use_cuda_amp:
                with amp.autocast("cuda", enabled=True):
                    x_1, x_2, x_3, x_4, x_5, x_id, x_age = forward_backbone()
                x_1 = x_1.float()
                x_2 = x_2.float()
                x_3 = x_3.float()
                x_4 = x_4.float()
                x_5 = x_5.float()
                x_id = x_id.float()
                x_age = x_age.float()
                with amp.autocast("cuda", enabled=True):
                    y_a = run_gen(int(group_a))
                with amp.autocast("cuda", enabled=True):
                    y_b = run_gen(int(group_b))
            else:
                x_1, x_2, x_3, x_4, x_5, x_id, x_age = forward_backbone()
                x_1 = x_1.float()
                x_2 = x_2.float()
                x_3 = x_3.float()
                x_4 = x_4.float()
                x_5 = x_5.float()
                x_id = x_id.float()
                x_age = x_age.float()
                y_a = run_gen(int(group_a))
                y_b = run_gen(int(group_b))
        finally:
            self.generator.train(prev_train)

        def to_01(t):
            t = t.float().squeeze(0).cpu().clamp(-1.0, 1.0)
            t = t * 0.5 + 0.5
            return t.clamp(0.0, 1.0)

        ya = to_01(y_a)
        yb = to_01(y_b)
        return float((ya - yb).abs().mean().item() * 255.0)

    @torch.inference_mode()
    def synthesize_all_groups(
        self,
        img: Image.Image,
        *,
        residual_scale: float = 1.0,
        generator_train_mode: bool = False,
    ) -> list[Image.Image]:
        out: list[Image.Image] = []
        for g in range(self.age_group):
            out.append(
                self.synthesize(
                    img,
                    g,
                    residual_scale=residual_scale,
                    generator_train_mode=generator_train_mode,
                )
            )
        return out


def discover_fas_iteration(weights_dir: str) -> int | None:
    """Largest iteration N such that backbone-N and generator-N both exist."""
    import glob

    def iters(prefix: str) -> set[int]:
        found: set[int] = set()
        for p in glob.glob(os.path.join(weights_dir, f"{prefix}-*")):
            base = os.path.basename(p)
            parts = base.split("-", 1)
            if len(parts) != 2:
                continue
            try:
                found.add(int(parts[1]))
            except ValueError:
                continue
        return found

    if not os.path.isdir(weights_dir):
        return None
    common = iters("backbone") & iters("generator")
    return max(common) if common else None

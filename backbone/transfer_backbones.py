"""
ImageNet-pretrained CNN / ViT bodies adapted to the MTLFace FR tensor contract:
512-D embedding and (when return_age=True) x_id, x_age of shape (B, 512, S, S) with
S = input_size // 16, then AttentionModule + output_layer matching AIResNet.

return_shortcuts is not supported (FAS / AgingModule require IResNet stages).

ViT-B/32: use torchvision.models.vit_b_32 when available (torchvision >= 0.13).
Otherwise install timm and use vit_base_patch32_224 pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsm import AttentionModule


def _output_head(input_size: int) -> nn.Sequential:
    s = input_size // 16
    return nn.Sequential(
        nn.BatchNorm2d(512),
        nn.Dropout(),
        nn.Flatten(),
        nn.Linear(512 * s * s, 512),
        nn.LayerNorm(512),
    )


def _init_head_modules(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class MobileNetFsmBackbone(nn.Module):
    """MobileNetV2 features (ImageNet) -> 512 x S x S -> FSM -> 512-D embedding."""

    def __init__(self, input_size: int, pretrained: bool = True):
        super().__init__()
        self.input_size = input_size
        s = input_size // 16
        from torchvision.models import mobilenet_v2

        try:
            from torchvision.models import MobileNet_V2_Weights

            w = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            full = mobilenet_v2(weights=w)
        except Exception:
            full = mobilenet_v2(pretrained=pretrained)
        self.body = full.features
        self.proj = nn.Conv2d(1280, 512, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((s, s))
        self.fsm = AttentionModule()
        self.output_layer = _output_head(input_size)
        _init_head_modules(self.proj)
        _init_head_modules(self.output_layer)

    def forward(self, x, return_age=False, return_shortcuts=False):
        if return_shortcuts:
            raise NotImplementedError(
                "return_shortcuts is only supported for IResNet backbones (required for --train_fas)."
            )
        x = self.body(x)
        x = self.proj(x)
        x = self.pool(x)
        x_id, x_age = self.fsm(x)
        embedding = self.output_layer(x_id)
        if return_age:
            return embedding, x_id, x_age
        return embedding


def _build_torchvision_vit_b32(pretrained: bool):
    from torchvision.models import vit_b_32

    try:
        from torchvision.models import ViT_B_32_Weights

        w = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        vit = vit_b_32(weights=w)
    except Exception:
        vit = vit_b_32(pretrained=pretrained)
    return vit


class ViTFsmBackbone(nn.Module):
    """
    ViT-B/32 patch tokens -> 512 channels -> pool to S x S -> FSM -> embedding.
    Internally resizes input to the ViT's expected resolution (224) before conv_proj.
    """

    def __init__(self, input_size: int, pretrained: bool = True):
        super().__init__()
        self.input_size = input_size
        s = input_size // 16
        self._impl = "timm"
        self.vit = None
        self.conv_proj = None
        self.class_token = None
        self.encoder = None
        self.hidden_dim = 768
        self._vit_h = 224
        self._vit_w = 224

        try:
            vit = _build_torchvision_vit_b32(pretrained)
            self._impl = "torchvision"
            self.conv_proj = vit.conv_proj
            self.class_token = vit.class_token
            self.encoder = vit.encoder
            self.hidden_dim = vit.hidden_dim
            size = vit.image_size
            if isinstance(size, int):
                self._vit_h = self._vit_w = size
            else:
                self._vit_h, self._vit_w = int(size[0]), int(size[1])
        except Exception:
            try:
                import timm
            except ImportError as e:
                raise ImportError(
                    "ViT-B/32 needs torchvision >= 0.13 (vit_b_32) or the timm package. "
                    "Install timm: pip install timm"
                ) from e
            self.vit = timm.create_model(
                "vit_base_patch32_224",
                pretrained=pretrained,
                num_classes=0,
            )
            self.hidden_dim = self.vit.embed_dim
            self._vit_h = self._vit_w = 224

        self.proj = nn.Conv2d(self.hidden_dim, 512, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((s, s))
        self.fsm = AttentionModule()
        self.output_layer = _output_head(input_size)
        _init_head_modules(self.proj)
        _init_head_modules(self.output_layer)

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=(self._vit_h, self._vit_w),
            mode="bicubic",
            align_corners=False,
        )
        if self._impl == "torchvision":
            n, c, h, w = x.shape
            x = self.conv_proj(x)
            B, D, hp, wp = x.shape
            x = x.reshape(B, D, hp * wp).permute(0, 2, 1)
            cls = self.class_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = self.encoder(x)
            patches = x[:, 1:, :]
            x = patches.permute(0, 2, 1).reshape(B, D, hp, wp)
            return x
        # timm
        feat = self.vit.forward_features(x)
        if feat.dim() == 2:
            raise RuntimeError("Unexpected ViT forward_features rank-2 output")
        B, ntok, D = feat.shape
        patches = feat[:, 1:, :] if ntok > 1 else feat
        n = patches.size(1)
        g = int(n**0.5)
        if g * g != n:
            raise RuntimeError(f"Cannot reshape {n} patch tokens to a square grid")
        x = patches.transpose(1, 2).reshape(B, D, g, g)
        return x

    def forward(self, x, return_age=False, return_shortcuts=False):
        if return_shortcuts:
            raise NotImplementedError(
                "return_shortcuts is only supported for IResNet backbones (required for --train_fas)."
            )
        x = self._tokens_to_map(x)
        x = self.proj(x)
        x = self.pool(x)
        x_id, x_age = self.fsm(x)
        embedding = self.output_layer(x_id)
        if return_age:
            return embedding, x_id, x_age
        return embedding

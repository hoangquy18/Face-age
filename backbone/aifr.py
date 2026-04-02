import torch.nn.functional as F
import torch
import tqdm
from torch import nn

###########
from .fsm import AttentionModule
from .irse import IResNet


class AIResNet(IResNet):
    def __init__(self, input_size, num_layers, mode="ir", **kwargs):
        super(AIResNet, self).__init__(input_size, num_layers, mode)
        self.fsm = AttentionModule()
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(512 * (input_size // 16) ** 2, 512),
            # BatchNorm1d fails for batch_size==1; LayerNorm normalizes per-sample over 512-dim.
            nn.LayerNorm(512),
        )
        self._initialize_weights()

    def forward(self, x, return_age=False, return_shortcuts=False):
        x_1 = self.input_layer(x)
        x_2 = self.block1(x_1)
        x_3 = self.block2(x_2)
        x_4 = self.block3(x_3)
        x_5 = self.block4(x_4)
        x_id, x_age = self.fsm(x_5)
        embedding = self.output_layer(x_id)
        if return_shortcuts:
            return x_1, x_2, x_3, x_4, x_5, x_id, x_age
        if return_age:
            return embedding, x_id, x_age
        return embedding


class AgeEstimationModule(nn.Module):
    def __init__(self, input_size, age_group, dist=False):
        super(AgeEstimationModule, self).__init__()
        out_neurons = 101
        self.age_output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512 * (input_size // 16) ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True) if dist else nn.ReLU(inplace=True),
            nn.Linear(512, out_neurons),
        )
        self.group_output_layer = nn.Linear(out_neurons, age_group)

    def forward(self, x_age):
        x_age = self.age_output_layer(x_age)
        x_group = self.group_output_layer(x_age)
        return x_age, x_group


from functools import partial

from .transfer_backbones import MobileNetFsmBackbone, ViTFsmBackbone

# IResNet only: FAS / AgingModule needs return_shortcuts from the backbone.
FAS_COMPATIBLE_BACKBONE_NAMES = frozenset(
    {"ir34", "ir50", "ir64", "ir101", "irse101"}
)

backbone_dict = {
    "ir34": partial(AIResNet, num_layers=[3, 4, 6, 3], mode="ir"),
    "ir50": partial(AIResNet, num_layers=[3, 4, 14, 3], mode="ir"),
    "ir64": partial(AIResNet, num_layers=[3, 4, 10, 3], mode="ir"),
    "ir101": partial(AIResNet, num_layers=[3, 13, 30, 3], mode="ir"),
    "irse101": partial(AIResNet, num_layers=[3, 13, 30, 3], mode="ir_se"),
    "mobilenet_v2": partial(MobileNetFsmBackbone, pretrained=True),
    "vit_b_32": partial(ViTFsmBackbone, pretrained=True),
}

BACKBONE_CHOICES = tuple(backbone_dict.keys())

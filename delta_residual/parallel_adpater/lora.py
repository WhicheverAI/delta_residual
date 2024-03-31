import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .general_parallel_adapter import GeneralParallelAdapter


class LowRankLinear(nn.Module):
    """This is only the delta part.
    It shares the same i/o protocol of nn.Linear.
    """

    #  ------------------------------------------------------------------------------------------
    #  Copyright (c) Microsoft Corporation. All rights reserved.
    #  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
    #  ------------------------------------------------------------------------------------------
    #  copy from loralib and do some refactor
    def __init__(
        self,
        in_features,
        out_features,
        weight,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        if r > 0:
            self.lora_A = nn.Parameter(weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LowRankAdapterLayer(GeneralParallelAdapter):
    def __init__(
        self,
        reference_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        # super().__init__(reference_layer, None)
        super().__init__(
            reference_layer,
            additional_layer=LowRankLinear(
                reference_layer.in_features,
                reference_layer.out_features,
                reference_layer.weight,
                r,
                lora_alpha,
                lora_dropout,
            ),
        )

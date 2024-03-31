import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from .general_delta import AbstractDeltaLayer, AbstractDeltaModule
from .utils import auto_tuple_output_for_forward_hook


class GeneralParallelAdapter(AbstractDeltaLayer):
    def __init__(
        self,
        reference_layer: nn.Module,
        additional_layer: nn.Module = None,
    ):
        super().__init__(reference_layer)
        self.additional_layer = additional_layer

    @auto_tuple_output_for_forward_hook
    def _forward_hook(self, module: nn.Module, inputs: tuple, outputs: tuple) -> tuple:
        parallel_output = self.additional_layer(*inputs)
        parallel_output = list(
            parallel_output
        )  # 如果本来是单个输出，也可以转成单个元素的list。如果是tuple那么就是tuple变成list。
        new_outputs = list()
        # logger.warning(new_outputs)
        for i, o in enumerate(outputs):
            if isinstance(o, torch.Tensor):
                new_output = o + parallel_output[i]
                new_outputs.append(new_output)
            else:
                logger.warning(
                    f"Output {i} is not a tensor, I will not add the parallel output to it."
                )
        return tuple(new_outputs)


import math


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

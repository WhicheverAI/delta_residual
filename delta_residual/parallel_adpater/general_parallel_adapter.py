import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from ..general_delta import AbstractDeltaLayer, AbstractDeltaModule
from ..utils import auto_tuple_output_for_forward_hook


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

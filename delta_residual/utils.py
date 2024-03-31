import inspect
from typing import Any, Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from delta_residual.general_delta import AbstractDeltaModule


def get_sorted_function_inputs_from_args(fun, *args, **kwargs) -> dict[str, Any]:
    args = list(args)
    # 将输入函数的参数正则化，变成按照函数调用顺序的、写出参数名称的字典调用
    sig = inspect.signature(fun)
    result = dict()  # python字典是有序的。
    # kwargs的顺序不一定和函数定义的顺序一致，所以要先把kwargs里的参数按照函数定义的顺序放到result里。
    length_args = len(args)
    for i, (name, param) in enumerate(sig.parameters.items()):
        kind = param.kind
        # TODO 暂时不支持 forward本身也有*args, **kwargs的情况， 如果是函数的中间有这两玩意更加恐怖。
        if (
            kind == inspect.Parameter.VAR_POSITIONAL
            or kind == inspect.Parameter.VAR_KEYWORD
        ):
            raise TypeError(f'"{kind}" is not supported in forward.')

        # 这个是因为 onnx.export 必须要以 positional arguments的形式传进去。
        if kind == inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(f'"{kind}" is not supported in forward.')
        if i < length_args:
            item = args.pop(0)
            result[name] = item
        else:
            # result[name] = ... # 先占一个位置。为了保证和Python行为一样
            if name in kwargs:
                result[name] = kwargs.pop(name)
            elif param.default is not inspect._empty:
                result[name] = param.default
            else:
                raise TypeError(f'missing a required argument "{name}"')
    # args 和 kwargs可能有剩余的参数，这就是错误
    if len(args) > 0:
        raise TypeError("forward() got too many positional arguments")
    if len(kwargs) > 0:
        raise TypeError("forward() got too many keyword arguments")
    return result


def auto_tuple_output_for_forward_hook(
    unwrapped_hook: Callable[[nn.Module, nn.Module, Tuple, Tuple], Tuple]
) -> Callable[[nn.Module, nn.Module, Tuple, Tuple | torch.Tensor], Tuple]:
    def wrapped_hook(
        self, module: nn.Module, inputs: tuple, outputs: tuple | torch.Tensor
    ) -> tuple:
        # logger.warning(self)
        not_tuple = False
        if not isinstance(outputs, tuple):
            not_tuple = True
            outputs = (outputs,)

        new_outputs = unwrapped_hook(self, module, inputs, outputs)

        if not_tuple:
            return new_outputs[0]
        else:
            return new_outputs

    return wrapped_hook

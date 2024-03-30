import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from utils import *
from .matching_strategy import find_modules

# 这个只修改Attention。 不考虑LayerNorm的话，prompt只对attention产生了影响
# class GeneralSoftPromptAttentionLayer(nn.Module):

# 这个对每一层做修改。
# class GeneralSoftPromptForEncoderLayer(nn.Module):


# 实际上修改哪一层可以指定。
# class DeltaModel(nn.Module):
#     """Some Information about LayerReplacedModel"""
#     def __init__(self, original_model:nn.Module):
#         super().__init__()
#         # self.original_model = (original_model, )
#         # self.forward = self.original_model[0].forward # 没有改变original_layer的行为
#         # self.hooked:bool = False
#     # def hook_in(self):
#     #     if self.injected:
#     #         return
#     #     self.injected = True


#     # def hook_out(self):
#     #     if not self.injected:
#     #         return
#     #     self.injected = False

#     def forward(self, x):
#         # assert self.injected, "If not injected, you can't forward the model."
#         assert False
#         return x
class AbstractDeltaModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def refer_to(self, model: nn.Module = None):
        """Simply let the DeltaModel `forward()` equals the reference model's `forward()`.
            Note: If the model is not hooked into by `self`, then the self.`__call__()` may not work as expected. Put it differently, DeltaModel's own delta computations will not be called.
            Note: This shall not change the behavior of `model`.
        Args:
            model (nn.Module, optional): reference Pytorch model. Defaults to None. If None, the DeltaModel is set to be not callable.
        """
        if model is None:
            self.forward = None
        else:
            self.forward = model.forward
        self.reference_model_tup = (model,)

    def hook_into(self, model: nn.Module):
        """Let the DeltaModel injects its computation into the reference model.
        After that, the reference model's `__call__()` is modified, with not only reference model's own `forward()`, but also the delta computations.
        The hooking method is designed to be invertible. To cancel the modification, see also `remove_hook_from()`.
        Note: This method would change the behavior of `model`.
        Args:
            model (nn.Module): reference Pytorch model.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def remove_hook_from(self, model: nn.Module):
        """Remove the hooking effect of `self` on `model`.
        Note: This method would change the behavior of `model`.
        Args:
            model (nn.Module): reference Pytorch model.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def merge_into(self, model: nn.Module):
        """Re-parameterize the reference model with the delta model.
        Note: This method would change the behavior of `model`.
        Args:
            model (nn.Module): reference Pytorch model.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError


class AbstractDeltaLayer(AbstractDeltaModule):
    """
    1. DeltaLayer is a more rigorous version of DeltaModel.
    DeltaLayer can compute `__call__()` with reference parameters and delta parameters by simply `refer_to` a reference model,
    without modifying the behavior of the reference model.
    Hooking, which would change the behavior of the reference model, is supported but not necessary.

    2. This class provides resource management of hooks and handles for the subclasses.
    """

    def refer_to(self, model: nn.Module = None):
        """Simply let the DeltaModel `forward()` equals the reference model's `forward()`.
            Note: This shall not change the behavior of `model`.
        Args:
            model (nn.Module, optional): reference Pytorch model. Defaults to None. If None, the DeltaModel is set to be not callable.
        """
        super().refer_to(model)


class GeneralDeltaModel(AbstractDeltaModule):
    """我不是个抽象类, 我是任何DeltaModel都直接能用。
    我是个Layer替代器。
    """

    def __init__(
        self,
        reference_model: nn.Module,
        modified_modules: list[str],
        adapter_name="delta",
        layer_delta_class=nn.Module,
        layer_config: dict = None,
    ) -> None:
        super().__init__()
        self.layer_delta_class = layer_delta_class
        self.layer_config = layer_config or dict()
        self.adapter_name = adapter_name
        self.refer_to(reference_model)
        self.initiate(reference_model, modified_modules)
        self.hook_into(reference_model)

    def initiate(self, reference_model: nn.Module, modified_modules: list[str]):
        self.delta_layers: nn.ModuleDict[str, AbstractDeltaModule] = nn.ModuleDict()
        for name, module in find_modules(reference_model, modified_modules):
            # self.delta_layers.add_module(f"{self.adapter_name}.{name}",
            self.delta_layers.add_module(
                name.replace(".", "=="),
                self.layer_delta_class(module, **self.layer_config),
            )

    def hook_into(self, model: nn.Module):
        for name, layer in self.delta_layers.items():
            original = model.get_submodule(name.replace("==", "."))
            layer.hook_into(original)

    def remove_hook_from(self, model: nn.Module):
        for name, layer in self.delta_layers.items():
            original = model.get_submodule(name.replace("==", "."))
            layer.remove_hook_from(original)

    def merge_into(self, model: nn.Module):
        for name, layer in self.delta_layers.items():
            original = model.get_submodule(name.replace("==", "."))
            layer.merge_into(original)

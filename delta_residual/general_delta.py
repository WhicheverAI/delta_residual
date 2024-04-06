from typing import Any, Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from .matching_strategy import find_modules
from .utils import (
    ModuleDeviceAddOn,
    SeeTrainableParametersAddOn,
    get_module_device,
    get_tuple_device,
    set_requires_grad,
)

# class ModificationReceipt(nn.Module):


@ModuleDeviceAddOn
@SeeTrainableParametersAddOn
class AbstractDeltaModule(nn.Module):
    def __init__(self, reference_model: nn.Module = None) -> None:
        super().__init__()
        self.refer_to(reference_model)

    def refer_to(self, model: nn.Module = None):
        """Simply let the DeltaModel `forward()` equals the reference model's `forward()`.
            Note: If the model is not hooked into by `self`, then the self.`__call__()` may not work as expected. Put it differently, DeltaModel's own delta computations will not be called.
            Note: This shall not change the behavior of `model`.
        Args:
            model (nn.Module, optional): reference Pytorch model. Defaults to None. If None, the DeltaModel is set to be not callable.
        """
        # 接口保留
        # 注意，original_layer不是GeneralSoftPromptLayer的子模块，参数不应该被保存和加载，而是应该总是从外部传入。
        # https://discuss.pytorch.org/t/unregister-prevent-from-registering-a-nn-module/134768
        # self.reference_layer = (reference_layer, )
        # self.forward = self.reference_layer[0].forward
        if model is None:
            self.forward = None
        else:
            self.forward = model.forward
        self.reference_model_tup = (model,)

    @property
    def reference_model(self):
        return self.reference_model_tup[0]

    @reference_model.setter
    def reference_model(self, new_reference_model: nn.Module):
        self.refer_to(new_reference_model)

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

    def prepare_for_training(self, model: nn.Module):
        """Notice that `hook_into` is just changing the calling behavior of the model,
        it is not freezing the optimizable parameters for training.
        It is this function that is doing preparation for training.
        Args:
            model (nn.Module): _description_
        """
        self.refer_to(model)
        self.hook_into(model)
        set_requires_grad(model, False)
        set_requires_grad(self, True)  # that's it, very simple.


class AbstractDeltaLayer(AbstractDeltaModule):
    """
    One new layer to substitute another, don't change the original one's behavior unless specified.
    1. DeltaLayer is a more rigorous version of DeltaModel.
    DeltaLayer can compute `__call__()` with reference parameters and delta parameters by simply `refer_to` a reference model,
    without modifying the behavior of the reference model.
    Hooking, which would change the behavior of the reference model, is supported but not necessary.

    2. This class provides resource management of hooks and handles for the subclasses.
    """

    def __init__(self, reference_layer: nn.Module = None) -> None:
        super().__init__(
            reference_model=reference_layer
        )  # 这里会调用`refer_to`, 没有改变original_layer的行为。如果需要里面的性质才能推导出来layer要怎么初始化的话，可以用self.reference_model
        # 对自身的forward进行hook
        self.forward_pre_hook_handle = self.register_forward_pre_hook(
            hook=self._forward_pre_hook
        )  # 闭包，知晓这个类的信息的。
        self.forward_hook_handle = self.register_forward_hook(
            hook=self._forward_hook
        )  # 闭包，知晓这个类的信息的。
        # 去hook其他模型
        self.others_forward_pre_hook_handles = dict()
        self.others_forward_hook_handles = dict()

    def __del__(self):
        for layer in self.others_forward_pre_hook_handles.keys():
            self.remove_hook_from(layer)

    def refer_to(self, model: nn.Module = None):
        """Simply let the DeltaModel `forward()` equals the reference model's `forward()`.
            Note: This shall not change the behavior of `model`.
        Args:
            model (nn.Module, optional): reference Pytorch model. Defaults to None. If None, the DeltaModel is set to be not callable.
        """
        super().refer_to(
            model
        )  # Compared to AbstractDeltaModule, we just change the documentation here.

    def hook_into(self, layer: nn.Module):
        logger.debug(
            f"Try to hook delta:{self.device} into model:{get_module_device(layer)}. "
        )

        if self.others_forward_pre_hook_handles.get(layer) is not None:
            logger.warning(
                f"Layer {layer} has already been hooked. I will remove the old first and then replace it with the new."
            )
            self.remove_hook_from(layer)
        self.others_forward_pre_hook_handles[layer] = layer.register_forward_pre_hook(
            hook=self._forward_pre_hook
        )
        self.others_forward_hook_handles[layer] = layer.register_forward_hook(
            hook=self._forward_hook
        )

    def remove_hook_from(self, layer: nn.Module):
        logger.debug(
            f"Try to remove delta:{self.device} from model:{get_module_device(layer)}. "
        )
        if not self.others_forward_pre_hook_handles.get(layer):
            logger.warning(
                f"Layer {layer} has not been hooked. I will do nothing and return."
            )
            return
        self.others_forward_pre_hook_handles[layer].remove()
        self.others_forward_hook_handles[layer].remove()
        del self.others_forward_pre_hook_handles[layer]
        del self.others_forward_hook_handles[layer]

    def _forward_pre_hook(self, module: nn.Module, inputs: tuple) -> tuple:
        # logger.warning("Shall be implemented by subclasses. ")
        logger.debug(
            f"delta:{self.device} is called in addition to model:{get_module_device(module)} with input:{get_tuple_device(inputs)}."
        )
        return inputs

    def _forward_hook(
        self, module: nn.Module, inputs: tuple, outputs: tuple | torch.Tensor
    ) -> tuple:
        logger.debug(
            f"delta:{self.device} is called in addition to model:{get_module_device(module)} with input:{get_tuple_device(inputs)}."
        )
        # logger.warning("Shall be implemented by subclasses. ")
        return outputs


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
        super().__init__(reference_model)
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
        # set_requires_grad(model, True)
        for name, layer in self.delta_layers.items():
            original = model.get_submodule(name.replace("==", "."))
            layer.remove_hook_from(original)

    def merge_into(self, model: nn.Module):
        for name, layer in self.delta_layers.items():
            original = model.get_submodule(name.replace("==", "."))
            layer.merge_into(original)


@ModuleDeviceAddOn
class ModelWithDelta(nn.Module):
    def __init__(self, model: nn.Module, delta: AbstractDeltaModule):
        super().__init__()
        self.model = model
        # self.model.delta = delta
        self.delta = delta
        self.delta.refer_to(self.model)
        self.delta.hook_into(self.model)
        # self.delta.remove_hook_from(model)

    def _replicate_for_data_parallel(self):
        logger.debug(
            f"Replicating {self.__class__.__name__}:{self.device} for data parallel. "
        )
        # 首先，
        # self.model.hook == self.delta
        # self.delta.handle == remover(self.model.hook)

        # replica = super()._replicate_for_data_parallel()
        # 进行了操作之后，self的关系没有变化, 但是
        # replica.model.hook == self.delta # 问题是这个无法被删除
        # replica.delta.handle == remover(self.model.hook)
        # 我们期望的操作是应该是
        # replica.model.hook == replica.delta
        # replica.delta.handle == remover(replica.model.hook)

        # 我们直接重启
        self.delta.remove_hook_from(self.model)

        replica = super()._replicate_for_data_parallel()
        # ?这里self和replica是浅拷贝，里面的东西还是一样的，所以重启失败了
        # 对象是新的同类对象
        # _buffers _modules 都是浅拷贝
        # __dict__是浅拷贝
        # _parameters 是 新的 OrderedDict，所以没有新的参数产生！都是旧的玩意

        replica.model = self.model._replicate_for_data_parallel()
        replica.delta = self.delta._replicate_for_data_parallel()

        assert id(self.delta) != id(replica.delta)
        assert id(self.model) != id(replica.model)

        self.delta.hook_into(self.model)  # 问题就在这里，仍然重复了。 replica的hooks和model的hooks是一样的。
        replica.delta.hook_into(replica.model)  # replica并没有自己独立的hook字典。

        # self.delta.remove_hook_from(replica.model)
        # self.delta.refer_to(self.model)
        # self.delta.hook_into(self.model)

        # replica.delta.remove_hook_from(self.model)
        # replica.delta.refer_to(replica.model)
        # replica.delta.hook_into(replica.model)
        return replica

    def forward(self, *args, **kwargs):
        logger.debug(
            f"Forwarding {self.__class__.__name__}:{self.device} with inputs:{get_tuple_device(*args)}. "
        )
        return self.model(*args, **kwargs)

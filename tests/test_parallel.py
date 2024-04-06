#%%
from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from delta_residual.utils import ModuleDeviceAddOn


@ModuleDeviceAddOn
class SimpleModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        logger.info(
            f"I am now at {self.device}. SimpleModule is called with input x={x} . My weight is {self.linear.weight}, bias is {self.linear.bias}"
        )
        return self.linear(x)


model = SimpleModule()
# x = torch.Tensor([1, 2, 3, 4]).unsqueeze(1)
# x = torch.zeros(4).unsqueeze(1)
x = torch.ones(4).unsqueeze(1)
model(x)
from delta_residual.general_delta import GeneralDeltaModel

#%%
from delta_residual.parallel_adpater import LowRankAdapterLayer

delta = GeneralDeltaModel(
    model,
    ["linear"],
    layer_delta_class=partial(LowRankAdapterLayer, r=1, lora_alpha=1, lora_dropout=0.0),
)
lora_linear = list(delta.delta_layers.values())[0].additional_layer
nn.init.ones_(lora_linear.lora_B)
nn.init.ones_(lora_linear.lora_A)
# lora_linear(x)
# delta.hook_into(model) # 这hook了两次
delta(x)

#%%
from delta_residual.general_delta import ModelWithDelta

model_with_delta = ModelWithDelta(model, delta)
model_with_delta(x)
#%%
# 首先是to(device)或者 to(type)逻辑要对
# 也就是他们一起改参数位置，他们互相绑定了，这个要对
# model_with_delta.to('cuda:1')(x.to('cuda:1'))
model_with_delta.to("cuda:2")(x.to("cuda:2"))
#%%
# 这是特殊的复制，参数没有to，复制的是计算逻辑。
# 核心问题在于， hook发生变化
replica = model_with_delta._replicate_for_data_parallel()
replica(x.to("cuda:2"))  # 暂时看不出问题，因为replica.model就算引用错误，也能计算出正确结果。
# replica.model.device
#%%
id(list(model_with_delta.named_parameters())[0][0]), id(
    list(replica.named_parameters())[0][0]
)
# next(replica.parameters()).device, next(replica.parameters()).device
#%%
model_with_delta.to("cpu")(x)
#%%
list(model_with_delta.delta.delta_layers.values())[0].others_forward_hook_handles
#%%
list(replica.model._forward_hooks.values())
# %%
p_m_d = nn.DataParallel(model_with_delta.cuda())
# p_m_d.model(x.cuda()) # 去掉了
# p_m_d(x.cuda())
# %%
# model.to('cuda:1') # 这个是inplace修改
# model.linear.weight.device # 变了
# %%

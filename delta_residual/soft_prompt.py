#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from .general_delta import AbstractDeltaModule


class GeneralSoftPromptLayer(AbstractDeltaModule):
    """One Layer to substitute another, don't change the original one's behavior unless specified."""

    def __init__(
        self,
        reference_layer: nn.Module,
        soft_token_num=100,
        hidden_dim=768,
        init_range=0.5,
        original_embedding=None,  # 使用原本的embedding去生成token，这样可能满足约束。但是我认为没必要，本来就可逆的。
        # prepended_inputs:list=["hidden_states"], # 用名字指定
        prepended_inputs: list = [0],  # 用位置指定，应该也是合理的。
        removed_outputs: list = [0],  # 如果空就不remove，对于shallow vpt就是这样，只操作layer1。
        # return的1个也被我们当做是tuple。
        #  假设直接选择出来就是tensor
        # tensor_selector = lambda x:x, # 有时候可以 x[0]
        dim_of_tokens=-2,
        dim_of_hidden=-1,
        dim_of_batches=0,
    ):
        super().__init__()
        # 接口保留
        # 注意，original_layer不是GeneralSoftPromptLayer的子模块，参数不应该被保存和加载，而是应该总是从外部传入。
        # https://discuss.pytorch.org/t/unregister-prevent-from-registering-a-nn-module/134768
        # self.reference_layer = (reference_layer, )
        # self.forward = self.reference_layer[0].forward
        self.refer_to(reference_layer)  # 没有改变original_layer的行为
        # super().__init__() # 这样的话，original_layer就没有被记录，非常好
        # peft方法的参数
        self.soft_token_num = soft_token_num
        self.hidden_dim = hidden_dim
        self.init_range = init_range
        self.original_embedding = original_embedding
        # 初始化prompt
        # self.compiled = False
        self.soft_prompts: torch.Tensor = None
        if self.hidden_dim is not None:
            self.prompts = self.instantiate(hidden_dim)
        # 操作位置的指定
        self.prepended_inputs = prepended_inputs
        self.removed_outputs = removed_outputs
        # self.tensor_selector = tensor_selector
        self.dim_of_tokens = dim_of_tokens
        self.dim_of_hidden = dim_of_hidden
        self.dim_of_batches = dim_of_batches

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
        for layer in self.others_forward_pre_hook_handles:
            self.remove_hook_from(layer)

    def _forward_pre_hook(self, module: nn.Module, input: tuple) -> tuple:
        # 返回新的input
        new_input = list(input)
        for i in self.prepended_inputs:
            # selected_tensor = self.tensor_selector(input[i]) # 得到一个指针
            # selected_tensor = torch.cat([selected_tensor, self.prompts], dim=self.dim_of_tokens) # 并没有修改原来的tensor
            # self.tensor_selector(input[i]) = selected_tensor
            selected_tensor: torch.Tensor = input[i]
            b = selected_tensor.shape[self.dim_of_batches]
            # TODO 操作不太对, 元组不能直接assign
            new_input[i] = torch.cat(
                [
                    selected_tensor,
                    #   torch.tile(self.soft_prompts, dims=selected_tensor.size()[:self.dim_of_tokens])
                    self.soft_prompts.repeat(b, 1, 1),
                ],
                dim=self.dim_of_tokens,
            )
        return tuple(new_input)

    def _forward_hook(self, module: nn.Module, input: tuple, output: tuple) -> tuple:
        # 返回新的output
        not_tuple = False
        if not isinstance(output, tuple):
            not_tuple = True
            output = (output,)
        new_output = list(output)
        for i in self.removed_outputs:
            # 我想要对 output[i]这个tensor进行操作,
            selected_tensor: torch.Tensor = output[i]
            # 我要在 self.dim_of_tokens 这一个维度上，去掉最后添加的 self.soft_token_num 个元素
            new_output[i] = selected_tensor.narrow(
                self.dim_of_tokens,
                0,
                selected_tensor.size(self.dim_of_tokens) - self.soft_token_num,
            )
        if not_tuple:
            return new_output[0]
        return tuple(new_output)

    def instantiate(self, hidden_dim) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        soft_prompts = torch.FloatTensor(1, self.soft_token_num, hidden_dim)
        if self.original_embedding is not None:
            soft_prompts.data = torch.clone(
                self.original_embedding(
                    torch.tensor([i for i in range(self.soft_token_num)])
                )
            )
        else:
            soft_prompts = soft_prompts.uniform_(-self.init_range, self.init_range)

        self.soft_prompts: torch.Tensor = nn.Parameter(soft_prompts, requires_grad=True)
        # .to(self.device)

    def hook_into(self, layer: nn.Module):
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
        self.others_forward_pre_hook_handles[layer].remove()
        self.others_forward_hook_handles[layer].remove()
        del self.others_forward_pre_hook_handles[layer]
        del self.others_forward_hook_handles[layer]

    def merge_into(self, layer: nn.Module):
        raise ArithmeticError("General Soft Prompt Tuning cannot be re-parameterized.")
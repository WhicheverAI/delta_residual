"""Tests for Tuning Method `GeneralSoftPromptLayer` 's Implementation ."""
#%%
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

#%%
from transformers import AutoModel

from delta_residual.general_delta import GeneralDeltaModel
from delta_residual.soft_prompt import GeneralSoftPromptLayer
from delta_residual.utils import get_sorted_function_inputs_from_args


class TestGeneralSoftPromptLayer:
    def setup_class(self):
        name = "facebook/dinov2-base"
        self.model = AutoModel.from_pretrained(name)
        self.atten: nn.Module = self.model.encoder.layer[0].attention
        b, c, h, w = 1, 3, 224, 224
        self.data = torch.randn(b, c, h, w)
        self.atol = 1e-5

    def teardown_class(self):
        print("TestGeneralSoftPromptLayer finished.")

    def test_my_understanding_for_attention(self):
        assert self.atten(torch.randn(4, 16 * 16, 768))[0].shape == (4, 16 * 16, 768)
        # from bigmodelvis import Visualization
        # Visualization(model).structure_graph()
        # Visualization(delta).structure_graph()
        shape = None

        def hook(m, x):
            nonlocal shape
            logger.debug(type(x))  # 是个tuple
            shape = get_sorted_function_inputs_from_args(m.forward, x)["hidden_states"][
                0
            ].shape

        self.atten.register_forward_pre_hook(hook)
        self.model(self.data)
        assert shape == (1, 16 * 16 + 1, 768)

    def test_difference(self):
        before_res = self.model(self.data)[0]
        logger.info(f"Before: {before_res.norm()}, {before_res.shape}")
        self.delta = GeneralDeltaModel(
            self.model,
            modified_modules=["attention.attention"],
            layer_delta_class=GeneralSoftPromptLayer,
        )
        self.delta.hook_into(self.model)
        after_res = self.delta(self.data)[0]
        logger.info(f"After: {after_res.norm()}, {after_res.shape}")
        assert not torch.isclose(before_res, after_res, atol=self.atol).all()
        assert after_res.norm() != before_res.norm()
        assert after_res.shape == before_res.shape
        self.delta.remove_hook_from(self.model)
        new_after_res = self.model(self.data)[0]
        assert torch.isclose(before_res, new_after_res, atol=self.atol).all()
        self.delta.hook_into(self.model)
        new_after_res = self.delta(self.data)[0]
        assert torch.isclose(after_res, new_after_res, atol=self.atol).all()

"""Tests for Tuning Method `LowRankAdapterLayer` 's Implementation ."""
#%%
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from transformers import AutoModel

#%%
from delta_residual.general_delta import GeneralDeltaModel
from delta_residual.parallel_adpater import LowRankAdapterLayer


class TestGeneralSoftPromptLayer:
    def setup_class(self):
        name = "facebook/dinov2-base"
        self.model = AutoModel.from_pretrained(name)
        self.atten: nn.Module = self.model.encoder.layer[0].attention
        b, c, h, w = 1, 3, 224, 224
        self.data = torch.randn(b, c, h, w)
        self.atol = 1e-5

    def test_see(self):
        from bigmodelvis import Visualization

        Visualization(self.model).structure_graph()

    def make_delta(self):
        self.delta = GeneralDeltaModel(
            self.model,
            modified_modules=["attention.attention.query", "attention.attention.value"],
            layer_delta_class=LowRankAdapterLayer,
        )

    def test_difference(self):
        before_res = self.model(self.data)[0]
        logger.info(f"Before: {before_res.norm()}, {before_res.shape}")
        self.make_delta()
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

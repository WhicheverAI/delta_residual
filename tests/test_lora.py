"""Tests for Tuning Method `LowRankAdapterLayer` 's Implementation ."""
#%%
import copy

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
from delta_residual.utils import get_nb_trainable_parameters, set_requires_grad


class TestLoraLayer:
    def setup_class(self):
        name = "facebook/dinov2-base"
        self.model = AutoModel.from_pretrained(name)
        self.atten: nn.Module = self.model.encoder.layer[0].attention
        b, c, h, w = 1, 3, 224, 224
        self.data = torch.randn(b, c, h, w)
        self.atol = 1e-5
        self.delta = None
        self.peft_delta = None
        self.opendelta_delta = None

    def test_see(self):
        from bigmodelvis import Visualization

        Visualization(self.model).structure_graph()

    def make_delta(self):
        if self.delta is None:
            self.delta = GeneralDeltaModel(
                self.model,
                modified_modules=[
                    "attention.attention.query",
                    "attention.attention.value",
                ],
                layer_delta_class=LowRankAdapterLayer,
                layer_config=dict(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.0,
                ),
            )

    def make_difference_test_delta(self):
        from opendelta import LoraModel
        from peft import LoraConfig, get_peft_model

        if self.peft_delta is None:
            self.peft_delta = get_peft_model(
                copy.deepcopy(self.model),
                LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    target_modules=[
                        "attention.attention.query",
                        "attention.attention.value",
                    ],
                ),
            )
        if self.opendelta_delta is None:
            self.opendelta_delta = LoraModel(
                backbone_model=copy.deepcopy(self.model),
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                modified_modules=[
                    "attention.attention.query",
                    "attention.attention.value",
                ],
            )

    def test_difference(self):
        # difference test 差分测试，确保我们和其他人
        self.make_difference_test_delta()
        self.make_delta()
        ours = get_nb_trainable_parameters(self.delta)
        pefts = get_nb_trainable_parameters(self.peft_delta)
        open_deltas = get_nb_trainable_parameters(self.opendelta_delta)
        assert ours[0] == open_deltas[0]
        assert ours[0] == pefts[0]

    def test_modification_effect_on_backbone(self):
        before_res = self.model(self.data)[0]
        logger.info(f"Before: {before_res.norm()}, {before_res.shape}")
        self.make_delta()
        after_res = self.delta(self.data)[0]
        logger.info(f"After: {after_res.norm()}, {after_res.shape}")
        # assert not torch.isclose(before_res, after_res, atol=self.atol).all()
        # assert after_res.norm() != before_res.norm()
        assert after_res.shape == before_res.shape
        self.delta.remove_hook_from(self.model)
        new_after_res = self.model(self.data)[0]
        assert torch.isclose(before_res, new_after_res, atol=self.atol).all()
        self.delta.hook_into(self.model)
        new_after_res = self.delta(self.data)[0]
        assert torch.isclose(after_res, new_after_res, atol=self.atol).all()

    def test_freeze(self):
        before = get_nb_trainable_parameters(self.model)
        self.make_delta()
        after = get_nb_trainable_parameters(self.model)
        assert before == after  # 我们库的设计是这样的。
        # 只有在hook into之后，才会把model冻结。
        deltas = get_nb_trainable_parameters(self.delta)
        logger.info(f"Model: {before}, Delta: {deltas}")
        set_requires_grad(self.model, requires_grad=False)
        set_requires_grad(self.delta, requires_grad=True)
        deltas = get_nb_trainable_parameters(self.delta)
        models = get_nb_trainable_parameters(self.model)
        logger.info(f"Model: {models}, Delta: {deltas}")

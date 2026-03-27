# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
from einops import rearrange
from jaxtyping import Float
from functools import partial
from torch import Tensor
from transformers import AutoImageProcessor
from transformers import AutoModel, AutoConfig
from transformers.feature_extraction_utils import BatchFeature
from types import SimpleNamespace
import torch.nn.functional as F
from cxrclip.model.modules.dinov2_utils import (
    LayerScale,
    NestedTensorBlock as AttentionBlock,
    SwiGLUFFNAligned as SwiGLUFFN,
)
import logging
log = logging.getLogger(__name__)

class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)

__version__ = "0.1.0"

TypeClsToken = Float[Tensor, "batch_size embed_dim"]
TypePatchTokensFlat = Float[Tensor, "batch_size (height width) embed_dim"]
TypePatchTokens = Float[Tensor, "batch_size embed_dim height width"]
TypeInputImages = Tensor

class RadDinoLocal(nn.Module):

    def __init__(self, ckpt_path, freeze_backbone=False, interpolate_pos_encoding=False):
        super().__init__()
        config = AutoConfig.from_pretrained(ckpt_path, local_files_only=True)
        config.output_hidden_states = True  # need this to access the intermediate layer outputs

        self.model = AutoModel.from_pretrained(ckpt_path, config=config, local_files_only=True)
        self.processor = AutoImageProcessor.from_pretrained(ckpt_path, use_fast=False, local_files_only=True)
        self.interpolate_pos_encoding = interpolate_pos_encoding
        if freeze_backbone:
            log.info('[RadDinoLocal]: Disabling training parameters of the whole raddino backbone.')
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def encode(self, inputs: BatchFeature) -> tuple[TypeClsToken, TypePatchTokensFlat]:
        if self.interpolate_pos_encoding:
            # for 224 resolution or 256 resolution (non-518)
            outputs = self.model(inputs, output_hidden_states=True, interpolate_pos_encoding=True)
        else:
            # for 518 resolution
            outputs = self.model(inputs, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]
        patch_tokens = outputs.last_hidden_state[:, 1:]

        return cls_token, patch_tokens, outputs.hidden_states

    def reshape_patch_tokens(
        self,
        patch_tokens_flat: TypePatchTokensFlat,
    ) -> TypePatchTokens:
        input_size = self.processor.crop_size["height"]
        patch_size = self.model.config.patch_size
        embeddings_size = input_size // patch_size
        patches_grid = rearrange(
            patch_tokens_flat,
            "batch (height width) embed_dim -> batch embed_dim height width",
            height=embeddings_size,
        )
        return patches_grid

    def extract_features(
        self,
        inputs: TypeInputImages,
    ) -> tuple[TypeClsToken, TypePatchTokens]:
        cls_token, patch_tokens, hidden_states = self.encode(inputs)
        return cls_token, patch_tokens, hidden_states

    def extract_cls_token(self, image_or_images: TypeInputImages) -> TypeClsToken:
        cls_token, _ = self.extract_features(image_or_images)
        return cls_token

    def extract_patch_tokens(self, image_or_images: TypeInputImages) -> TypePatchTokens:
        _, patch_tokens = self.extract_features(image_or_images)
        return patch_tokens

    def forward(self, *args) -> tuple[TypeClsToken, TypePatchTokens]:
        return self.extract_features(*args)

class VisionHead(nn.Module):
    """
    this is used to augment the dino encoder for segmentation.
    borrowed from https://github.dev/facebookresearch/dinov2
    - refer the dinotxt.py and vision_tower.py
    """
    def __init__(
        self,
        input_dim: int,
        # embed_dim: int,
        num_heads: int,
        num_blocks: int,
        blocks_drop_path: float,
        # use_class_token: bool,
        # use_image_patch_tokens: bool,
        # use_linear_projection: bool,
    ):
        super().__init__()
        block_list = [nn.Identity()]
        self.ln_final = nn.Identity()
        if num_blocks > 0:
            block_list = [
                AttentionBlock(
                    input_dim,
                    num_heads,
                    ffn_layer=partial(SwiGLUFFN, align_to=64),
                    init_values=1e-5,
                    drop_path=blocks_drop_path,
                )
                for _ in range(num_blocks)
            ]
            self.ln_final = nn.LayerNorm(input_dim)
        self.block_list = nn.ModuleList(block_list)
        self.num_blocks = num_blocks
        self.linear_projection = nn.Identity()
        # multiplier = 2 if use_class_token and use_image_patch_tokens else 1
        # if multiplier * input_dim != embed_dim or use_linear_projection:
        #     assert embed_dim % multiplier == 0, f"Expects {embed_dim} to be divisible by {multiplier}"
        #     self.linear_projection = nn.Linear(input_dim, embed_dim // multiplier, bias=False)

    def init_weights(self):
        if self.num_blocks > 0:
            for i in range(self.num_blocks):
                block = self.block_list[i]
                named_apply(init_weights_vit_timm, block)
            self.ln_final.reset_parameters()
        if isinstance(self.linear_projection, nn.Linear):
            nn.init.normal_(self.linear_projection.weight, std=self.linear_projection.in_features**-0.5)

    def forward(self, image_tokens: Tensor) -> Tensor:
        hidden_states = []
        for block in self.block_list:
            image_tokens = block(image_tokens)
            hidden_states.append(image_tokens)

        image_tokens = self.ln_final(image_tokens)
        return SimpleNamespace(
            last_hidden_state=self.linear_projection(image_tokens), # identity
            hidden_states=tuple(hidden_states)
        )
    

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, nn.Conv2d):
        module.reset_parameters()

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
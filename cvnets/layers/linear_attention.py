#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import torch
from torch import Tensor
from typing import Optional, Tuple
from torch.nn import functional as F

from .base_layer import BaseLayer
from .conv_layer import ConvLayer
from .dropout import Dropout
from ..misc.profiler import module_profile

from torch import nn
import math

# TODO: Add link to the paper

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(b, -1, h, w)

        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LinearSelfAttention(BaseLayer):
    """
    This layer applies a self-attention with linear complexity, as described in `this paper <>`_
    This layer can be used for self- as well as cross-attention.

    Args:
        opts: command line arguments
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input

    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        # R = 2
        # G = int(math.sqrt(embed_dim//R))
        # G = G if G>0 else 1
        # G = embed_dim//(embed_dim//G)
        G = 16
        
        self.qkv_proj = ConvLayer(
            opts=opts,
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        # mid_dim = _make_divisible((2*embed_dim+1)//2, G)
        # print(mid_dim)
        # self.qkv_proj1 = ConvLayer(
        #     opts=opts,
        #     in_channels=embed_dim,
        #     out_channels=mid_dim,
        #     bias=bias,
        #     kernel_size=1,
        #     use_norm=False,
        #     use_act=False,
        #     groups=G,
        # )
        self.channel_shuffle1 = ChannelShuffle(G)
        # self.qkv_proj2 = ConvLayer(
        #     opts=opts,
        #     in_channels=mid_dim,
        #     out_channels=1 + (2 * embed_dim),
        #     bias=bias,
        #     kernel_size=1,
        #     use_norm=False,
        #     use_act=False,
        #     groups=G,
        # )
        self.attn_dropout = Dropout(p=attn_dropout)
        # self.out_proj = ConvLayer(
        #     opts=opts,
        #     in_channels=embed_dim,
        #     out_channels=embed_dim,
        #     bias=bias,
        #     kernel_size=1,
        #     use_norm=False,
        #     use_act=False,
        # )

        mid_dim = _make_divisible(embed_dim//2, G)
        self.out_proj1 = ConvLayer(
            opts=opts,
            in_channels=embed_dim,
            out_channels=mid_dim,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
            groups=G,
        )
        
        self.out_proj2 = ConvLayer(
            opts=opts,
            in_channels=mid_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
            groups=G,
        )
        self.embed_dim = embed_dim

    def __repr__(self):
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    @staticmethod
    def visualize_context_scores(context_scores):
        # [B, 1, P, N]
        batch_size, channels, num_pixels, num_patches = context_scores.shape

        assert batch_size == 1, "For visualization purposes, use batch size of 1"
        assert (
            channels == 1
        ), "The inner-product between input and latent node (query) is a scalar"

        up_scale_factor = int(num_pixels ** 0.5)
        patch_h = patch_w = int(context_scores.shape[-1] ** 0.5)
        # [1, 1, P, N] --> [1, P, h, w]
        context_scores = context_scores.reshape(1, num_pixels, patch_h, patch_w)
        # Fold context scores [1, P, h, w] using pixel shuffle to obtain [1, 1, H, W]
        context_map = F.pixel_shuffle(context_scores, upscale_factor=up_scale_factor)
        # [1, 1, H, W] --> [H, W]
        context_map = context_map.squeeze()

        # For ease of visualization, we do min-max normalization
        min_val = torch.min(context_map)
        max_val = torch.max(context_map)
        context_map = (context_map - min_val) / (max_val - min_val)

        try:
            import cv2
            from glob import glob
            import os

            # convert from float to byte
            context_map = (context_map * 255).byte().cpu().numpy()
            context_map = cv2.resize(
                context_map, (80, 80), interpolation=cv2.INTER_NEAREST
            )

            colored_context_map = cv2.applyColorMap(context_map, cv2.COLORMAP_JET)
            # Lazy way to dump feature maps in attn_res folder. Make sure that directory is empty and copy
            # context maps before running on different image. Otherwise, attention maps will be overridden.
            res_dir_name = "attn_res"
            if not os.path.isdir(res_dir_name):
                os.makedirs(res_dir_name)
            f_name = "{}/h_{}_w_{}_index_".format(res_dir_name, patch_h, patch_w)

            files_cmap = glob(
                "{}/h_{}_w_{}_index_*.png".format(res_dir_name, patch_h, patch_w)
            )
            idx = len(files_cmap)
            f_name += str(idx)

            cv2.imwrite("{}.png".format(f_name), colored_context_map)
            return colored_context_map
        except ModuleNotFoundError as mnfe:
            print("Please install OpenCV to visualize context maps")
            return context_map

    def _forward_self_attn(self, x: Tensor, *args, **kwargs) -> Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)
        # qkv1= self.qkv_proj1(x)
        # qkv2 = self.channel_shuffle1(qkv2)
        # qkv3 = self.qkv_proj2(qkv2)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        # out = self.out_proj(out)
        out = self.out_proj1(out)
        out = self.channel_shuffle1(out)
        out = self.out_proj2(out)
        return out

    def _forward_cross_attn(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        # x --> [B, C, P, N]
        # x_prev = [B, C, P, M]

        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape

        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.block.conv.weight[: self.embed_dim + 1, ...],
            bias=self.qkv_proj.block.conv.bias[: self.embed_dim + 1, ...],
        )
        # qk1 = F.conv2d(
        #     x_prev,
        #     weight=self.qkv_proj1.block.conv.weight[: self.embed_dim + 1, ...],
        #     bias=self.qkv_proj1.block.conv.bias[: self.embed_dim + 1, ...],
        # )
        # qk2 = self.channel_shuffle1(qk2)
        # qk3 = F.conv2d(
        #     x_prev,
        #     weight=self.qkv_proj2.block.conv.weight,
        #     bias=self.qkv_proj2.block.conv.bias,
        # )
        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = F.conv2d(
            x,
            weight=self.qkv_proj.block.conv.weight[self.embed_dim + 1 :, ...],
            bias=self.qkv_proj.block.conv.bias[self.embed_dim + 1 :, ...],
        )
        # value1 = F.conv2d(
        #     x,
        #     weight=self.qkv_proj.block.conv.weight[self.embed_dim + 1 :, ...],
        #     bias=self.qkv_proj.block.conv.bias[self.embed_dim + 1 :, ...],
        # )
        # value2 = self.channel_shuffle1(value2)
        # value3 = F.conv2d(
        #     x,
        #     weight=self.qkv_proj.block.conv.weight,
        #     bias=self.qkv_proj.block.conv.bias,
        # )
        # apply softmax along M dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # compute context vector
        # [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M]
        context_vector = key * context_scores
        # [B, d, P, M] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        # out = self.out_proj(out)
        out = self.out_proj1(out)
        out = self.channel_shuffle1(out)
        out = self.out_proj2(out)
        return out

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            return self._forward_self_attn(x, *args, **kwargs)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev, *args, **kwargs)

    def profile_module(self, input) -> Tuple[Tensor, float, float]:
        params = macs = 0.0

        qkv, p, m = module_profile(module=self.qkv_proj, x=input)
        params += p
        macs += m

        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        if self.out_proj is not None:
            out_p, p, m = module_profile(module=self.out_proj, x=value)
            params += p
            macs += m

        return input, params, macs

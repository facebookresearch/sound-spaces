#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from ss_baselines.common.utils import Flatten
from habitat_sim.utils.common import d3_40_colors_rgb


class VisualCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, extra_rgb=False):
        super().__init__()
        self._output_size = output_size
        if "rgb" in observation_space.spaces and not extra_rgb:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        if "semantic" in observation_space.spaces:
            self._n_input_semantic = 6
        else:
            self._n_input_semantic = 0

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (2, 2)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )
        elif self._n_input_semantic > 0:
            cnn_dims = np.array(
                observation_space.spaces["semantic"].shape[:2], dtype=np.float32
            )

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            self._input_shape = (self._n_input_rgb + self._n_input_depth + self._n_input_semantic,
                                 int(cnn_dims[0]), int(cnn_dims[1]))
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth + self._n_input_semantic,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                #  nn.ReLU(True),
                Flatten(),
                nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
                nn.ReLU(True),
            )

        self.layer_init()

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth + self._n_input_semantic == 0

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return 1, self._output_size

    @property
    def feature_dims(self):
        return self._output_size

    def forward(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        if self._n_input_semantic > 0:
            semantic_observations = convert_semantics_to_rgb(observations["semantic"]).float()
            semantic_object_observations = observations["semantic_object"].float()
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            semantic_observations = torch.cat([semantic_observations, semantic_object_observations], dim=-1)
            semantic_observations = semantic_observations.permute(0, 3, 1, 2) / 255.0
            cnn_input.append(semantic_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)


def convert_semantics_to_rgb(semantics):
    r"""Converts semantic IDs to RGB images.
    """
    semantics = semantics.long() % 40
    mapping_rgb = torch.from_numpy(d3_40_colors_rgb).to(semantics.device)
    semantics_r = torch.take(mapping_rgb[:, 0], semantics)
    semantics_g = torch.take(mapping_rgb[:, 1], semantics)
    semantics_b = torch.take(mapping_rgb[:, 2], semantics)
    semantics_rgb = torch.stack([semantics_r, semantics_g, semantics_b], -1)

    return semantics_rgb
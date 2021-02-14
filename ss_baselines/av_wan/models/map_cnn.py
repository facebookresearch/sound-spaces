# !/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from ss_baselines.common.utils import Flatten
from ss_baselines.av_nav.models.visual_cnn import conv_output_dim, layer_init


class MapCNN(nn.Module):
    r"""A Simple CNN for processing map inputs (acoustic map or geometric map)

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, map_type='gm'):
        super().__init__()
        self._map_type = map_type
        self._n_input_gm = observation_space.spaces[map_type].shape[2]

        cnn_dims = np.array(
            observation_space.spaces[map_type].shape[:2], dtype=np.float32
        )
        # input image of dimension N reduces to (ceil((N-f+1)/s),ceil((N-f+1)/s),Number of filters)
        # where f is the filter size and s is the stride length
        # kernel size for different CNN layers
        if self._map_type == 'gm':
            if cnn_dims[0] == 200:
                self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

                # strides for different CNN layers
                self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
            else:
                assert cnn_dims[0] == 400
                self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

                # strides for different CNN layers
                self._cnn_layers_stride = [(5, 5), (4, 4), (2, 2)]
        elif self._map_type == 'am':
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]

            # strides for different CNN layers
            self._cnn_layers_stride = [(2, 2), (1, 1), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_gm,
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
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            #  nn.ReLU(True),
            Flatten(),
            nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )

        layer_init(self.cnn)

    def forward(self, observations):
        cnn_input = []

        gm_observations = observations[self._map_type]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        gm_observations = gm_observations.permute(0, 3, 1, 2)
        cnn_input.append(gm_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)

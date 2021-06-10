#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
import logging

from ss_baselines.common.utils import Flatten, ResizeCenterCropper
from ss_baselines.savi.ddppo.policy import resnet
from ss_baselines.savi.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.savi.ppo.policy import Net, Policy
from ss_baselines.savi.models.visual_cnn import VisualCNN
from ss_baselines.savi.models.audio_cnn import AudioCNN
from soundspaces.tasks.nav import PoseSensor, SpectrogramSensor, Category


class AudioNavResNetPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        resnet_baseplanes=32,
        backbone="resnet50",
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
        force_blind_policy=False,
        use_category_input=False,
        has_distractor_sound=False
    ):
        super().__init__(
            AudioNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
                force_blind_policy=force_blind_policy,
                use_category_input=use_category_input,
                has_distractor_sound=has_distractor_sound
            ),
            action_space.n,
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        make_backbone=None,
        normalize_visual_inputs=False,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
    ):
        super().__init__()

        self.obs_transform = obs_transform
        if self.obs_transform is not None:
            observation_space = self.obs_transform.transform_observation_space(
                observation_space
            )

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            spatial_size = observation_space.spaces["rgb"].shape[0] // 2
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._n_input_depth = 0

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial = int(
                spatial_size * self.backbone.final_spatial_compress
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial ** 2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations):
        if self.is_blind:
            return None

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

        if self.obs_transform:
            cnn_input = [self.obs_transform(inp) for inp in cnn_input]

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class AudioNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs,
        obs_transform=ResizeCenterCropper(size=(256, 256)),
        force_blind_policy=False,
        use_category_input=False,
        has_distractor_sound=False
    ):
        super().__init__()
        self._use_category_input = use_category_input
        self._hidden_size = hidden_size

        self._is_continuous = False
        if action_space.__class__.__name__ == "ActionSpace":
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        else:
            self.prev_action_embedding = nn.Linear(action_space.shape[0] + 1, 32)
            self._is_continuous = True
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action

        if backbone == 'custom_resnet18':
            # self.visual_encoder = SMTCNN(observation_space)
            self.visual_encoder = VisualCNN(observation_space, hidden_size)
        else:
            self.visual_encoder = ResNetEncoder(
                observation_space if not force_blind_policy else spaces.Dict({}),
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
                obs_transform=obs_transform,
            )
        if PoseSensor.cls_uuid in observation_space.spaces:
            self.pose_encoder = nn.Linear(5, 16)
            pose_feature_dims = 16
            rnn_input_size += pose_feature_dims

        if SpectrogramSensor.cls_uuid in observation_space.spaces:
            self.audio_encoder = AudioCNN(observation_space, 128, SpectrogramSensor.cls_uuid,
                                          has_distractor_sound=has_distractor_sound)
            rnn_input_size += 128
        else:
            logging.info("Input has no audio")

        if use_category_input:
            rnn_input_size += 21

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks):
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        if not self._is_continuous:
            prev_actions = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
            )
        else:
            prev_actions = self.prev_action_embedding(
                prev_actions.float() * masks
            )
        x.append(prev_actions)

        if SpectrogramSensor.cls_uuid in observations:
            x.append(self.audio_encoder(observations))

        if PoseSensor.cls_uuid in observations:
            pose_formatted = self._format_pose(observations[PoseSensor.cls_uuid])
            pose_encoded = self.pose_encoder(pose_formatted)
            x.append(pose_encoded)

        if self._use_category_input:
            x.append(observations[Category.cls_uuid])

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        ext_memory_feats = None

        return x, rnn_hidden_states, ext_memory_feats

    def _format_pose(self, pose):
        """
        Args:
            pose: (N, 4) Tensor containing x, y, heading, time
        """
        x, y, theta, time = torch.unbind(pose, dim=1)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        e_time = torch.exp(-time)
        formatted_pose = torch.stack([x, y, cos_theta, sin_theta, e_time], 1)
        return formatted_pose


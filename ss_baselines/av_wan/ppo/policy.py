#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging

import torch
import torch.nn as nn
from torchsummary import summary

from ss_baselines.common.utils import CategoricalNetWithMask
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.av_wan.models.visual_cnn import VisualCNN
from ss_baselines.av_wan.models.map_cnn import MapCNN
from ss_baselines.av_wan.models.audio_cnn import AudioCNN

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions, masking=True):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNetWithMask(
            self.net.output_size, self.dim_actions, masking
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
            self,
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
            self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features, observations['action_map'])
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class AudioNavBaselinePolicy(Policy):
    def __init__(
            self,
            observation_space,
            goal_sensor_uuid,
            masking,
            action_map_size,
            hidden_size=512,
            encode_rgb=False,
            encode_depth=False
    ):
        super().__init__(
            AudioNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                encode_rgb=encode_rgb,
                encode_depth=encode_depth
            ),
            # action_space.n,
            action_map_size ** 2,
            masking=masking
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, encode_rgb, encode_depth):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._spectrogram = False
        self._gm = 'gm' in observation_space.spaces
        self._am = 'am' in observation_space.spaces

        self._spectrogram = 'spectrogram' == self.goal_sensor_uuid
        self.visual_encoder = VisualCNN(observation_space, hidden_size, encode_rgb, encode_depth)
        if self._spectrogram:
            self.audio_encoder = AudioCNN(observation_space, hidden_size)
        if self._gm:
            self.gm_encoder = MapCNN(observation_space, hidden_size, map_type='gm')
        if self._am:
            self.am_encoder = MapCNN(observation_space, hidden_size, map_type='am')

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._hidden_size if self._spectrogram else 0) + \
                         (self._hidden_size if self._gm else 0) + \
                         (self._hidden_size if self._am else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and encode_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces and encode_depth:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if 'spectrogram' in observation_space.spaces:
            audio_shape = observation_space.spaces['spectrogram'].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')
        if self._gm:
            gm_shape = observation_space.spaces['gm'].shape
            summary(self.gm_encoder.cnn, (gm_shape[2], gm_shape[0], gm_shape[1]), device='cpu')
        if self._am:
            am_shape = observation_space.spaces['am'].shape
            summary(self.am_encoder.cnn, (am_shape[2], am_shape[0], am_shape[1]), device='cpu')

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

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []

        if self._spectrogram:
            x.append(self.audio_encoder(observations))
        if self._gm:
            x.append(self.gm_encoder(observations))
        if self._am:
            x.append(self.am_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states1

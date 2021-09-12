#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import pickle
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models

from habitat import logger

from ss_baselines.common.utils import ResizeCenterCropper
from ss_baselines.saven.models.smt_resnet import custom_resnet18
from habitat_sim.utils.common import d3_40_colors_rgb


class SMTCNN(nn.Module):
    r"""A modified ResNet-18 architecture from https://arxiv.org/abs/1903.03878.

    Takes in observations and produces an embedding of the rgb and/or depth
    and/or semantic components.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(
        self,
        observation_space,
        obs_transform: nn.Module = ResizeCenterCropper(size=(64, 64)),
    ):
        super().__init__()

        self.obs_transform = obs_transform
        if self.obs_transform is not None:
            observation_space = obs_transform.transform_observation_space(
                observation_space
            )

        self._feat_dims = 0
        self.input_modalities = []
        if "rgb" in observation_space.spaces:
            self.input_modalities.append("rgb")
            n_input_rgb = observation_space.spaces["rgb"].shape[2]
            self.rgb_encoder = custom_resnet18(num_input_channels=n_input_rgb)
            self._feat_dims += 64

        if "depth" in observation_space.spaces:
            self.input_modalities.append("depth")
            n_input_depth = observation_space.spaces["depth"].shape[2]
            self.depth_encoder = custom_resnet18(num_input_channels=n_input_depth)
            self._feat_dims += 64

        # Semantic instance segmentation
        if "semantic" in observation_space.spaces:
            # Semantic object segmentation
            self.input_modalities.append("semantic")
            self.input_modalities.append("semantic_object")
            self.semantic_encoder = custom_resnet18(num_input_channels=6)
            self._feat_dims += 64

        self.layer_init()

    def layer_init(self):
        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        self.apply(weights_init)

    def forward(self, observations):
        cnn_features = []
        if "rgb" in self.input_modalities:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            if self.obs_transform:
                rgb_observations = self.obs_transform(rgb_observations)
            cnn_features.append(self.rgb_encoder(rgb_observations))

        if "depth" in self.input_modalities:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            if self.obs_transform:
                depth_observations = self.obs_transform(depth_observations)
            cnn_features.append(self.depth_encoder(depth_observations))

        if "semantic" in self.input_modalities:
            assert "semantic_object" in observations.keys(), \
                "SMTCNN: Both instance and class segmentations must be available"
            semantic_observations = convert_semantics_to_rgb(
                observations["semantic"]
            ).float()
            semantic_object_observations = observations["semantic_object"].float()
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            semantic_observations = torch.cat(
                [semantic_observations, semantic_object_observations], -1
            )
            semantic_observations = semantic_observations.permute(0, 3, 1, 2) / 255.0
            if self.obs_transform:
                semantic_observations = self.obs_transform(semantic_observations)
            cnn_features.append(self.semantic_encoder(semantic_observations))

        cnn_features = torch.cat(cnn_features, dim=1)

        return cnn_features

    @property
    def feature_dims(self):
        return self._feat_dims

    # TODO: This is kept for backward compatibility. The two functions have to be
    # combined.
    @property
    def output_shape(self):
        return (self._feat_dims, )

    # TODO: This needs to be made available.
    @property
    def is_blind(self):
        return False


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

# Gyan Tatiya:

class SMTCNN_saven(nn.Module):

    def __init__(self, observation_space, obs_transform: nn.Module = ResizeCenterCropper(size=(64, 64))):
        super().__init__()

        self.obs_transform = obs_transform
        if self.obs_transform is not None:
            observation_space = obs_transform.transform_observation_space(observation_space)

        mp3d_objects_of_interest_filepath = r"data/metadata/mp3d_objects_of_interest_data.bin"
        with open(mp3d_objects_of_interest_filepath, 'rb') as bin_file:
            self.ooi_objects_id_name = pickle.load(bin_file)
            self.ooi_regions_id_name = pickle.load(bin_file)

        self.num_objects = len(self.ooi_objects_id_name)
        self.num_regions = len(self.ooi_regions_id_name)

        self._feat_dims = 0
        self.input_modalities = []
        if "rgb" in observation_space.spaces:
            self.input_modalities.append("rgb")
            n_input_rgb = observation_space.spaces["rgb"].shape[2]
            # self.rgb_encoder = custom_resnet18(num_input_channels=n_input_rgb)

            logger.info("Loading pre-trained vision network for rgb")
            self.rgb_encoder = models.resnet18(pretrained=True)
            output_size = self.num_objects + self.num_regions
            num_ftrs = self.rgb_encoder.fc.in_features
            self.rgb_encoder.fc = nn.Linear(num_ftrs, output_size)

            state_dict = torch.load('data/models/saven_gt/vision/best_test.pth', map_location="cpu")
            cleaned_state_dict = {k[len('module.predictor.'):]: v for k, v in state_dict['vision_predictor'].items()
                                  if 'module.predictor.' in k}
            self.rgb_encoder.load_state_dict(cleaned_state_dict)

            self.rgb_encoder.fc = nn.Linear(num_ftrs, 64)

            self._feat_dims += 64

        if "depth" in observation_space.spaces:
            self.input_modalities.append("depth")
            n_input_depth = observation_space.spaces["depth"].shape[2]
            # self.depth_encoder = custom_resnet18(num_input_channels=n_input_depth)

            logger.info("Loading pre-trained vision network for depth")
            self.depth_encoder = models.resnet18(pretrained=True)
            output_size = self.num_objects + self.num_regions
            num_ftrs = self.depth_encoder.fc.in_features
            self.depth_encoder.fc = nn.Linear(num_ftrs, output_size)

            state_dict = torch.load('data/models/saven_gt/vision/best_test.pth', map_location="cpu")
            cleaned_state_dict = {k[len('module.predictor.'):]: v for k, v in state_dict['vision_predictor'].items()
                                  if 'module.predictor.' in k}
            self.depth_encoder.load_state_dict(cleaned_state_dict)

            self.depth_encoder.conv1 = nn.Conv2d(n_input_depth, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.depth_encoder.fc = nn.Linear(num_ftrs, 64)

            self._feat_dims += 64

        # Semantic instance segmentation
        # if "semantic" in observation_space.spaces:
        #     # Semantic object segmentation
        #     self.input_modalities.append("semantic")
        #     self.input_modalities.append("semantic_object")
        #     self.semantic_encoder = custom_resnet18(num_input_channels=6)
        #     self._feat_dims += 64
        #
        # self.layer_init()

    # def layer_init(self):
    #     def weights_init(m):
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             nn.init.kaiming_normal_(
    #                 m.weight, nn.init.calculate_gain("relu")
    #             )
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, val=0)
    #
    #     self.apply(weights_init)

    def forward(self, observations):
        cnn_features = []
        if "rgb" in self.input_modalities:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            if self.obs_transform:
                rgb_observations = self.obs_transform(rgb_observations)
            cnn_features.append(self.rgb_encoder(rgb_observations))

        if "depth" in self.input_modalities:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            if self.obs_transform:
                depth_observations = self.obs_transform(depth_observations)
            cnn_features.append(self.depth_encoder(depth_observations))

        # if "semantic" in self.input_modalities:
        #     assert "semantic_object" in observations.keys(), \
        #         "SMTCNN: Both instance and class segmentations must be available"
        #     semantic_observations = convert_semantics_to_rgb(
        #         observations["semantic"]
        #     ).float()
        #     semantic_object_observations = observations["semantic_object"].float()
        #     # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        #     semantic_observations = torch.cat(
        #         [semantic_observations, semantic_object_observations], -1
        #     )
        #     semantic_observations = semantic_observations.permute(0, 3, 1, 2) / 255.0
        #     if self.obs_transform:
        #         semantic_observations = self.obs_transform(semantic_observations)
        #     cnn_features.append(self.semantic_encoder(semantic_observations))

        cnn_features = torch.cat(cnn_features, dim=1)

        return cnn_features

    @property
    def feature_dims(self):
        return self._feat_dims

    # TODO: This is kept for backward compatibility. The two functions have to be
    # combined.
    @property
    def output_shape(self):
        return (self._feat_dims, )

    # TODO: This needs to be made available.
    @property
    def is_blind(self):
        return False

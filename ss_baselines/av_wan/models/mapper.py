#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import logging

import torch.nn as nn
import torch
import numpy as np

from habitat.sims.habitat_simulator.actions import HabitatSimActions


def to_array(x):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    else:
        x = x
    return x


class Mapper(nn.Module):
    def __init__(self, gm_config, am_config, action_map_config, use_acoustic_map):
        super(Mapper, self).__init__()
        self._internal_gm_size = gm_config.INTERNAL_MAP_SIZE
        self._gm_size = gm_config.MAP_SIZE
        self._gm_res = gm_config.MAP_RESOLUTION
        self._use_acoustic_map = use_acoustic_map
        self._am_encoding = am_config.ENCODING
        self._action_map_res = action_map_config.MAP_RESOLUTION
        self._stride = int(self._action_map_res / self._gm_res)
        # allocentric map w.r.t the agent's initial pose
        # both global map and local maps are square, the first channel is obstacle map and the second is explored map
        self._geometric_map = None
        self._prev_geometric_map = None
        self._acoustic_map = None
        self._x = None
        self._y = None
        self._orientation = None
        self._initial_orientation = None
        self._navigable_xs = None
        self._navigable_ys = None
        self._rotated_xs = dict()
        self._rotated_ys = dict()

        self.reset()

    def compute_navigable_xys(self):
        navigable_xs = []
        for n in range(int(-self._x / self._stride), int((self._internal_gm_size - self._x) / self._stride)):
            navigable_xs.append(self._x + n * self._stride)

        navigable_ys = []
        for n in range(int(-self._y / self._stride), int((self._internal_gm_size - self._y) / self._stride)):
            navigable_ys.append(self._y + n * self._stride)

        self._navigable_xs, self._navigable_ys = navigable_xs, navigable_ys

        for angle in [0, 90, 180, 270]:
            navigable_xs = []
            navigable_ys = []
            for a, b in zip(self._navigable_xs, self._navigable_ys):
                c, d = transform_coordinates(a, b, angle, self._geometric_map.shape[1], self._geometric_map.shape[0])
                navigable_xs.append(c)
                navigable_ys.append(d)
            navigable_xs = sorted(navigable_xs)
            navigable_ys = sorted(navigable_ys)
            self._rotated_xs[angle] = navigable_xs
            self._rotated_ys[angle] = navigable_ys

        return self._navigable_xs, self._navigable_ys

    def reset(self):
        self._geometric_map = np.zeros((self._internal_gm_size, self._internal_gm_size, 2))
        if self._use_acoustic_map:
            if self._am_encoding == 'intensity':
                num_channel = 1
            elif self._am_encoding == 'average_intensity':
                num_channel = 1
            else:
                raise ValueError
            self._acoustic_map = np.zeros((self._internal_gm_size // self._stride,
                                           self._internal_gm_size // self._stride, num_channel))
        self._x = int(self._internal_gm_size / 2)
        self._y = int(self._internal_gm_size / 2)
        # set the initial orientation to be 270 on X-Z plane in 3D coordinate frame
        self._orientation = 270
        self._initial_orientation = self._orientation

    @property
    def _rotation(self):
        # orientation increases clockwise, rotation increases counterclockwise
        return -(self._orientation - self._initial_orientation)

    def update(self, prev_action: int, ego_map, intensity) -> Tuple[list, list]:
        if logging.root.level == logging.DEBUG:
            self._prev_geometric_map = np.copy(self._geometric_map)

        if prev_action == HabitatSimActions.MOVE_FORWARD:
            self._x += int(self._stride * np.cos(np.deg2rad(self._orientation)))
            self._y += int(self._stride * np.sin(np.deg2rad(self._orientation)))
        elif prev_action == HabitatSimActions.TURN_LEFT:
            self._orientation = (self._orientation - 90) % 360
        elif prev_action == HabitatSimActions.TURN_RIGHT:
            self._orientation = (self._orientation + 90) % 360
        else:
            # do nothing for the first step
            pass

        # update global map
        rotated_geometric_map = rotate_map(self._geometric_map, -self._rotation, create_copy=False)
        rotated_x, rotated_y = transform_coordinates(self._x, self._y, -self._rotation,
                                                     self._geometric_map.shape[1], self._geometric_map.shape[0])
        left = rotated_x - int(ego_map.shape[1] / 2)
        right = left + ego_map.shape[1]
        # does not update the agent's current location
        top = rotated_y
        bottom = top - ego_map.shape[0]
        rotated_geometric_map[bottom: top, left: right, :] = \
            np.logical_or(rotated_geometric_map[bottom: top, left: right, :] > 0.5, ego_map > 0.5)

        # update acoustic map
        if self._use_acoustic_map:
            am_x = self._x // self._stride
            am_y = self._y // self._stride
            if self._am_encoding == 'intensity':
                self._acoustic_map[am_y, am_x, 0] = intensity
            elif self._am_encoding == 'average_intensity':
                if self._acoustic_map[am_y, am_x] == 0:
                    self._acoustic_map[am_y, am_x] = intensity
                else:
                    self._acoustic_map[am_y, am_x] = 0.5 * intensity + 0.5 * self._acoustic_map[am_y, am_x]

        # compute new blocked paths and non-navigable points in the affected region
        new_left = max(left - self._stride, 0)
        new_bottom = max(bottom - self._stride, 0)
        new_right = min(right + self._stride, self._geometric_map.shape[1])
        new_top = min(top + self._stride, self._geometric_map.shape[0])
        m = self._stride
        navigable_xs = []
        for n in range(int((new_left - rotated_x) / m), int((new_right + 1 - rotated_x) / m)):
            navigable_xs.append(rotated_x + n * m)
        navigable_ys = []
        for n in range(int((new_bottom - rotated_y) / m), int((new_top + 1 - rotated_y) / m)):
            navigable_ys.append(rotated_y + n * m)

        def convert(a, b):
            return transform_coordinates(a, b, self._rotation, rotated_geometric_map.shape[1], rotated_geometric_map.shape[0])

        non_navigable_points = []
        blocked_paths = []
        for idx_y, y in enumerate(navigable_ys):
            for idx_x, x in enumerate(navigable_xs):
                if rotated_geometric_map[y, x, 0]:
                    if x == rotated_x and y == rotated_y:
                        logging.warning("Mapper: marked current position as obstacle")
                        self._geometric_map[self._y, self._x, 0] = 0
                    else:
                        non_navigable_points.append(convert(x, y))

                # no obstacle to the next navigable point along +Z direction
                if idx_y < len(navigable_ys) - 1:
                    next_y = navigable_ys[idx_y + 1]
                    if any(rotated_geometric_map[y: next_y + 1, x, 0]):
                        blocked_paths.append((convert(x, y), convert(x, next_y)))

                # no obstacle to the next navigable point along +X direction
                if idx_x < len(navigable_xs) - 1:
                    next_x = navigable_xs[idx_x + 1]
                    if any(rotated_geometric_map[y, x: next_x + 1, 0]):
                        blocked_paths.append((convert(x, y), convert(next_x, y)))
        assert (self._x, self._y) not in non_navigable_points
        return non_navigable_points, blocked_paths

    def get_adjacent_point_coordinates(self):
        return self._x + int(self._stride * np.cos(np.deg2rad(self._orientation))), \
                self._y + int(self._stride * np.sin(np.deg2rad(self._orientation)))

    def get_maps_and_agent_pose(self):
        return self._geometric_map, self._acoustic_map, self._x, self._y, self._orientation

    def get_orientation(self):
        return self._orientation

    def egocentric_to_allocentric(self, delta_x, delta_y, action_map_res=None):
        """
        apply the agent's rotation to the relative delta_x, delta_y, rotates counterclockwise

        """
        if action_map_res is not None:
            delta_x *= int(action_map_res / self._gm_res)
            delta_y *= int(action_map_res / self._gm_res)
        rotation = self._rotation % 360
        if rotation == 0:
            return delta_x, delta_y
        elif rotation == 90:
            return delta_y, -delta_x
        elif rotation == 180:
            return -delta_x, -delta_y
        else:
            return -delta_y, delta_x

    def allocentric_to_egocentric(self, x, y, action_map_res=None):
        if action_map_res is not None:
            x /= int(action_map_res / self._gm_res)
            y /= int(action_map_res / self._gm_res)

        rotation = self._rotation % 360
        if rotation == 0:
            return x, y
        elif rotation == 90:
            return -y, x
        elif rotation == 180:
            return -x, -y
        else:
            return y, -x

    def global_to_egocentric(self, x, y):
        return self.allocentric_to_egocentric(x - self._x, y - self._y, self._action_map_res)

    def egocentric_to_global(self, delta_x, delta_y):
        allocentric = self.egocentric_to_allocentric(delta_x, delta_y, self._action_map_res)
        return self._x + allocentric[0], self._y + allocentric[1]

    def is_explored(self, x, y):
        return self._geometric_map[y][x][1] > 0.5

    def get_egocentric_geometric_map(self):
        # crop internal gm to external gm
        rotated_geometric_map = rotate_map(self._geometric_map, -self._rotation, create_copy=False)
        x, y = transform_coordinates(self._x, self._y, -self._rotation,
                                     self._geometric_map.shape[1], self._geometric_map.shape[0])
        map_size = rotated_geometric_map.shape[0]

        cropped_map = np.zeros((self._gm_size, self._gm_size, self._geometric_map.shape[2]))
        top = max(self._gm_size // 2 - y, 0)
        left = max(self._gm_size // 2 - x, 0)
        bottom = min(map_size + self._gm_size // 2 - y, self._gm_size)
        right = min(map_size + self._gm_size // 2 - x, self._gm_size)
        cropped_map[top: bottom, left: right] = \
            rotated_geometric_map[max(y - self._gm_size // 2, 0):
                                  min(y + self._gm_size // 2, map_size),
                                  max(x - self._gm_size // 2, 0):
                                  min(x + self._gm_size // 2, map_size), :]

        return cropped_map

    def get_egocentric_acoustic_map(self, crop_map_size=20):
        channels = []

        if self._am_encoding == 'intensity':
            acoustic_map = self._acoustic_map
        elif self._am_encoding == 'average_intensity':
            acoustic_map = self._acoustic_map
        else:
            raise ValueError('Encoding does not exist')
        rotated_acoustic_map = rotate_map(acoustic_map, -self._rotation, create_copy=False)
        x, y = transform_coordinates(self._x // self._stride, self._y // self._stride,
                                     -self._rotation, acoustic_map.shape[1], acoustic_map.shape[0])
        map_size = rotated_acoustic_map.shape[0]
        cropped_map = np.zeros((crop_map_size, crop_map_size, rotated_acoustic_map.shape[2]))
        top = max(crop_map_size // 2 - y, 0)
        left = max(crop_map_size // 2 - x, 0)
        bottom = min(map_size + crop_map_size // 2 - y, crop_map_size)
        right = min(map_size + crop_map_size // 2 - x, crop_map_size)
        cropped_map[top: bottom, left: right] = \
            rotated_acoustic_map[max(y - crop_map_size // 2, 0):
                                 min(y + crop_map_size // 2, map_size),
                                 max(x - crop_map_size // 2, 0):
                                 min(x + crop_map_size // 2, map_size), :]
        channels.append(cropped_map)
        channels = np.concatenate(channels, axis=2)

        return channels

    def get_egocentric_occupancy_map(self, size, action_map_res):
        # 1 represent free space and 0 represents occupancy
        rotated_geometric_map = rotate_map(self._geometric_map, -self._rotation, create_copy=False)
        x, y = transform_coordinates(self._x, self._y, -self._rotation,
                                     self._geometric_map.shape[1], self._geometric_map.shape[0])
        grid_map = rotated_geometric_map[np.ix_(self._rotated_ys[-self._rotation % 360],
                                                self._rotated_xs[-self._rotation % 360])]
        grid_x = x // self._stride
        grid_y = y // self._stride
        ego_om = 1 - grid_map[grid_y - size // 2: grid_y + size // 2 + 1,
                              grid_x - size // 2: grid_x + size // 2 + 1, 0]

        if logging.root.level == logging.DEBUG:
            for j in range(size):
                for i in range(size):
                    navigability = ego_om[j, i]
                    pg_x = int(i - size // 2)
                    pg_y = int(j - size // 2)
                    delta_x, delta_y = self.egocentric_to_allocentric(pg_x, pg_y, action_map_res=action_map_res)
                    goal_x = self._x + delta_x
                    goal_y = self._y + delta_y
                    assert navigability == (grid_map[grid_y + pg_y, grid_x + pg_x, 0] == 0) \
                                        == (self._geometric_map[goal_y, goal_x, 0] == 0)

        return ego_om


def rotate_map(om: np.array, rotation: float, create_copy=True) -> np.array:
    """
    rotate the input map counterclockwise
    :param om:
    :param rotation: counterclockwise, from axis 0 to axis 1
    :param create_copy: decides whether the returned map is a copy of the original
    :return:
    """
    rotation = rotation % 360
    if create_copy:
        rotated_map = np.copy(om)
    else:
        rotated_map = om

    if rotation != 0:
        rotated_map = np.rot90(rotated_map, k=int(rotation / 90))

    return rotated_map


def transform_coordinates(x: int, y: int, rotation: int, width: int, height: int) -> Tuple[int, int]:
    """
    Rotates x,y counterclockwise
    """
    rotation = rotation % 360
    if rotation == 0:
        new_x = x
        new_y = y
    elif rotation == 90:
        new_x = y
        new_y = width - x - 1
    elif rotation == 180:
        new_x = width - x - 1
        new_y = height - y - 1
    else:
        new_x = height - y - 1
        new_y = x

    return new_x, new_y
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import numpy as np
import torch
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb


def load_metadata(parent_folder):
    points_file = os.path.join(parent_folder, 'points.txt')
    if "replica" in parent_folder:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5528907,
            -points_data[:, 2])
        )
    else:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5,
            -points_data[:, 2])
        )
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph


def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def convert_semantic_object_to_rgb(x):
    semantic_img = Image.new("P", (x.shape[1], x.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((x.flatten() % 40).astype(np.uint8))
    semantic_img = np.array(semantic_img.convert("RGB"))
    return semantic_img
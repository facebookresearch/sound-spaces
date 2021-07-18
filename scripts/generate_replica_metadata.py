#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import os
import shutil

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import habitat_sim
import habitat_sim.bindings as hsim
from soundspaces.utils import load_metadata


def adjust_graph(graph, points, name):
    # manual fixes
    if name == 'apartment_0':
        upper_floor_nodes = list()
        for node in graph.nodes():
            if graph.nodes[node]['point'][1] > 0.5:
                upper_floor_nodes.append(node)
        graph.remove_nodes_from(upper_floor_nodes)
        graph.remove_nodes_from([0, 30, 31, 32, 47, 48, 49, 50, 67, 90, 173, 378, 509])
    elif name == 'apartment_2':
        graph.remove_nodes_from([0, 168, 187, 218, 237])
        graph.add_node(191, point=points[191])
        graph.add_edges_from([(173, 191), (191, 209)])
    elif name == 'frl_apartment_1':
        graph.remove_nodes_from([164, 177])
    elif name == 'frl_apartment_2':
        graph.remove_nodes_from([174])
    elif name == 'frl_apartment_5':
        graph.remove_nodes_from([9, 18, 26, 35])
    elif name == 'hotel_0':
        graph.remove_nodes_from([1])
    elif name == 'office_0':
        # check
        graph.remove_nodes_from([6, 7])
        graph.add_edges_from([(12, 21)])
    elif name == 'office_2':
        graph.remove_nodes_from([0, 11, 3, 9, 15, 10, 16])
    elif name == 'office_3':
        graph.remove_nodes_from([48, 82, 115])
        # graph.add_edges_from([(56, 69)])
    elif name == 'room_0':
        graph.remove_nodes_from([123, 124, 125, 126, 127, 118, 117, 102, 103, 111, 112, 120, 121])
        # graph.add_edges_from([(95, 103)])
    elif name == 'room_1':
        # graph.add_edges_from([(37, 45), (51, 59)])
        graph.remove_nodes_from([45, 51])
    else:
        return False
    return True


def visualize(points, graph, filename=None, save_figure=False, plot_indices=False, output_dir=''):
    if not plot_indices:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        if filename is not None:
            fig.suptitle(filename, fontsize=20)
        for point in points:
            ax1.scatter(point[0], point[2], 9, c='black')
        ax1.set_title('All Points')

        for node in graph.nodes():
            point = graph.nodes[node]['point']
            ax2.scatter(point[0], point[2], 9, c='black')
        ax2.set_title('Naivigable Points')

        for node in graph.nodes():
            point = graph.nodes[node]['point']
            ax3.scatter(point[0], point[2], 9, c='black')

        for n1, n2 in graph.edges():
            p1 = graph.nodes[n1]['point']
            p2 = graph.nodes[n2]['point']
            ax3.plot([p1[0], p2[0]], [p1[2], p2[2]], c='green')
        ax3.set_title('Connected Graph')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        if filename is not None:
            fig.suptitle(filename, fontsize=20)
        for node in graph.nodes():
            point = graph.nodes[node]['point']
            ax.scatter(point[0], point[2], 9, c='black')
            ax.annotate(str(node), (point[0], point[2]))
        for n1, n2 in graph.edges():
            p1 = graph.nodes[n1]['point']
            p2 = graph.nodes[n2]['point']
            ax.plot([p1[0], p2[0]], [p1[2], p2[2]], c='green')

    if save_figure:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        plt.savefig(file_path)
    plt.close()


def generate_graph(points, pathfinder):
    navigable_idx = [i for i, p in enumerate(points) if pathfinder.is_navigable(p)]
    graph = nx.Graph()
    for idx in navigable_idx:
        graph.add_node(idx, point=points[idx])

    for a_idx, a_loc in enumerate(points):
        if a_idx not in navigable_idx:
            continue
        for b_idx, b_loc in enumerate(points):
            if b_idx not in navigable_idx:
                continue
            if a_idx == b_idx:
                continue

            euclidean_distance = np.linalg.norm(np.array(a_loc) - np.array(b_loc))
            if 0.1 < euclidean_distance < 0.51:
                path = habitat_sim.ShortestPath()
                path.requested_start = np.array(a_loc, dtype=np.float32)
                path.requested_end = np.array(b_loc, dtype=np.float32)
                pathfinder.find_path(path)
                # relax the constraint a bit
                if path.geodesic_distance < 0.8:
                    graph.add_edge(a_idx, b_idx)

    return graph


def main():
    metadata_folder = os.path.join('data/metadata/replica')
    scenes = os.listdir(metadata_folder)
    for scene in scenes:
        navmesh_file = "data/scene_datasets/replica/{}/habitat/mesh_semantic.navmesh".format(scene)
        scene_metadata_folder = os.path.join(metadata_folder, scene)
        graph_file = os.path.join(scene_metadata_folder, 'graph.pkl')
        visualization_dir = 'data/visualizations/replica'
        os.makedirs(scene_metadata_folder, exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)

        pathfinder = hsim.PathFinder()
        pathfinder.load_nav_mesh(navmesh_file)
        points, _ = load_metadata(scene_metadata_folder)

        graph = generate_graph(points, pathfinder)
        visualize(points, graph, scene, save_figure=True, plot_indices=True, output_dir=visualization_dir)

        adjusted = adjust_graph(graph, points, scene)
        if adjusted:
            visualize(points, graph, scene + '_fix', save_figure=True, plot_indices=True, output_dir=visualization_dir)

        with open(graph_file, 'wb') as fo:
            pickle.dump(graph, fo)


if __name__ == '__main__':
    main()

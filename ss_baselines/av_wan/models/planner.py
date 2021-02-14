#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import networkx as nx
import numpy as np
import torch
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from ss_baselines.av_wan.models.mapper import Mapper, to_array


class Planner:
    def __init__(self, task_config=None, use_acoustic_map=False, model_dir=None, masking=True):
        self.mapper = Mapper(
            gm_config=task_config.TASK.GEOMETRIC_MAP,
            am_config=task_config.TASK.ACOUSTIC_MAP,
            action_map_config=task_config.TASK.ACTION_MAP,
            use_acoustic_map=use_acoustic_map
        )

        self._action_map_res = task_config.TASK.ACTION_MAP.MAP_RESOLUTION
        self._action_map_size = task_config.TASK.ACTION_MAP.MAP_SIZE
        self._prev_depth = None
        self._prev_next_node = None
        self._prev_action = None
        self._obstacles = []
        self._obstacle_threshold = 0.5
        self._navigable_xs, self._navigable_ys = self.mapper.compute_navigable_xys()
        self._graph = self._map_to_graph(self.mapper.get_maps_and_agent_pose()[0])
        self._removed_edges = list()
        self._removed_nodes = list()
        self._model_dir = model_dir
        self._masking = masking

        self.reset()

    def reset(self):
        self._prev_depth = None
        self._prev_next_node = None
        self._prev_action = None
        self._obstacles = []
        self.mapper.reset()
        self._graph.add_nodes_from(self._removed_nodes)
        self._graph.add_edges_from(self._removed_edges)
        self._removed_nodes.clear()
        self._removed_edges.clear()

    def update_map_and_graph(self, observation):
        ego_map = to_array(observation['ego_map'])
        depth = to_array(observation['depth'])
        collided = to_array(observation['collision'][0])
        intensity = to_array(observation['intensity'][0]) if 'intensity' in observation else None

        geometric_map, acoustic_map, x, y, orientation = self.mapper.get_maps_and_agent_pose()
        if not collided:
            non_navigable_points, blocked_paths = self.mapper.update(self._prev_action, ego_map, intensity)
            self._update_graph(non_navigable_points, blocked_paths)
        elif self._prev_next_node in self._graph.nodes:
            # only the edge to the previous next node should be removed
            current_node = self._map_index_to_graph_nodes([(x, y)])[0]
            self._graph.remove_edge(self._prev_next_node, current_node)
            self._removed_edges.append((self._prev_next_node, current_node))
        self._prev_depth = depth

        if logging.root.level == logging.DEBUG:
            geometric_map, acoustic_map, x, y, orientation = self.mapper.get_maps_and_agent_pose()
            assert not geometric_map[y, x, 0]
            for node, attr in self._removed_nodes:
                index = attr['map_index']
                assert self.mapper._geometric_map[index[1], index[0]][0]

    def add_maps_to_observation(self, observation):
        if 'gm' in observation:
            observation['gm'] = self.mapper.get_egocentric_geometric_map().astype(np.float32)
        if 'am' in observation:
            observation['am'] = self.mapper.get_egocentric_acoustic_map().astype(np.float32)
        if 'action_map' in observation:
            observation['action_map'] = np.expand_dims(self.mapper.get_egocentric_occupancy_map(
                    size=self._action_map_size, action_map_res=self._action_map_res), -1).astype(np.float32)

    def plan(self, observation: dict, goal, stop, distribution=None) -> torch.Tensor:
        geometric_map, acoustic_map, x, y, orientation = self.mapper.get_maps_and_agent_pose()
        graph_nodes = self._map_index_to_graph_nodes([(x, y), (goal[0], goal[1])])

        next_node = next_node_idx = None
        if stop:
            action = HabitatSimActions.STOP
            self._prev_next_node = None
        else:
            try:
                shortest_path = nx.shortest_path(self._graph, source=graph_nodes[0], target=graph_nodes[1])
                # decide if the agent needs to rotate based on the connectivity with the next node
                next_node_idx = self._graph.nodes[shortest_path[1]]['map_index']
                self._prev_next_node = shortest_path[1]
                desired_orientation = np.round(
                    np.rad2deg(np.arctan2(next_node_idx[1] - y, next_node_idx[0] - x))) % 360
                rotation = (desired_orientation - orientation) % 360

                # egocentric frame where the agent faces +x direction
                if rotation == 0:
                    action = HabitatSimActions.MOVE_FORWARD
                elif rotation == 90:
                    action = HabitatSimActions.TURN_RIGHT
                elif rotation == 180:
                    action = np.random.choice([HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT])
                elif rotation == 270:
                    action = HabitatSimActions.TURN_LEFT
                else:
                    raise ValueError('Invalid rotation')
            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound) as e:
                assert not (self._masking and isinstance(e, nx.exception.NodeNotFound))
                # randomly select a node from neighbors
                adjacent_point_coordinates = self.mapper.get_adjacent_point_coordinates()
                adjacent_node = self._map_index_to_graph_nodes([adjacent_point_coordinates])[0]
                if adjacent_node in self._graph.nodes and (graph_nodes[0], adjacent_node) in self._graph.edges:
                    action = np.random.choice([HabitatSimActions.MOVE_FORWARD, HabitatSimActions.TURN_LEFT,
                                               HabitatSimActions.TURN_RIGHT])
                else:
                    action = np.random.choice([HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT])
                self._prev_next_node = None
        self._prev_action = action

        return action

    def get_map_coordinates(self, relative_goal):
        map_size = self._action_map_size
        geometric_map, acoustic_map, x, y, orientation = self.mapper.get_maps_and_agent_pose()
        pg_y, pg_x = np.unravel_index(relative_goal, (map_size, map_size))
        pg_x = int(pg_x - map_size // 2)
        pg_y = int(pg_y - map_size // 2)

        # transform goal location to be in the global coordinate frame
        delta_x, delta_y = self.mapper.egocentric_to_allocentric(pg_x, pg_y, action_map_res=self._action_map_res)
        return x + delta_x, y + delta_y

    def check_navigability(self, goal):
        geometric_map, acoustic_map, x, y, orientation = self.mapper.get_maps_and_agent_pose()
        graph_nodes = self._map_index_to_graph_nodes([(x, y), goal])
        navigable = graph_nodes[1] in self._graph.nodes, graph_nodes[1] in self._graph.nodes \
                        and nx.has_path(self._graph, source=graph_nodes[0], target=graph_nodes[1])

        return all(navigable)

    def _update_graph(self, non_navigable_points, blocked_paths):
        non_navigable_nodes = self._map_index_to_graph_nodes(non_navigable_points)
        blocked_edges = [self._map_index_to_graph_nodes([a, b]) for a, b in blocked_paths]

        for node in non_navigable_nodes:
            if node in self._graph.nodes:
                self._removed_nodes.append((node, self._graph.nodes[node]))
                self._removed_edges += [(node, neighbor) for neighbor in self._graph[node]]
        self._removed_edges += blocked_edges

        self._graph.remove_nodes_from(non_navigable_nodes)
        self._graph.remove_edges_from(blocked_edges)

    def _map_index_to_graph_nodes(self, map_indices: list) -> list:
        graph_nodes = list()
        for map_index in map_indices:
            graph_nodes.append(map_index[1] * len(self._navigable_ys) + map_index[0])
        return graph_nodes

    def _map_to_graph(self, geometric_map: np.array) -> nx.Graph:
        # after bitwise_and op, 0 indicates free or unexplored, 1 indicate obstacles
        occupancy_map = np.bitwise_and(geometric_map[:, :, 0] >= self._obstacle_threshold,
                                       geometric_map[:, :, 1] >= self._obstacle_threshold)
        graph = nx.Graph()
        for idx_y, y in enumerate(self._navigable_ys):
            for idx_x, x in enumerate(self._navigable_xs):
                node_index = y * len(self._navigable_ys) + x

                if occupancy_map[y][x]:
                    # obstacle
                    continue

                # no obstacle to the next navigable point along +Z direction
                if idx_y < len(self._navigable_ys) - 1:
                    next_y = self._navigable_ys[idx_y + 1]
                    if not any(occupancy_map[y: next_y+1, x]):
                        next_node_index = next_y * len(self._navigable_ys) + x
                        if node_index not in graph:
                            graph.add_node(node_index, map_index=(x, y))
                        if next_node_index not in graph:
                            graph.add_node(next_node_index, map_index=(x, next_y))
                        graph.add_edge(node_index, next_node_index)

                # no obstacle to the next navigable point along +X direction
                if idx_x < len(self._navigable_xs) - 1:
                    next_x = self._navigable_xs[idx_x + 1]
                    if not any(occupancy_map[y, x: next_x+1]):
                        next_node_index = y * len(self._navigable_ys) + next_x
                        if node_index not in graph:
                            graph.add_node(node_index, map_index=(x, y))
                        if next_node_index not in graph:
                            graph.add_node(next_node_index, map_index=(next_x, y))
                        graph.add_edge(node_index, next_node_index)

        # trim the graph such that it only keeps the largest subgraph
        connected_subgraphs = (graph.subgraph(c) for c in nx.connected_components(graph))
        max_connected_graph = max(connected_subgraphs, key=len)

        return nx.Graph(max_connected_graph)

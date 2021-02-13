#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Tuple, Union, Dict, Any
import copy

import networkx as nx
import numpy as np
import torch
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from habitat.utils.visualizations.utils import observations_to_image
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from av_wan.rl.models.mapper import Mapper, to_array


class Planner:
    def __init__(self, task_config=None, use_acoustic_map=False, plot_map=False, model_dir=None, masking=True):
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

        self._step_count = 0
        self.plot_map = plot_map
        self._output_dir = None

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
        # assert len(self._graph.nodes) == 10000 and len(self._graph.edges) == 19800

        self._step_count = 0
        self._output_dir = os.path.join(self._model_dir, 'planner_maps', str(np.random.randint(0, 100000)))

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
                # logging.warning("planner.act(): no path to {}, perform random action {}".format(goal, action))
                # self.plot_top_down_map(geometric_map, x, y, delta_x, delta_y, observation, distribution, next_node=None,
                #                        next_node_idx=None, show=False)
                self._prev_next_node = None
        self._prev_action = action

        self._step_count += 1
        if self.plot_map:
            self.plot_top_down_map(geometric_map, x, y, goal[0] - x, goal[1] - y, observation, distribution,
                                   next_node=next_node, next_node_idx=next_node_idx, show=False)

        return action

    def plan_and_act(self, envs, env_index, observation, relative_goal, action_map_size, max_num_step,
                     distribution=None, frames=None, audio_files=None):
        goal = self.get_map_coordinates(relative_goal)
        done = False
        cumulative_reward = 0

        for step_count in range(max_num_step):
            # TODO: better way of handling non-navigable goal in the beginning
            if done or (step_count != 0 and not self.check_navigability(goal)):
                break

            geometric_map, acoustic_map, x, y, orientation = self.mapper.get_maps_and_agent_pose()
            graph_nodes = self._map_index_to_graph_nodes([(x, y), (goal[0], goal[1])])

            next_node = next_node_idx = None
            if relative_goal == action_map_size ** 2 // 2:
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
                        logging.warning('Invalid rotation: {}'.format(rotation))
                        raise nx.exception.NetworkXNoPath
                except nx.exception.NetworkXNoPath:
                    # randomly select a node from neighbors
                    adjacent_point_coordinates = self.mapper.get_adjacent_point_coordinates()
                    adjacent_node = self._map_index_to_graph_nodes([adjacent_point_coordinates])[0]
                    if adjacent_node in self._graph.nodes and (graph_nodes[0], adjacent_node) in self._graph.edges:
                        action = np.random.choice([HabitatSimActions.MOVE_FORWARD, HabitatSimActions.TURN_LEFT,
                                                   HabitatSimActions.TURN_RIGHT])
                    else:
                        action = np.random.choice([HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT])
                    # logging.warning("planner.act(): no path to {}, perform random action {}".format(goal, action))
                    self._prev_next_node = None
            self._prev_action = action

            self._step_count += 1
            if self.plot_map:
                plot_distribution = distribution if step_count == 0 else None
                self.plot_top_down_map(geometric_map, x, y, goal[0] - x, goal[1] - y, observation, plot_distribution,
                                       next_node=next_node, next_node_idx=next_node_idx, show=False)

            observation, reward, done, info = envs.step_at(env_index, {"action": action})[0]
            self.update_map_and_graph(observation)
            cumulative_reward += reward
            if done:
                self.reset()
                self.update_map_and_graph(observation)

            if frames is not None:
                if "rgb" not in observation:
                    observation["rgb"] = np.zeros((128, 128, 3))
                frame = observations_to_image(observation, info)
                frames.append(frame)
                audio_files.append(info['audio_file'])
                del observation["rgb"]

            # reaching intermediate goal
            x, y = self.mapper.get_maps_and_agent_pose()[2:4]
            if (x - goal[0]) == (y - goal[1]) == 0:
                break

        self.add_maps_to_observation(observation)

        return observation, cumulative_reward, done, info

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

    def find_closest_navigable_point(self, goal):
        geometric_map, acoustic_map, x, y, orientation = self.mapper.get_maps_and_agent_pose()
        coarser_map = geometric_map[:, :, 0][np.ix_(self._navigable_ys, self._navigable_xs)]

        agent_x = int(x * self.mapper._gm_res / self._action_map_res)
        agent_y = int(y * self.mapper._gm_res / self._action_map_res)

        goal = np.clip(goal, -15, 15)
        pg_x = int(np.round(goal[0] * 2) / 2 / self._action_map_res)
        pg_y = int(np.round(goal[1] * 2) / 2 / self._action_map_res)
        delta_x, delta_y = self.mapper.egocentric_to_allocentric(pg_x, pg_y, action_map_res=self._action_map_res)
        goal_x = int((x + delta_x) * self.mapper._gm_res / self._action_map_res)
        goal_y = int((y + delta_y) * self.mapper._gm_res / self._action_map_res)

        indices = np.indices(coarser_map.shape)
        distances = (np.linalg.norm(np.stack([(indices[0] - goal_y), (indices[1] - goal_x)]), axis=0) + 1) \
                        * (coarser_map * 100 + 1)

        # make it egocentric
        sorted_indices = np.unravel_index(np.argsort(distances, axis=None), distances.shape)
        i = 0
        while True:
            closest_index = np.array([sorted_indices[0][2 ^ i], sorted_indices[1][2 ^ i]])
            if logging.root.level == logging.DEBUG:
                assert self._map_index_to_graph_nodes([(closest_index[1] * 5, closest_index[0] * 5)])[0] in self._graph.nodes
            allocentric_delta_x = (closest_index[1] - agent_x) * self._action_map_res
            allocentric_delta_y = (closest_index[0] - agent_y) * self._action_map_res
            ego_x, ego_y = self.mapper.allocentric_to_egocentric(allocentric_delta_x, allocentric_delta_y)
            point = np.array([ego_x, ego_y])
            if self.check_navigability(point) or i >= 4:
                break
            else:
                i += 1

        noisy_point = point + (np.random.random(2) - 0.5) * (self._action_map_res - 1e-3)
        assert np.all(np.round(noisy_point * 2) / 2 == point)

        return noisy_point

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
        # TODO: apartment 3?
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

    def plot_top_down_map(self, geometric_map, x, y, delta_x, delta_y, observation, distribution=None, next_node=None,
                          next_node_idx=None, show=False):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.invert_yaxis()
        cmap = colors.ListedColormap(['white', 'grey'])
        ax.pcolor(geometric_map[:, :, 0] > 0.5, cmap=cmap, edgecolor='k')
        agent = patches.Rectangle((x + 0.25, y + 0.25), 0.5, 0.5, linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(agent)
        ax.arrow(x + 0.5, y + 0.5, np.cos(np.deg2rad(self.mapper.get_orientation())) * 0.75,
                  np.sin(np.deg2rad(self.mapper.get_orientation())) * 0.75, width=0.1)
        predicted_goal_patch = patches.Rectangle((x + delta_x + 0.25, y + delta_y + 0.25), 0.5, 0.5, linewidth=1,
                                                 edgecolor='green', facecolor='green')
        ax.add_patch(predicted_goal_patch)

        if torch.is_tensor(observation['pointgoal_with_gps_compass']):
            pointgoal = observation['pointgoal_with_gps_compass'].cpu().numpy()
        else:
            pointgoal = observation['pointgoal_with_gps_compass']
        gt_goal_x = int(np.round(pointgoal[1] / self._action_map_res))
        gt_goal_y = int(np.round(-pointgoal[0] / self._action_map_res))
        delta_gt_x, delta_gt_y = self.mapper.egocentric_to_allocentric(gt_goal_x, gt_goal_y,
                                                                       action_map_res=self._action_map_res)
        gt_goal_patch = patches.Rectangle((x + delta_gt_x + 0.25, y + delta_gt_y + 0.25),
                                          0.5, 0.5, linewidth=1, edgecolor='red', facecolor='red')
        ax.add_patch(gt_goal_patch)

        am = self.mapper.get_egocentric_acoustic_map()[:, :, 0]
        for i in range(am.shape[0]):
            for j in range(am.shape[1]):
                alpha = am[i][j] / am.max()
                am_x, am_y = self.mapper.egocentric_to_allocentric((j - am.shape[1] // 2), (i - am.shape[0] // 2))
                am_x += x
                am_y += y
                ax.add_patch(patches.Rectangle((am_x + 0.25, am_y + 0.25), 0.5, 0.5, linewidth=1, edgecolor='purple',
                                               facecolor='purple', alpha=alpha))

        # draw mean and std of bivariate gaussian distribution
        if distribution is not None:
            if distribution is not None:
                probs = distribution.probs.detach().cpu().numpy()[0]
                for i, prob in enumerate(probs):
                    point = self.get_map_coordinates(i, 9)
                    agent = patches.Rectangle((point[0] + 0.25, point[1]), 0.5, 0.5, linewidth=1, edgecolor='brown',
                                              facecolor='brown', alpha=prob * 100)
                    ax.add_patch(agent)

        if next_node is not None:
            next_node = patches.Rectangle((next_node_idx[0] + 0.25, next_node_idx[1] + 0.25),
                                          0.5, 0.5, linewidth=1, edgecolor='blue', facecolor='blue')
            ax.add_patch(next_node)

        plt.draw()
        if show:
            plt.show()
        os.makedirs(self._output_dir, exist_ok=True)
        plt.savefig(os.path.join(self._output_dir, '{}.png'.format(str(self._step_count))),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

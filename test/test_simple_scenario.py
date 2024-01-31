import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt

from agent import Agent
from astar import astar
from map import Map
from multi_agent_search import multi_agent_search
from utils import generate_map

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


def check_maps(
    width: int,
    height: int,
    agent_locations: List[Tuple[int, int]],
    means: List[float],
    centers: List[Tuple[int, int]],
) -> Map:
    """
    Plot the gaussian reward map for verification
    :param width: Width of the map
    :param height: Height of the map
    :param agent_locations: Start locations of the agents
    :param means: Means of the gaussian for different locations
    :param centers: Centers of the gaussian corresponding to the means
    """
    map = generate_map(width, height, agent_locations=agent_locations)
    im = plt.imshow(map.grid, cmap="hot")
    plt.colorbar(im)
    for i in range(height):
        for j in range(width):
            plt.text(j, i, map.grid[i, j], ha="center", va="center", color="b")
    # plt.show()
    return map


def test_astar(map: Map, init_pos: int, horizon: int):
    """
    Test the single-agent A* algorithm
    :param map: Map object to query for collisions and valid neighbors
    :param init_pos: Initial position of the agent
    :param horizon: Planning horizon
    """
    path = astar(map, init_pos, horizon)
    if path is None:
        return False

    path_coords = []
    for point in path:
        path_coords.append(map.get_coordinate(point))
    print("Path taken by the agent -> ", path_coords)
    return True


def test_vulcan(map: Map, init_pos: int, horizon: int):
    """
    Test the single-agent Vulcan algorithm
    :param map: Map object to query the underlying GP
    :param init_pos: Initial position of the agent
    :param horizon: Planning horizon
    """
    agent = Agent(0, init_pos, map, 10, 3, True)
    agent.adaptive_search()
    path = []
    for observation in agent.mdp_handle.observations:
        location = observation.location
        path.append(map.get_coordinate(location))

    if len(path) == 0:
        return False

    print("Path taken by the agent -> ", path)
    return True


def test_mastar(map: Map, init_poses: List[Tuple[int, int]], horizon: int):
    """
    Test the multi-agent A* algorithm when agents are within communication range
    :param map: Map object to query for collisions and valid neighbors along with underlying GP
    :param init_poses: Initial positions of the agents
    :param horizon: Planning horizon
    """
    paths, best_gain = multi_agent_search(map, init_poses, horizon)
    if paths is None:
        return False

    for path in paths:
        if path is None:
            return False

        path_coords = []
        for point in path:
            path_coords.append(map.get_coordinate(point))
        print("Path taken by the agent -> ", path_coords)
    print("Best Information Gain -> ", best_gain)
    return True


width = 5
height = 5
agent_locations = [(1, 1), (3, 3)]
means = [2.0, 1.0]
centers = [(0, 0), (4, 4)]
map = check_maps(width, height, agent_locations, means, centers)

single_agent_start_location = map.linearize_coordinate(
    agent_locations[0][0], agent_locations[0][1]
)
test_astar(map, single_agent_start_location, 3)
test_vulcan(map, single_agent_start_location, 3)
test_mastar(map, agent_locations, 3)

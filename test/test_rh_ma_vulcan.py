import os
import sys
import numpy as np
from copy import deepcopy
from cProfile import Profile
from typing import List, Tuple
from numpy.typing import NDArray
from pstats import SortKey, Stats
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from map import Map, Parameters  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA


def visualize_path(paths: List[List[NDArray[np.int64]]], map: Map):
    fig, ax = plt.subplots()
    map_limits = [0, map.num_of_rows, 0, map.num_of_cols]
    ax.set_xlim(map_limits[0], map_limits[1])
    ax.set_ylim(map_limits[2], map_limits[3])

    agent_colors = "rbgkymc"
    num_of_agents = len(paths)

    lines = []
    for agent in range(num_of_agents):
        (line,) = ax.plot([], [], lw=2, color=agent_colors[agent], alpha=0.7)
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for i, path in enumerate(paths):
            x_data = [point[0] for point in path[: frame + 1]]
            y_data = [point[1] for point in path[: frame + 1]]
            lines[i].set_data(x_data, y_data)
        return lines

    frames = max(len(path) for path in paths)
    _ = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

    plt.imshow(map.grid, cmap="hot")
    plt.show()


if __name__ == "__main__":
    figures_base_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/../figures/testing/"
    )

    parser = ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="single-small",
        choices=["single-small", "single-large", "multi-small", "multi-large"],
        help="Type(phenomenon-map_size) of Receding-Horizon Multi-Agent Vulcan testing to perform",
    )

    parser.add_argument(
        "--distance_simplification",
        type=bool,
        default=True,
        help="Whether to use the distance simplification",
    )

    parser.add_argument(
        "--measurement_noise",
        type=float,
        default=0.2,
        help="Measurement noise to use",
    )

    args = parser.parse_args()

    params = Parameters(
        theta_1=np.float64(0.4),
        theta_2=np.float64(0.01),
        u_tilde=np.float64(1.4),
        P_1=np.float64(0.98),
        P_2=np.float64(0.002),
        J=np.int64(5),
        measurement_noise=np.float64(args.measurement_noise),
        distance_simplification=args.distance_simplification,
    )

    agent_colors = ["r", "b"]
    agent_locations: List[Tuple[int, int]] = [(0, 0)]

    if args.type == "single-small":
        agent_locations.append((4, 4))
        map = generate_map(5, 5, agent_locations=agent_locations, parameters=params)
    elif args.type == "single-large":
        agent_locations.append((10, 10))
        map = generate_map(11, 11, agent_locations=agent_locations, parameters=params)
    elif args.type == "multi-small":
        agent_locations.append((4, 4))
        map = generate_map(
            5,
            5,
            agent_locations=agent_locations,
            gp_means=[1, 1],
            gp_locations=[(1, 1), (4, 4)],
            parameters=params,
        )
    elif args.type == "multi-large":
        agent_locations.append((10, 10))
        map = generate_map(
            11,
            11,
            agent_locations=agent_locations,
            gp_means=[1, 1, 1, 1, 1],
            gp_locations=[(1, 1), (8, 2), (5, 5), (2, 8), (10, 10)],
            parameters=params,
        )
    else:
        raise ValueError("Invalid type")

    if "small" in args.type:
        mission_duration = 10
        communication_range = 2
    else:
        mission_duration = 35
        communication_range = 5

    vulcan_agents = []
    vulcan_map = deepcopy(map)

    for agent in range(len(agent_locations)):
        agent_location_linearized = vulcan_map.linearize_coordinate(
            agent_locations[agent][0], agent_locations[agent][1]
        )
        vulcan_agent = Agent(
            id=agent,
            start_location=agent_location_linearized,
            map=vulcan_map,
            mission_duration=mission_duration,
        )
        vulcan_agents.append(vulcan_agent)

    rh_ma_vulcan = MultiAgentVulcan(vulcan_map, vulcan_agents, communication_range)

    with Profile() as prof:
        print(f"{rh_ma_vulcan.planner()}")
        (Stats(prof).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())

    print("Vulcan Agents Path")
    vulcan_agents_paths = []
    for idx, agent in enumerate(vulcan_agents):
        print("Path for agent ", agent.id)
        vulcan_path = []
        for v_location in agent.visited_locations:
            vulcan_path.append(map.get_coordinate(v_location))
        print(vulcan_path)
        plt.plot(
            [x[1] for x in vulcan_path],
            [x[0] for x in vulcan_path],
            agent_colors[idx] + "--",
            alpha=0.7,
        )
        vulcan_agents_paths.append(vulcan_path)

    plt.imshow(map.grid, cmap="hot")
    plt.savefig(figures_base_path + "rh-ma-vulcan-" + args.type + ".png")

    visualize_path(vulcan_agents_paths, map)

    vulcan_map = deepcopy(map)
    vulcan_agents = []
    for agent in range(len(agent_locations)):
        agent_location_linearized = vulcan_map.linearize_coordinate(
            agent_locations[agent][0], agent_locations[agent][1]
        )
        vulcan_agent = Agent(
            id=agent,
            start_location=agent_location_linearized,
            map=vulcan_map,
            mission_duration=mission_duration,
        )
        vulcan_agents.append(vulcan_agent)

    vulcan_agents_paths = []
    for idx, agent in enumerate(vulcan_agents):
        agent.adaptive_search()
        vulcan_path = []
        for v_location in agent.visited_locations:
            vulcan_path.append(map.get_coordinate(v_location))
        print("Vulcan Path")
        print(vulcan_path)

        plt.plot(
            [x[1] for x in vulcan_path],
            [x[0] for x in vulcan_path],
            agent_colors[idx] + "--",
            alpha=0.7,
        )
        vulcan_agents_paths.append(vulcan_path)

    plt.imshow(map.grid, cmap="hot")
    plt.savefig(figures_base_path + "sa-vulcan-" + args.type + ".png")

    visualize_path(vulcan_agents_paths, map)

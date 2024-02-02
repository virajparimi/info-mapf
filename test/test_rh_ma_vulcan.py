import os
import sys
import numpy as np
from copy import deepcopy
from cProfile import Profile
from typing import List, Tuple
from pstats import SortKey, Stats
from argparse import ArgumentParser
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from map import Map, Parameters  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA

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
        for observation in agent.mdp_handle.observations:
            location = observation.location
            vulcan_path.append(map.get_coordinate(location))
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

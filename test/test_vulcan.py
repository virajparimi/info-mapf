import os
import sys
import numpy as np
from copy import deepcopy
from typing import Union, List
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from cProfile import Profile
from pstats import Stats, SortKey

sys.path.append("/home/mers/Desktop/Github/Multi-Agent_Path_Finding/info-mapf/src")

from agent import Agent  # NOQA
from map import Map, Parameters  # NOQA


def generate_map(
    rows: int,
    columns: int,
    agent_locations: Union[List[List[int]], None] = None,
    gp_means: Union[List[int], None] = None,
    gp_locations: Union[List[List[int]], None] = None,
    parameters: Union[Parameters, None] = None,
) -> Map:
    """
    Generates a random maze of size rows x columns
    :param rows: Number of rows
    :param columns: Number of columns
    :return: Map object
    """
    maze = np.ones((rows, columns), dtype=np.bool8)

    if agent_locations is None:
        center = [0, 0]
        agent_locations = [center]

    for agent_location in agent_locations:
        maze[agent_location[0], agent_location[1]] = False

    if gp_means is None:
        gp_means = [1]
    if gp_locations is None:
        gp_locations = [[rows // 2, columns // 2]]

    map = Map(maze, means=gp_means, locations=gp_locations, params=parameters)
    return map


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
        help="Type of Vulcan testing to perform",
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

    if args.type == "single-small":
        map = generate_map(5, 5, parameters=params)
    elif args.type == "single-large":
        map = generate_map(11, 11, parameters=params)
    elif args.type == "multi-small":
        map = generate_map(
            5, 5, gp_means=[1, 1], gp_locations=[[1, 1], [4, 4]], parameters=params
        )
    elif args.type == "multi-large":
        map = generate_map(
            11, 11, gp_means=[1, 1], gp_locations=[[1, 1], [10, 10]], parameters=params
        )
    else:
        raise ValueError("Invalid type")

    if "small" in args.type:
        mission_duration = 10
    else:
        mission_duration = 35

    vulcan_map = deepcopy(map)
    vulcan_agent = Agent(
        id=1, start_location=0, map=vulcan_map, mission_duration=mission_duration
    )

    with Profile() as prof:
        print(f"{vulcan_agent.adaptive_search()}")
        (Stats(prof).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())
    vulcan_path = []
    for observation in vulcan_agent.mdp_handle.observations:
        location = observation.location
        vulcan_path.append(map.get_coordinate(location))
    print("Vulcan Path")
    print(vulcan_path)

    plt.plot([x[1] for x in vulcan_path], [x[0] for x in vulcan_path], "r--", alpha=0.7)
    plt.imshow(map.grid)
    plt.savefig(figures_base_path + "vulcan-" + args.type + ".png")

    plt.clf()

    conventional_map = deepcopy(map)
    conventional_agent = Agent(
        id=1, start_location=0, map=conventional_map, mission_duration=mission_duration
    )
    with Profile() as prof:
        print(f"{conventional_agent.adaptive_search()}")
        (Stats(prof).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())
    conventional_path = []
    for observation in conventional_agent.mdp_handle.observations:
        location = observation.location
        conventional_path.append(map.get_coordinate(location))

    print("Conventional Path")
    print(conventional_path)
    plt.plot(
        [x[1] for x in conventional_path],
        [x[0] for x in conventional_path],
        "b--",
        alpha=0.7,
    )
    plt.imshow(map.grid)
    plt.savefig(figures_base_path + "conventional-" + args.type + ".png")

import os
import sys
from argparse import ArgumentParser
from copy import deepcopy
from cProfile import Profile
from pstats import SortKey, Stats

import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from map import Grid, RewardMap, Parameters  # NOQA

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
        grid, reward_map = generate_map(5, 5, parameters=params)
    elif args.type == "single-large":
        grid, reward_map = generate_map(11, 11, parameters=params)
    elif args.type == "multi-small":
        grid, reward_map = generate_map(
            5, 5, gp_means=[1, 1], gp_locations=[(1, 1), (4, 4)], parameters=params
        )
    elif args.type == "multi-large":
        grid, reward_map = generate_map(
            11,
            11,
            gp_means=[1, 1, 1, 1, 1],
            gp_locations=[(1, 1), (8, 2), (5, 5), (2, 8), (10, 10)],
            parameters=params,
        )
    else:
        raise ValueError("Invalid type")

    mission_duration = 10 if "small" in args.type else 35

    vulcan_grid = deepcopy(grid)
    vulcan_agent = Agent(
        id=1,
        start_location=0,
        grid=vulcan_grid,
        reward_map=reward_map,
        mission_duration=mission_duration,
    )

    with Profile() as prof:
        print(f"{vulcan_agent.adaptive_search()}")
        (Stats(prof).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())
    vulcan_path = []
    for observation in vulcan_agent.mdp_handle.observations:
        location = observation.location
        vulcan_path.append(vulcan_grid.get_coordinate(location))
    print("Vulcan Path")
    print(vulcan_path)

    plt.plot([x[1] for x in vulcan_path], [x[0] for x in vulcan_path], "r--", alpha=0.7)
    plt.imshow(reward_map.reward_map, cmap="hot")
    plt.savefig(figures_base_path + "vulcan-" + args.type + ".png")

    plt.clf()

    conventional_grid = deepcopy(grid)
    conventional_agent = Agent(
        id=1,
        start_location=0,
        grid=conventional_grid,
        reward_map=reward_map,
        mission_duration=mission_duration,
        use_vulcan=False,
    )
    with Profile() as prof:
        print(f"{conventional_agent.adaptive_search()}")
        (Stats(prof).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())
    conventional_path = []
    for observation in conventional_agent.mdp_handle.observations:
        location = observation.location
        conventional_path.append(conventional_grid.get_coordinate(location))

    print("Conventional Path")
    print(conventional_path)
    plt.plot(
        [x[1] for x in conventional_path],
        [x[0] for x in conventional_path],
        "b--",
        alpha=0.7,
    )
    plt.imshow(reward_map.reward_map, cmap="hot")
    plt.savefig(figures_base_path + "conventional-" + args.type + ".png")

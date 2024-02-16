import os
import sys
import logging
import numpy as np
from copy import deepcopy
from cProfile import Profile
from numpy.typing import NDArray
from pstats import SortKey, Stats
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from typing import List, Tuple, Any, Union
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA
from rh_sa_vulcan import SingleAgentVulcan  # NOQA
from map import Grid, RewardMap, Parameters  # NOQA


def visualize_path(
    paths: List[List[NDArray[np.int64]]],
    reward_map: RewardMap,
    filename: str,
    map_viz: Union[List[NDArray[Any]], None] = None,
    save_fig: bool = False,
):
    fig, ax = plt.subplots()
    if map_viz is not None:
        ax.imshow(
            map_viz[0],
            extent=(-1, reward_map.num_of_rows + 1, reward_map.num_of_cols + 1, -1),
            cmap="hot",
        )
    else:
        ax.imshow(reward_map.reward_map, cmap="hot")

    agent_colors = "gbrkymc"
    num_of_agents = len(paths)

    lines = []
    for agent in range(num_of_agents):
        (line,) = ax.plot([], [], lw=2, color=agent_colors[agent], ls="--", alpha=0.7)
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for i, path in enumerate(paths):
            x_data = [point[1] for point in path[: frame + 1]]
            y_data = [point[0] for point in path[: frame + 1]]
            lines[i].set_data(x_data, y_data)
        return lines

    frames = max(len(path) for path in paths)
    animation = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    if save_fig:
        animation.save(filename, writer="imagemagick", fps=1)

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

    parser.add_argument(
        "--save_figures", type=bool, default=False, help="Whether to save the plots"
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="info",
        choices=["info", "debug"],
        help="Logging level to use",
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

    logging.basicConfig()
    if args.logging_level == "debug":
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    agent_colors = "gbrkymc"
    agent_locations: List[Tuple[int, int]] = [(0, 0)]

    if args.type == "single-small":
        agent_locations.append((4, 4))
        grid, reward_map = generate_map(
            5, 5, agent_locations=agent_locations, parameters=params
        )
    elif args.type == "single-large":
        agent_locations.append((10, 10))
        grid, reward_map = generate_map(
            11, 11, agent_locations=agent_locations, parameters=params
        )
    elif args.type == "multi-small":
        agent_locations.append((4, 4))
        grid, reward_map = generate_map(
            5,
            5,
            agent_locations=agent_locations,
            gp_means=[1, 1],
            gp_locations=[(1, 1), (4, 4)],
            parameters=params,
        )
    elif args.type == "multi-large":
        agent_locations.append((10, 10))
        agent_locations.append((5, 5))
        grid, reward_map = generate_map(
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
        communication_range = 3
    else:
        mission_duration = 35
        communication_range = 5

    x = np.linspace(-1, reward_map.num_of_rows + 1, 1000)
    y = np.linspace(-1, reward_map.num_of_cols + 1, 1000)
    xx, yy = np.meshgrid(x, y)
    meshgrid = np.dstack((xx, yy))
    zz = np.zeros_like(xx)
    for i in range(len(reward_map.locations)):
        linear_location = reward_map.locations[i]
        location_coord = reward_map.get_coordinate(linear_location)
        gaussian = reward_map.means[i] * multivariate_normal.pdf(
            meshgrid, mean=location_coord, cov=1
        )
        zz += gaussian
    zz /= np.max(zz)

    vulcan_agents = []
    vulcan_grid = deepcopy(grid)
    for agent in range(len(agent_locations)):
        agent_location_linearized = vulcan_grid.linearize_coordinate(
            agent_locations[agent][0], agent_locations[agent][1]
        )
        vulcan_agent = Agent(
            id=agent,
            start_location=agent_location_linearized,
            grid=vulcan_grid,
            reward_map=reward_map,
            mission_duration=mission_duration,
        )
        vulcan_agents.append(vulcan_agent)

    rh_ma_vulcan = MultiAgentVulcan(
        grid=vulcan_grid,
        reward_map=reward_map,
        agents=vulcan_agents,
        communication_range=communication_range,
    )

    with Profile() as prof:
        print(f"{rh_ma_vulcan.planner()}")
        (Stats(prof).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())

    print("Vulcan Agents Path")
    vulcan_agents_paths = []
    for idx, agent in enumerate(vulcan_agents):
        print("Path for agent ", agent.id)
        vulcan_path = []
        for v_location in agent.visited_locations:
            vulcan_path.append(vulcan_grid.get_coordinate(v_location))
        print(vulcan_path)
        plt.plot(
            [x[1] for x in vulcan_path],
            [x[0] for x in vulcan_path],
            agent_colors[idx] + "--",
            alpha=0.7,
        )
        vulcan_agents_paths.append(vulcan_path)

    plt.imshow(
        zz,
        extent=(-1, reward_map.num_of_rows + 1, reward_map.num_of_cols + 1, -1),
        cmap="hot",
    )
    if args.save_figures:
        plt.savefig(figures_base_path + "rh-ma-vulcan-" + args.type + ".png")

    visualize_path(
        vulcan_agents_paths,
        reward_map,
        figures_base_path + "rh-ma-vulcan-" + args.type + ".gif",
        [zz],
        save_fig=args.save_figures,
    )

    # Simple ablative test with only single agent vulcan without collision avoidance

    vulcan_grid = deepcopy(grid)
    vulcan_agents = []
    for agent in range(len(agent_locations)):
        agent_location_linearized = vulcan_grid.linearize_coordinate(
            agent_locations[agent][0], agent_locations[agent][1]
        )
        vulcan_agent = Agent(
            id=agent,
            start_location=agent_location_linearized,
            grid=vulcan_grid,
            reward_map=reward_map,
            mission_duration=mission_duration,
        )
        vulcan_agents.append(vulcan_agent)

    vulcan_agents_paths = []
    for idx, agent in enumerate(vulcan_agents):
        agent.adaptive_search()
        vulcan_path = []
        for v_location in agent.visited_locations:
            vulcan_path.append(vulcan_grid.get_coordinate(v_location))
        print("Vulcan Path")
        print(vulcan_path)

        plt.plot(
            [x[1] for x in vulcan_path],
            [x[0] for x in vulcan_path],
            agent_colors[idx] + "--",
            alpha=0.7,
        )
        vulcan_agents_paths.append(vulcan_path)

    plt.imshow(
        zz,
        extent=(-1, reward_map.num_of_rows + 1, reward_map.num_of_cols + 1, -1),
        cmap="hot",
    )
    if args.save_figures:
        plt.savefig(figures_base_path + "sa-vulcan-" + args.type + ".png")

    visualize_path(
        vulcan_agents_paths,
        reward_map,
        figures_base_path + "sa-vulcan-" + args.type + ".gif",
        [zz],
        save_fig=args.save_figures,
    )

    # Simple ablative test with only single agent vulcan with collision avoidance

    vulcan_grid = deepcopy(grid)
    vulcan_agents = []
    for agent in range(len(agent_locations)):
        agent_location_linearized = vulcan_grid.linearize_coordinate(
            agent_locations[agent][0], agent_locations[agent][1]
        )
        vulcan_agent = Agent(
            id=agent,
            start_location=agent_location_linearized,
            grid=vulcan_grid,
            reward_map=reward_map,
            mission_duration=mission_duration,
        )
        vulcan_agents.append(vulcan_agent)

    rh_sa_vulcan = SingleAgentVulcan(
        grid=vulcan_grid,
        reward_map=reward_map,
        agents=vulcan_agents,
    )

    with Profile() as prof:
        print(f"{rh_sa_vulcan.planner()}")
        (Stats(prof).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())

    vulcan_agents_paths = []
    for idx, agent in enumerate(vulcan_agents):
        vulcan_path = []
        for v_location in agent.visited_locations:
            vulcan_path.append(vulcan_grid.get_coordinate(v_location))
        print("Vulcan Path")
        print(vulcan_path)

        plt.plot(
            [x[1] for x in vulcan_path],
            [x[0] for x in vulcan_path],
            agent_colors[idx] + "--",
            alpha=0.7,
        )
        vulcan_agents_paths.append(vulcan_path)

    plt.imshow(
        zz,
        extent=(-1, reward_map.num_of_rows + 1, reward_map.num_of_cols + 1, -1),
        cmap="hot",
    )
    if args.save_figures:
        plt.savefig(figures_base_path + "sa-ca-vulcan-" + args.type + ".png")

    visualize_path(
        vulcan_agents_paths,
        reward_map,
        figures_base_path + "sa-ca-vulcan-" + args.type + ".gif",
        [zz],
        save_fig=args.save_figures,
    )

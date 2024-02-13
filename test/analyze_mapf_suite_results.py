import os
import sys
import pickle
import logging
import numpy as np
from numpy.typing import NDArray
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from typing import List, Tuple, Any, Union
from matplotlib.animation import FuncAnimation
from test_mapf_suite import Statistics, SampleStats, VulcanStats  # NOQA

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from map import Grid, RewardMap, Parameters  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA


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
            zz,
            extent=(-1, reward_map.num_of_rows, reward_map.num_of_cols, -1),
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
    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, repeat=False
    )

    if save_fig:
        ani.save(filename, writer="imagemagick", fps=1)
    else:
        plt.show()


def validate_paths(
    agent_i: Tuple[int, List[NDArray[np.int64]]],
    agent_j: Tuple[int, List[NDArray[np.int64]]],
    mission_duration: int,
    multi: bool = True,
) -> int:

    string = "Multi-agent" if multi else "Single-agent"

    agent_i_id, path_i = agent_i
    agent_j_id, path_j = agent_j
    assert len(path_i) == len(path_j)
    for step in range(mission_duration):
        agent_i_location = path_i[step]
        agent_j_location = path_j[step]

        # Vertex collision
        if np.array_equal(agent_i_location, agent_j_location):
            logging.error(
                f"{string} collision detected at step {step} between agents {agent_i_id} and {agent_j_id}"
            )
            return step
        elif step < mission_duration - 1:
            agent_i_next_location = path_i[step + 1]
            agent_j_next_location = path_j[step + 1]

            # Edge collision
            if np.array_equal(
                agent_i_location, agent_j_next_location
            ) and np.array_equal(agent_i_next_location, agent_j_location):
                logging.error(
                    f"{string} edge collision detected at step {step} between agents {agent_i_id} and {agent_j_id}"
                )
                return step
    return mission_duration


if __name__ == "__main__":
    results_base_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/testing/"

    parser = ArgumentParser()
    parser.add_argument(
        "--results_pkl",
        type=str,
        required=True,
        help="Name of the results to analyze",
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="info",
        choices=["info", "debug"],
        help="Logging level to use",
    )

    args = parser.parse_args()

    logging.basicConfig()
    if args.logging_level == "debug":
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    statistics = None
    with open(results_base_path + args.results_pkl, "rb") as f:
        statistics = pickle.load(f)

    num_samples = len(statistics.stats)

    multi_agent_steps, single_agent_steps = [], []
    multi_agent_phenomenons_discovered, single_agent_phenomenons_discovered = (
        [],
        [],
    )

    mission_duration = statistics.mission_duration
    for sample in range(num_samples):

        gp_locations = statistics.stats[sample].gp_locations
        agent_locations = statistics.stats[sample].agent_locations

        sample_multi_agent_phenomenons_discovered = set()
        sample_single_agent_phenomenons_discovered = set()

        # Verify that the paths of agents do not lead to collisions!
        multi_agent_last_valid_step, single_agent_last_valid_step = (
            mission_duration,
            mission_duration,
        )

        multi_collision, single_collision = False, False
        for agent_i in range(len(agent_locations)):
            for agent_j in range(len(agent_locations)):
                if agent_i == agent_j:
                    continue
                agent_i_path = statistics.stats[sample].multi_agent_stats[agent_i].path
                agent_j_path = statistics.stats[sample].multi_agent_stats[agent_j].path

                multi_agent_paths_valid_steps = (
                    validate_paths(
                        (
                            agent_i,
                            statistics.stats[sample].multi_agent_stats[agent_i].path,
                        ),
                        (
                            agent_j,
                            statistics.stats[sample].multi_agent_stats[agent_j].path,
                        ),
                        mission_duration,
                    ),
                )

                multi_collision = (
                    True
                    if multi_agent_paths_valid_steps[0] < mission_duration
                    else False
                )
                multi_agent_last_valid_step = min(
                    multi_agent_last_valid_step, multi_agent_paths_valid_steps[0]
                )

                single_agent_paths_valid_steps = (
                    validate_paths(
                        (
                            agent_i,
                            statistics.stats[sample].single_agent_stats[agent_i].path,
                        ),
                        (
                            agent_j,
                            statistics.stats[sample].single_agent_stats[agent_j].path,
                        ),
                        mission_duration,
                        False,
                    ),
                )
                single_collision = (
                    True
                    if single_agent_paths_valid_steps[0] < mission_duration
                    else False
                )
                single_agent_last_valid_step = min(
                    single_agent_last_valid_step, single_agent_paths_valid_steps[0]
                )

        multi_last_gp_found_step, single_last_gp_found_step = 0, 0
        for step in range(multi_agent_last_valid_step):
            for agent in range(len(agent_locations)):
                coord = statistics.stats[sample].multi_agent_stats[agent].path[step]
                coord_compare = (coord[0], coord[1])
                if (
                    coord_compare in gp_locations
                    and coord_compare not in sample_multi_agent_phenomenons_discovered
                ):
                    multi_last_gp_found_step = step
                    sample_multi_agent_phenomenons_discovered.add(coord_compare)

        for step in range(single_agent_last_valid_step):
            for agent in range(len(agent_locations)):
                coord = statistics.stats[sample].single_agent_stats[agent].path[step]
                coord_compare = (coord[0], coord[1])
                if (
                    coord_compare in gp_locations
                    and coord_compare not in sample_single_agent_phenomenons_discovered
                ):
                    single_last_gp_found_step = step
                    sample_single_agent_phenomenons_discovered.add(coord_compare)

        multi_agent_phenomenons_discovered.append(
            len(sample_multi_agent_phenomenons_discovered)
        )
        single_agent_phenomenons_discovered.append(
            len(sample_single_agent_phenomenons_discovered)
        )

        if (
            len(sample_multi_agent_phenomenons_discovered) != len(gp_locations)
            or multi_collision
        ):
            multi_agent_steps.append(mission_duration)
        else:
            multi_agent_steps.append(multi_last_gp_found_step)

        if (
            len(sample_single_agent_phenomenons_discovered) != len(gp_locations)
            or single_collision
        ):
            single_agent_steps.append(mission_duration)
        else:
            single_agent_steps.append(single_last_gp_found_step)

    # Compile the results and print them
    multi_agent_steps = np.array(multi_agent_steps)
    single_agent_steps = np.array(single_agent_steps)
    multi_agent_phenomenons_discovered = np.array(multi_agent_phenomenons_discovered)
    single_agent_phenomenons_discovered = np.array(single_agent_phenomenons_discovered)

    print(
        f"Single-agent steps: {np.mean(single_agent_steps)} +/- {np.std(single_agent_steps)}"
    )
    print(
        f"Multi-agent steps: {np.mean(multi_agent_steps)} +/- {np.std(multi_agent_steps)}"
    )
    print(
        "Single-agent phenomenons discovered: "
        f"{np.mean(single_agent_phenomenons_discovered)} +/- {np.std(single_agent_phenomenons_discovered)}"
    )
    print(
        "Multi-agent phenomenons discovered: "
        f"{np.mean(multi_agent_phenomenons_discovered)} +/- {np.std(multi_agent_phenomenons_discovered)}"
    )

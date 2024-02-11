import os
import sys
import pickle
import logging
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray
from argparse import ArgumentParser
from test_mapf_suite import Statistics, SampleStats, VulcanStats  # NOQA

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from map import Grid, RewardMap, Parameters  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA


def validate_paths(
    agent_i: Tuple[int, List[NDArray[np.int64]]],
    agent_j: Tuple[int, List[NDArray[np.int64]]],
    mission_duration: int,
) -> int:

    agent_i_id, path_i = agent_i
    agent_j_id, path_j = agent_j
    assert len(path_i) == len(path_j)
    for step in range(mission_duration):
        agent_i_location = path_i[step]
        agent_j_location = path_j[step]

        # Vertex collision
        if agent_i_location == agent_j_location:
            logging.error(
                f"Multi-agent collision detected at step {step} between agents {agent_i_id} and {agent_j_id}"
            )
            return step
        elif step < mission_duration - 1:
            agent_i_next_location = path_i[step + 1]
            agent_j_next_location = path_j[step + 1]

            # Edge collision
            if (
                agent_i_location == agent_j_next_location
                and agent_i_next_location == agent_j_location
            ):
                logging.error(
                    f"Multi-agent edge collision detected at step {step} between agents {agent_i_id} and {agent_j_id}"
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

    for sample in range(num_samples):

        gp_locations = statistics.stats[sample].gp_locations
        mission_duration = statistics.stats[sample].mission_duration
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
                    True if multi_agent_paths_valid_steps < mission_duration else False
                )
                multi_agent_last_valid_step = min(
                    multi_agent_last_valid_step, multi_agent_paths_valid_steps
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
                    ),
                )
                single_collision = (
                    True if single_agent_paths_valid_steps < mission_duration else False
                )
                single_agent_last_valid_step = min(
                    single_agent_last_valid_step, single_agent_paths_valid_steps
                )

        for step in range(multi_agent_last_valid_step):
            for agent in range(len(agent_locations)):
                sample_multi_agent_phenomenons_discovered |= (
                    statistics.stats[sample]
                    .multi_agent_stats[agent]
                    .phenomenons_discovered
                )

        for step in range(single_agent_last_valid_step):
            for agent in range(len(agent_locations)):
                sample_single_agent_phenomenons_discovered |= (
                    statistics.stats[sample]
                    .single_agent_stats[agent]
                    .phenomenons_discovered
                )

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
            multi_agent_steps.append(multi_agent_last_valid_step)

        if (
            len(sample_single_agent_phenomenons_discovered) != len(gp_locations)
            or single_collision
        ):
            single_agent_steps.append(mission_duration)
        else:
            single_agent_steps.append(single_agent_last_valid_step)

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

import os
import sys
import pickle
import logging
import numpy as np
from numpy.typing import NDArray
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Any, Union, Set
from test_mapf_suite import (  # NOQA
    Statistics,
    AugmentedStatistics,
    SampleStats,
    AugmentedSampleStats,
    VulcanStats,
    load_mapf_map,
)  # NOQA

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA
from map import Grid, RewardMap, Parameters, ActionType  # NOQA


def get_row_coordinate(location_id: int, num_of_cols: int) -> int:
    return location_id // num_of_cols


def get_column_coordinate(location_id: int, num_of_cols: int) -> int:
    return location_id % num_of_cols


def get_coordinate(location_id: int, num_of_cols: int) -> NDArray[np.int64]:
    return np.array(
        [
            get_row_coordinate(location_id, num_of_cols),
            get_column_coordinate(location_id, num_of_cols),
        ]
    )


def linearize_coordinate(row: int, column: int, num_of_cols: int) -> int:
    return num_of_cols * row + column


def get_manhattan_distance(
    location_id_a: int, location_id_b: int, num_of_cols: int
) -> int:
    location_a = get_coordinate(location_id_a, num_of_cols)
    location_b = get_coordinate(location_id_b, num_of_cols)
    return np.sum(np.abs(location_a - location_b))


def find_minimal_disjoint_sets(agent_bubbles: List[Set[int]]) -> List[Set[int]]:
    minimal_disjoint_sets = []

    for bubble in agent_bubbles:
        intersecting_sets = []
        for idx, existing_set in enumerate(minimal_disjoint_sets):
            if set(bubble).intersection(existing_set):
                intersecting_sets.append(idx)

        if not intersecting_sets:
            minimal_disjoint_sets.append(set(bubble))
        else:
            merged_set = set(bubble)
            for idx in sorted(intersecting_sets, reverse=True):
                merged_set |= minimal_disjoint_sets.pop(idx)
            minimal_disjoint_sets.append(merged_set)

    return minimal_disjoint_sets


def within_range_agents(
    num_agents: int,
    agent_locations: List[int],
    num_of_cols: int,
    communication_range: int,
) -> List[int]:
    agent_bubbles = [[i] for i in range(num_agents)]
    for agent_i in range(num_agents):
        for agent_j in range(num_agents):
            if agent_i == agent_j:
                continue
            agent_i_location = agent_locations[agent_i]
            agent_j_location = agent_locations[agent_j]
            if (
                get_manhattan_distance(agent_i_location, agent_j_location, num_of_cols)
                < communication_range
            ):
                agent_bubbles[agent_i].append(agent_j)

        agent_bubbles[agent_i] = sorted(agent_bubbles[agent_i])

    modified_agent_bubbles = [set(bubble) for bubble in agent_bubbles]
    minimal_disjoint_sets = find_minimal_disjoint_sets(modified_agent_bubbles)
    mapped_agent_bubbles = {}
    for idx, bubble in enumerate(minimal_disjoint_sets):
        list_bubble = list(bubble)
        min_id = min([agent for agent in list_bubble])
        mapped_agent_bubbles[min_id] = list_bubble

    return [len(value) for key, value in mapped_agent_bubbles.items()]


def analyse_results(
    sample_to_visualize: int,
    statistics: Statistics,
    agent_locations: List[Tuple[int, int]],
    reward_map: RewardMap,
    agent_colors: List[str],
    map_viz: List[NDArray[Any]],
    filename: str,
    type_of_analysis: str = "multi",
):

    vulcan_agents_paths = []
    for agent_idx in range(len(agent_locations)):

        if type_of_analysis == "multi":
            vulcan_path = (
                statistics.stats[sample_to_visualize].multi_agent_stats[agent_idx].path
            )
        elif type_of_analysis == "single":
            vulcan_path = (
                statistics.stats[sample_to_visualize].single_agent_stats[agent_idx].path
            )
        else:
            vulcan_path = (
                statistics.stats[sample_to_visualize]
                .single_agent_collision_avoidance_stats[agent_idx]
                .path
            )

        plt.plot(
            [x[1] for x in vulcan_path],
            [x[0] for x in vulcan_path],
            color=agent_colors[agent_idx],
            linestyle="--",
            alpha=0.7,
        )
        plt.plot(
            agent_locations[agent_idx][1],
            agent_locations[agent_idx][0],
            color=agent_colors[agent_idx],
            marker="x",
        )
        vulcan_agents_paths.append(vulcan_path)

    plt.imshow(
        map_viz[0],
        extent=(0, reward_map.num_of_cols, reward_map.num_of_rows, 0),
        cmap="hot",
        alpha=0.7,
    )
    if len(map_viz) > 1:
        plt.imshow(
            1.0 - map_viz[1],
            extent=(0, reward_map.num_of_cols, reward_map.num_of_rows, 0),
            cmap="binary",
            alpha=0.5,
        )

    plt.savefig(filename + ".png")
    plt.clf()

    visualize_path(
        vulcan_agents_paths,
        reward_map,
        filename + ".gif",
        map_viz,
        save_fig=True,
    )
    plt.clf()


def visualize_path(
    paths: List[List[NDArray[np.int64]]],
    reward_map: RewardMap,
    filename: str,
    map_viz: List[NDArray[Any]],
    save_fig: bool = False,
):
    fig, ax = plt.subplots()
    if len(map_viz) == 1:
        ax.imshow(
            map_viz[0],
            extent=(0, reward_map.num_of_cols, reward_map.num_of_rows, 0),
            cmap="hot",
        )
    else:
        ax.imshow(
            map_viz[0],
            extent=(0, reward_map.num_of_cols, reward_map.num_of_rows, 0),
            cmap="hot",
            alpha=0.7,
        )
        ax.imshow(
            1.0 - map_viz[1],
            extent=(0, reward_map.num_of_cols, reward_map.num_of_rows, 0),
            cmap="binary",
            alpha=0.5,
        )

    agent_colors = ["g", "b", "r", "deeppink", "y", "m", "c", "w"]
    num_of_agents = len(paths)

    lines = []
    starts = []
    for agent in range(num_of_agents):
        (line,) = ax.plot([], [], lw=2, color=agent_colors[agent], ls="--", alpha=0.7)
        (start,) = ax.plot(
            [], [], lw=2, color=agent_colors[agent], marker="x", alpha=0.7
        )
        lines.append(line)
        starts.append(start)

    def init():
        for i, path in enumerate(paths):
            starts[i].set_data(path[0][1], path[0][0])
        return starts

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
    maze: Union[NDArray[np.bool_], None] = None,
    multi: bool = True,
    single_ca: bool = False,
) -> int:

    string = "Multi-agent" if multi else "Single-agent"
    string = "Single-agent with collision avoidance" if single_ca else string

    agent_i_id, path_i = agent_i
    agent_j_id, path_j = agent_j
    assert len(path_i) == len(path_j)
    for step in range(mission_duration):
        agent_i_location = path_i[step]
        agent_j_location = path_j[step]

        # Map collision
        if maze is not None:
            if not maze[agent_i_location[0], agent_i_location[1]]:
                logging.error(
                    f"{string} collision detected at step {step} on location "
                    f"({agent_i_location[0]} , {agent_i_location[1]}) for agent {agent_i_id} with map"
                )
                return step
            if not maze[agent_j_location[0], agent_j_location[1]]:
                logging.error(
                    f"{string} collision detected at step {step} on location "
                    f"({agent_j_location[0], agent_j_location[1]}) for agent {agent_j_id} with map"
                )
                return step

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
    results_base_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/../data/all_observed_set/"
    )
    store_base_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/../data/map_based_results/"
    )
    figures_base_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/../figures/testing/"
    )

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

    maze = None
    if "maze" in args.results_pkl:
        maze = load_mapf_map(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../data/maps/maze-32-32-4.map",
            )
        )
    elif "dense" in args.results_pkl:
        maze = load_mapf_map(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../data/maps/den312d.map",
            )
        )

    statistics = None
    with open(results_base_path + args.results_pkl, "rb") as f:
        statistics = pickle.load(f)

    num_samples = len(statistics.stats)

    (
        multi_agent_steps,
        multi_agent_mcts_steps,
        single_agent_steps,
        single_agent_ca_steps,
    ) = ([], [], [], [])
    (
        multi_agent_first_gp_steps,
        multi_agent_mcts_first_gp_steps,
        single_agent_first_gp_steps,
        single_agent_ca_first_gp_steps,
    ) = (
        [],
        [],
        [],
        [],
    )
    (
        multi_agent_phenomenons_discovered,
        multi_agent_mcts_phenomenons_discovered,
        single_agent_phenomenons_discovered,
        single_agent_ca_phenomenons_discovered,
    ) = ([], [], [], [])
    ratios = []
    total_nodes_expanded, total_nodes_generated, max_nodes_generated = 0, 0, 0

    mission_duration = statistics.mission_duration
    for sample in range(num_samples):

        gp_locations = statistics.stats[sample].gp_locations
        agent_locations = statistics.stats[sample].agent_locations

        num_expanded_nodes = statistics.stats[sample].nodes_expanded
        num_generated_nodes = statistics.stats[sample].nodes_generated

        total_nodes_expanded += num_expanded_nodes
        total_nodes_generated += num_generated_nodes
        ratio = num_expanded_nodes / num_generated_nodes
        ratios.append(ratio)

        sample_multi_agent_phenomenons_discovered = set()
        sample_multi_agent_mcts_phenomenons_discovered = set()
        sample_single_agent_phenomenons_discovered = set()
        sample_single_agent_ca_phenomenons_discovered = set()

        # Verify that the paths of agents do not lead to collisions!
        (
            multi_agent_last_valid_step,
            multi_agent_mcts_last_valid_step,
            single_agent_last_valid_step,
            single_agent_ca_last_valid_step,
        ) = (
            mission_duration,
            mission_duration,
            mission_duration,
            mission_duration,
        )

        multi_collision, multi_mcts_collision, single_collision, single_ca_collision = (
            False,
            False,
            False,
            False,
        )
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
                        maze=maze,
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

                multi_agent_mcts_paths_valid_steps = (
                    validate_paths(
                        (
                            agent_i,
                            statistics.stats[sample]
                            .multi_agent_mcts_stats[agent_i]
                            .path,
                        ),
                        (
                            agent_j,
                            statistics.stats[sample]
                            .multi_agent_mcts_stats[agent_j]
                            .path,
                        ),
                        mission_duration,
                        maze=maze,
                    ),
                )

                multi_mcts_collision = (
                    True
                    if multi_agent_mcts_paths_valid_steps[0] < mission_duration
                    else False
                )
                multi_agent_mcts_last_valid_step = min(
                    multi_agent_mcts_last_valid_step,
                    multi_agent_mcts_paths_valid_steps[0],
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
                        maze=maze,
                        multi=False,
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

                single_agent_ca_paths_valid_steps = (
                    validate_paths(
                        (
                            agent_i,
                            statistics.stats[sample]
                            .single_agent_collision_avoidance_stats[agent_i]
                            .path,
                        ),
                        (
                            agent_j,
                            statistics.stats[sample]
                            .single_agent_collision_avoidance_stats[agent_j]
                            .path,
                        ),
                        mission_duration,
                        maze=maze,
                        multi=False,
                        single_ca=True,
                    ),
                )
                single_ca_collision = (
                    True
                    if single_agent_ca_paths_valid_steps[0] < mission_duration
                    else False
                )
                single_agent_ca_last_valid_step = min(
                    single_agent_ca_last_valid_step,
                    single_agent_ca_paths_valid_steps[0],
                )

        (
            multi_last_gp_found_step,
            multi_mcts_last_gp_found_step,
            single_last_gp_found_step,
            single_ca_last_gp_found_step,
        ) = (0, 0, 0, 0)

        (
            avg_multi_agent_first_gp_found_step,
            avg_multi_agent_mcts_first_gp_found_step,
            avg_single_agent_first_gp_found_step,
            avg_single_agent_ca_first_gp_found_step,
        ) = (0, 0, 0, 0)

        local_agents_gp_found = [mission_duration for _ in range(len(agent_locations))]
        for step in range(multi_agent_last_valid_step):
            agent_coords = []
            for agent in range(len(agent_locations)):
                coord = statistics.stats[sample].multi_agent_stats[agent].path[step]
                agent_coords.append(
                    linearize_coordinate(coord[0], coord[1], statistics.cols)
                )
                coord_compare = (coord[0], coord[1])
                if (
                    coord_compare in gp_locations
                    and coord_compare not in sample_multi_agent_phenomenons_discovered
                ):
                    multi_last_gp_found_step = step
                    sample_multi_agent_phenomenons_discovered.add(coord_compare)
                    if local_agents_gp_found[agent] == mission_duration:
                        local_agents_gp_found[agent] = step

            within_range = within_range_agents(
                len(agent_locations),
                agent_coords,
                statistics.cols,
                statistics.communication_range,
            )

            if len(within_range) > 0:
                default_horizon = 2
                len_action_space = len(
                    [action_type.value for action_type in ActionType]
                )
                for multi_agent_spawn in within_range:
                    for horizon in range(1, default_horizon + 1):
                        max_nodes_generated += multi_agent_spawn ** (
                            len_action_space * horizon
                        )

        avg_multi_agent_first_gp_found_step = np.sum(local_agents_gp_found) / len(
            agent_locations
        )

        local_agents_gp_found = [mission_duration for _ in range(len(agent_locations))]
        for step in range(multi_agent_mcts_last_valid_step):
            agent_coords = []
            for agent in range(len(agent_locations)):
                coord = (
                    statistics.stats[sample].multi_agent_mcts_stats[agent].path[step]
                )
                agent_coords.append(
                    linearize_coordinate(coord[0], coord[1], statistics.cols)
                )
                coord_compare = (coord[0], coord[1])
                if (
                    coord_compare in gp_locations
                    and coord_compare
                    not in sample_multi_agent_mcts_phenomenons_discovered
                ):
                    multi_mcts_last_gp_found_step = step
                    sample_multi_agent_mcts_phenomenons_discovered.add(coord_compare)
                    if local_agents_gp_found[agent] == mission_duration:
                        local_agents_gp_found[agent] = step

        avg_multi_agent_mcts_first_gp_found_step = np.sum(local_agents_gp_found) / len(
            agent_locations
        )

        local_agents_gp_found = [mission_duration for _ in range(len(agent_locations))]
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
                    if local_agents_gp_found[agent] == mission_duration:
                        local_agents_gp_found[agent] = step
        avg_single_agent_first_gp_found_step = np.sum(local_agents_gp_found) / len(
            agent_locations
        )

        local_agents_gp_found = [mission_duration for _ in range(len(agent_locations))]
        for step in range(single_agent_ca_last_valid_step):
            for agent in range(len(agent_locations)):
                coord = (
                    statistics.stats[sample]
                    .single_agent_collision_avoidance_stats[agent]
                    .path[step]
                )
                coord_compare = (coord[0], coord[1])
                if (
                    coord_compare in gp_locations
                    and coord_compare
                    not in sample_single_agent_ca_phenomenons_discovered
                ):
                    single_ca_last_gp_found_step = step
                    sample_single_agent_ca_phenomenons_discovered.add(coord_compare)
                    if local_agents_gp_found[agent] == mission_duration:
                        local_agents_gp_found[agent] = step
        avg_single_agent_ca_first_gp_found_step = np.sum(local_agents_gp_found) / len(
            agent_locations
        )

        multi_agent_phenomenons_discovered.append(
            len(sample_multi_agent_phenomenons_discovered)
        )
        multi_agent_mcts_phenomenons_discovered.append(
            len(sample_multi_agent_mcts_phenomenons_discovered)
        )
        single_agent_phenomenons_discovered.append(
            len(sample_single_agent_phenomenons_discovered)
        )
        single_agent_ca_phenomenons_discovered.append(
            len(sample_single_agent_ca_phenomenons_discovered)
        )

        multi_agent_first_gp_steps.append(avg_multi_agent_first_gp_found_step)
        multi_agent_mcts_first_gp_steps.append(avg_multi_agent_mcts_first_gp_found_step)
        single_agent_first_gp_steps.append(avg_single_agent_first_gp_found_step)
        single_agent_ca_first_gp_steps.append(avg_single_agent_ca_first_gp_found_step)

        if (
            len(sample_multi_agent_phenomenons_discovered) != len(gp_locations)
            or multi_collision
        ):
            multi_agent_steps.append(mission_duration)
        else:
            multi_agent_steps.append(multi_last_gp_found_step)

        if (
            len(sample_multi_agent_mcts_phenomenons_discovered) != len(gp_locations)
            or multi_mcts_collision
        ):
            multi_agent_mcts_steps.append(mission_duration)
        else:
            multi_agent_mcts_steps.append(multi_mcts_last_gp_found_step)

        if (
            len(sample_single_agent_phenomenons_discovered) != len(gp_locations)
            or single_collision
        ):
            single_agent_steps.append(mission_duration)
        else:
            single_agent_steps.append(single_last_gp_found_step)

        if (
            len(sample_single_agent_ca_phenomenons_discovered) != len(gp_locations)
            or single_ca_collision
        ):
            single_agent_ca_steps.append(mission_duration)
        else:
            single_agent_ca_steps.append(single_ca_last_gp_found_step)

    # Compile the results and print them
    ratios = np.array(ratios)
    multi_agent_steps = np.array(multi_agent_steps)
    multi_agent_mcts_steps = np.array(multi_agent_mcts_steps)
    single_agent_steps = np.array(single_agent_steps)
    single_agent_ca_steps = np.array(single_agent_ca_steps)
    multi_agent_phenomenons_discovered = np.array(multi_agent_phenomenons_discovered)
    multi_agent_mcts_phenomenons_discovered = np.array(
        multi_agent_mcts_phenomenons_discovered
    )
    single_agent_phenomenons_discovered = np.array(single_agent_phenomenons_discovered)
    single_agent_ca_phenomenons_discovered = np.array(
        single_agent_ca_phenomenons_discovered
    )

    diff_array = np.abs(
        multi_agent_phenomenons_discovered - single_agent_phenomenons_discovered
    )
    max_diff = np.max(diff_array)
    sample_to_visualize = np.random.choice(np.flatnonzero(diff_array == max_diff))

    print(
        f"Single-agent steps: {np.mean(single_agent_steps)} +/- {np.std(single_agent_steps)}"
    )
    print(
        "Single-agent with collision avoidance steps: "
        f"{np.mean(single_agent_ca_steps)} +/- {np.std(single_agent_ca_steps)}"
    )
    print(
        f"Multi-agent steps: {np.mean(multi_agent_steps)} +/- {np.std(multi_agent_steps)}"
    )
    print(
        f"Multi-agent with MCTS steps: {np.mean(multi_agent_mcts_steps)} +/- {np.std(multi_agent_mcts_steps)}"
    )
    print(
        f"Single-agent first GP found step: {np.mean(single_agent_first_gp_steps)}"
        f" +/- {np.std(single_agent_first_gp_steps)}"
    )
    print(
        f"Single-agent with collision avoidance first GP found step: {np.mean(single_agent_ca_first_gp_steps)}"
        f" +/- {np.std(single_agent_ca_first_gp_steps)}"
    )
    print(
        f"Multi-agent first GP found step: {np.mean(multi_agent_first_gp_steps)}"
        f" +/- {np.std(multi_agent_first_gp_steps)}"
    )
    print(
        f"Multi-agent first GP found step: {np.mean(multi_agent_mcts_first_gp_steps)}"
        f" +/- {np.std(multi_agent_mcts_first_gp_steps)}"
    )
    print(
        "Single-agent phenomenons discovered: "
        f"{np.mean(single_agent_phenomenons_discovered)} +/- {np.std(single_agent_phenomenons_discovered)}"
    )
    print(
        "Single-agent with collision avoidance phenomenons discovered: "
        f"{np.mean(single_agent_ca_phenomenons_discovered)} +/- {np.std(single_agent_ca_phenomenons_discovered)}"
    )
    print(
        "Multi-agent phenomenons discovered: "
        f"{np.mean(multi_agent_phenomenons_discovered)} +/- {np.std(multi_agent_phenomenons_discovered)}"
    )
    print(
        "Multi-agent phenomenons discovered: "
        f"{np.mean(multi_agent_mcts_phenomenons_discovered)} +/- {np.std(multi_agent_mcts_phenomenons_discovered)}"
    )
    print(
        f"Ratio of expanded nodes to generated nodes: {np.mean(ratios)} +/- {np.std(ratios)}"
    )
    print(
        f"Ratio of generated nodes to maximum possible nodes: {total_nodes_generated / max_nodes_generated}"
    )
    print(
        f"Ratio of expanded nodes to maximum possible nodes: {total_nodes_expanded / max_nodes_generated}"
    )

    np.savez(
        store_base_path + args.results_pkl[:-4],
        multi_agent_steps=multi_agent_steps,
        single_agent_steps=single_agent_steps,
        single_agent_ca_steps=single_agent_ca_steps,
        multi_agent_phenomenons_discovered=multi_agent_phenomenons_discovered,
        single_agent_phenomenons_discovered=single_agent_phenomenons_discovered,
        single_agent_ca_phenomenons_discovered=single_agent_ca_phenomenons_discovered,
        multi_agent_first_gp_steps=multi_agent_first_gp_steps,
        single_agent_first_gp_steps=single_agent_first_gp_steps,
        single_agent_ca_first_gp_steps=single_agent_ca_first_gp_steps,
        ratios=ratios,
        max_possible_nodes_data=np.array(
            [total_nodes_generated, total_nodes_expanded, max_nodes_generated]
        ),
    )

    # Now we visualize the paths of the agents for a given sample

    logging.info("Going to visualize the sample : %d", sample_to_visualize)

    logging.info(
        "Number of GPs discovered by single-agent: %d",
        single_agent_phenomenons_discovered[sample_to_visualize],
    )
    logging.info(
        "Number of GPs discovered by single-agent with collision avoidance: %d",
        single_agent_ca_phenomenons_discovered[sample_to_visualize],
    )
    logging.info(
        "Number of GPs discovered by multi-agent: %d",
        multi_agent_phenomenons_discovered[sample_to_visualize],
    )

    rows, cols = (
        statistics.rows,
        statistics.cols,
    )
    max_gps, num_agents, mission_duration, communication_range = (
        statistics.max_gps,
        statistics.num_agents,
        statistics.mission_duration,
        statistics.communication_range,
    )
    gp_locations = statistics.stats[sample_to_visualize].gp_locations
    agent_locations = statistics.stats[sample_to_visualize].agent_locations

    params = Parameters(
        theta_1=np.float64(0.4),
        theta_2=np.float64(0.01),
        u_tilde=np.float64(1.4),
        P_1=np.float64(0.98),
        P_2=np.float64(0.002),
        J=np.int64(5),
        measurement_noise=np.float64(0.2),
        distance_simplification=True,
    )

    grid, reward_map = generate_map(
        rows,
        cols,
        grid=maze,
        agent_locations=agent_locations,
        gp_means=np.ones(len(gp_locations)).tolist(),
        gp_locations=gp_locations,
        parameters=params,
    )

    # (rows, columns) -> (y, x) for images!

    y = np.linspace(0, reward_map.num_of_rows, 1000)
    x = np.linspace(0, reward_map.num_of_cols, 1000)
    xx, yy = np.meshgrid(x, y)
    meshgrid = np.dstack((xx, yy))
    zz = np.zeros_like(xx)
    for i in range(len(reward_map.locations)):
        linear_location = reward_map.locations[i]
        location_coord = reward_map.get_coordinate(linear_location)
        location_coord = np.array([location_coord[1], location_coord[0]])
        gaussian = reward_map.means[i] * multivariate_normal.pdf(
            meshgrid, mean=location_coord, cov=1
        )
        zz += gaussian
    zz /= np.max(zz)

    agent_colors = ["g", "b", "r", "deeppink", "y", "m", "c", "w"]

    if maze is not None:
        y_obstacle = np.linspace(0, reward_map.num_of_rows, maze.shape[0])
        x_obstacle = np.linspace(0, reward_map.num_of_cols, maze.shape[1])

        obstacles_interpolated = interp2d(x_obstacle, y_obstacle, maze, kind="linear")
        zz_obstacle = obstacles_interpolated(x, y)
        zz_obstacle[np.where(zz_obstacle >= 0.25)] = 1.0

    # Analysis for multi-agent
    analyse_results(
        sample_to_visualize,
        statistics,
        agent_locations,
        reward_map,
        agent_colors,
        [zz, zz_obstacle] if maze is not None else [zz],
        figures_base_path + "rh-ma-vulcan-" + args.results_pkl[:-4],
        type_of_analysis="multi",
    )

    # Analysis for single-agent
    analyse_results(
        sample_to_visualize,
        statistics,
        agent_locations,
        reward_map,
        agent_colors,
        [zz, zz_obstacle] if maze is not None else [zz],
        figures_base_path + "sa-vulcan-" + args.results_pkl[:-4],
        type_of_analysis="single",
    )

    # Analysis for single-agent with collision avoidance
    analyse_results(
        sample_to_visualize,
        statistics,
        agent_locations,
        reward_map,
        agent_colors,
        [zz, zz_obstacle] if maze is not None else [zz],
        figures_base_path + "sa-ca-vulcan-" + args.results_pkl[:-4],
        type_of_analysis="single_ca",
    )

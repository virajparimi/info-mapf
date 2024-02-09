import os
import sys
import pickle
import logging
import numpy as np
from copy import deepcopy
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Tuple, Set
from argparse import ArgumentParser


@dataclass
class VulcanStats:
    path: List[NDArray[np.int64]]
    phenomenons_discovered: Set[Tuple[int, int]]


@dataclass
class SampleStats:
    nodes_expanded: int
    nodes_generated: int
    avg_multi_agent_steps: float
    avg_single_agent_steps: float
    gp_locations: List[Tuple[int, int]]
    multi_agent_stats: List[VulcanStats]
    single_agent_stats: List[VulcanStats]
    agent_locations: List[Tuple[int, int]]
    multi_agent_phenomenons_discovered: float
    single_agent_phenomenons_discovered: float


NUM_SAMPLES = 10

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from map import Map, Parameters  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA


def load_mapf_map(filename: str) -> NDArray[np.bool_]:
    filecontents = None
    with open(filename, "r") as f:
        filecontents = f.readlines()

    filecontents = [line.rstrip() for line in filecontents]

    height = int(filecontents[1].split(" ")[1])
    width = int(filecontents[2].split(" ")[1])
    mapcontent = filecontents[4:]

    maze = np.ones((height, width), dtype=np.bool_)
    for row, line in enumerate(mapcontent):
        for col, char in enumerate(line):
            if char == "@":
                maze[row, col] = False
            elif char == "T":
                maze[row, col] = False

    return maze


if __name__ == "__main__":
    results_base_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/testing/"
    parser = ArgumentParser()
    parser.add_argument(
        "--map_type",
        type=str,
        default="original",
        choices=["original", "empty-16", "empty-32", "maze-32", "dense"],
        help="Type of map to use",
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

    # Extract the map parameters
    maze = None
    if args.map_type == "original":
        rows, cols, max_gps, num_agents, mission_duration, communication_range = (
            11,
            11,
            4,
            2,
            35,
            5,
        )
    elif args.map_type == "empty-16":
        rows, cols, max_gps, num_agents, mission_duration, communication_range = (
            16,
            16,
            5,
            3,
            50,
            5,
        )
    elif args.map_type == "empty-32":
        rows, cols, max_gps, num_agents, mission_duration, communication_range = (
            32,
            32,
            10,
            4,
            100,
            5,
        )
    elif args.map_type == "maze-32":
        maze = load_mapf_map(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../data/maps/maze-32-32-4.map",
            )
        )
        rows, cols, max_gps, num_agents, mission_duration, communication_range = (
            maze.shape[0],
            maze.shape[1],
            10,
            4,
            100,
            5,
        )
    elif args.map_type == "dense":
        maze = load_mapf_map(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../data/maps/den312d.map",
            )
        )
        rows, cols, max_gps, num_agents, mission_duration, communication_range = (
            maze.shape[0],
            maze.shape[1],
            30,
            8,
            150,
            5,
        )
    else:
        raise ValueError("Invalid map type")

    results = []

    # Run the outer loop for 100 iterations
    for sample in range(NUM_SAMPLES):

        logging.info("Running sample: %d", sample + 1)

        # Sample the number of phenomenons to generate
        num_phenomenon = np.random.randint(num_agents, max_gps + 1)

        # Spawn the agents and the phenomenons such that the phenomenons are not spawned on the agents
        agent_locations = set()
        while len(agent_locations) < num_agents:
            agent_loc = (np.random.randint(0, rows), np.random.randint(0, cols))
            if len(agent_locations) == 0:
                within_communication_range = True
            else:
                within_communication_range = False
            for agent_location in agent_locations:
                if (
                    np.sum(np.abs(np.array(agent_loc) - np.array(agent_location)))
                    < communication_range
                ):
                    within_communication_range = True
                    break
            if (
                maze is not None
                and maze[agent_loc[0], agent_loc[1]]
                and within_communication_range
            ):
                agent_locations.add(agent_loc)
            elif maze is None and within_communication_range:
                agent_locations.add(agent_loc)
        agent_locations = list(agent_locations)

        gp_locations = set()
        while len(gp_locations) < num_phenomenon:
            gp_location = (
                np.random.randint(0, rows),
                np.random.randint(0, cols),
            )
            if (
                maze is not None
                and maze[gp_location[0], gp_location[1]]
                and gp_location not in agent_locations
            ):
                gp_locations.add(gp_location)
            elif maze is None and gp_location not in agent_locations:
                gp_locations.add(gp_location)
        gp_locations = list(gp_locations)

        # Generate the map
        map = generate_map(
            rows,
            cols,
            maze=maze,
            agent_locations=agent_locations,
            gp_means=np.ones(num_phenomenon).tolist(),
            gp_locations=gp_locations,
            parameters=params,
        )

        # Object to store the statistics of this iteration
        sample_stats = SampleStats(
            nodes_expanded=0,
            nodes_generated=0,
            multi_agent_stats=[],
            single_agent_stats=[],
            avg_multi_agent_steps=0,
            avg_single_agent_steps=0,
            gp_locations=gp_locations,
            agent_locations=agent_locations,
            multi_agent_phenomenons_discovered=0,
            single_agent_phenomenons_discovered=0,
        )

        # Generate the agents and run the planner
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
        rh_ma_vulcan.planner()

        sample_stats.nodes_expanded = rh_ma_vulcan.nodes_expanded
        sample_stats.nodes_generated = rh_ma_vulcan.nodes_generated

        # Extract the paths of the agents after running multi-agent vulcan
        last_step_when_gp_found = [0 for _ in range(len(vulcan_agents))]
        for idx, agent in enumerate(vulcan_agents):
            vulcan_agent_stats = VulcanStats(
                path=[],
                phenomenons_discovered=set(),
            )
            for v_location in agent.visited_locations:
                v_coord = vulcan_map.get_coordinate(v_location)
                v_coord_compare = (v_coord[0], v_coord[1])
                if (
                    v_coord_compare in gp_locations
                    and v_coord_compare not in vulcan_agent_stats.phenomenons_discovered
                ):
                    vulcan_agent_stats.phenomenons_discovered.add(v_coord_compare)
                    last_step_when_gp_found[idx] = len(vulcan_agent_stats.path)
                vulcan_agent_stats.path.append(v_coord)

            sample_stats.multi_agent_stats.append(vulcan_agent_stats)

        combined_gps_found = set()
        for idx, agent in enumerate(vulcan_agents):
            combined_gps_found |= sample_stats.multi_agent_stats[
                idx
            ].phenomenons_discovered
        if len(combined_gps_found) != len(gp_locations):
            sample_stats.avg_multi_agent_steps = mission_duration
        else:
            sample_stats.avg_multi_agent_steps = max(last_step_when_gp_found)

        sample_stats.multi_agent_phenomenons_discovered = len(combined_gps_found)

        # At this point multi-agent vulcan has been run. Now need to run single agent vulcan

        # Repeat the same process for single agent vulcan
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

        # Extract the paths of the agents after running single-agent vulcan
        for idx, agent in enumerate(vulcan_agents):
            agent.adaptive_search()

        phenomenons_discovered = set()
        agent_coords = [[] for _ in range(num_agents)]
        last_step_when_gp_found = [0 for _ in range(len(vulcan_agents))]
        agent_phenomenons_discovered = [set() for _ in range(num_agents)]
        for step in range(mission_duration):
            current_coords = []
            for idx, agent in enumerate(vulcan_agents):
                v_coord = vulcan_map.get_coordinate(agent.visited_locations[step])
                v_coord_tuple = (v_coord[0], v_coord[1])
                current_coords.append(v_coord_tuple)
                agent_coords[idx].append(v_coord_tuple)
            if len(set(current_coords)) != num_agents:
                sample_stats.avg_single_agent_steps = mission_duration
                break
            else:
                for idx, v_coords in enumerate(agent_coords):
                    v_coord = v_coords[step]
                    v_coord_compare = (v_coord[0], v_coord[1])
                    if (
                        v_coord_compare in gp_locations
                        and v_coord_compare not in agent_phenomenons_discovered[idx]
                    ):
                        phenomenons_discovered.add(v_coord_compare)
                        last_step_when_gp_found[idx] = len(v_coords)
                        agent_phenomenons_discovered[idx].add(v_coord_compare)

        for agent in range(num_agents):
            vulcan_agent_stats = VulcanStats(
                path=agent_coords[agent],
                phenomenons_discovered=agent_phenomenons_discovered[agent],
            )
            sample_stats.single_agent_stats.append(vulcan_agent_stats)

        if sample_stats.avg_single_agent_steps == 0:
            # This is still not set
            if len(phenomenons_discovered) != len(gp_locations):
                sample_stats.avg_single_agent_steps = mission_duration
            else:
                sample_stats.avg_single_agent_steps = max(last_step_when_gp_found)
        sample_stats.single_agent_phenomenons_discovered = len(phenomenons_discovered)

        results.append(sample_stats)

    # Compile the results and print them
    multi_agent_steps, single_agent_steps = [], []
    multi_agent_phenomenons_discovered, single_agent_phenomenons_discovered = [], []
    single_agent_wise_phenomenons_discovered = [
        [] for _ in range(len(results[0].single_agent_stats))
    ]
    multi_agent_wise_phenomenons_discovered = [
        [] for _ in range(len(results[0].multi_agent_stats))
    ]

    for result in results:
        multi_agent_steps.append(result.avg_multi_agent_steps)
        single_agent_steps.append(result.avg_single_agent_steps)
        multi_agent_phenomenons_discovered.append(
            result.multi_agent_phenomenons_discovered
        )
        single_agent_phenomenons_discovered.append(
            result.single_agent_phenomenons_discovered
        )
        for idx, agent in enumerate(result.single_agent_stats):
            single_agent_wise_phenomenons_discovered[idx].append(
                len(agent.phenomenons_discovered)
            )
        for idx, agent in enumerate(result.multi_agent_stats):
            multi_agent_wise_phenomenons_discovered[idx].append(
                len(agent.phenomenons_discovered)
            )

    single_agent_steps = np.array(single_agent_steps)
    multi_agent_steps = np.array(multi_agent_steps)

    print(
        f"Single-agent steps: {np.mean(single_agent_steps)} +/- {np.std(single_agent_steps)}"
    )
    print(
        f"Multi-agent steps: {np.mean(multi_agent_steps)} +/- {np.std(multi_agent_steps)}"
    )

    single_agent_phenomenons_discovered = np.array(single_agent_phenomenons_discovered)
    multi_agent_phenomenons_discovered = np.array(multi_agent_phenomenons_discovered)

    print(
        "Single-agent phenomenons discovered: "
        f"{np.mean(single_agent_phenomenons_discovered)} +/- {np.std(single_agent_phenomenons_discovered)}"
    )
    print(
        "Multi-agent phenomenons discovered: "
        f"{np.mean(multi_agent_phenomenons_discovered)} +/- {np.std(multi_agent_phenomenons_discovered)}"
    )

    for idx, agent in enumerate(single_agent_wise_phenomenons_discovered):
        agent = np.array(agent)
        if agent.shape[0] == 0:
            mean, std = 0, 0
        else:
            mean, std = np.mean(agent), np.std(agent)
        print(f"Single-agent {idx + 1} phenomenons discovered: {mean} +/- {std}")
    for idx, agent in enumerate(multi_agent_wise_phenomenons_discovered):
        agent = np.array(agent)
        if agent.shape[0] == 0:
            mean, std = 0, 0
        else:
            mean, std = np.mean(agent), np.std(agent)
        print(f"Multi-agent {idx + 1} phenomenons discovered: {mean} +/- {std}")

    with open(results_base_path + "results_" + args.map_type + ".pkl", "wb") as f:
        pickle.dump(results, f)

    print("Results saved")

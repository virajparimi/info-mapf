import os
import sys
import pickle
import logging
import numpy as np
from copy import deepcopy
from numpy.typing import NDArray
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import List, Tuple, Set, Union, Dict, Any


@dataclass
class VulcanStats:
    path: List[NDArray[np.int64]]
    phenomenons_discovered: Set[Tuple[int, int]]


@dataclass
class SampleStats:
    nodes_expanded: int
    nodes_generated: int
    gp_locations: List[Tuple[int, int]]
    multi_agent_stats: List[VulcanStats]
    single_agent_stats: List[VulcanStats]
    single_agent_collision_avoidance_stats: List[VulcanStats]
    agent_locations: List[Tuple[int, int]]


@dataclass
class Statistics:
    rows: int
    cols: int
    max_gps: int
    num_agents: int
    mission_duration: int
    communication_range: int
    stats: List[SampleStats]


NUM_SAMPLES = 100

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA
from rh_sa_vulcan import SingleAgentVulcan  # NOQA
from map import Grid, RewardMap, Parameters  # NOQA


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


def setup_experiment_parameters(
    map_type: str,
) -> Tuple[int, int, int, int, int, int, Union[NDArray[np.bool_], None]]:
    maze = None
    if map_type == "original":
        rows, cols, max_gps, num_agents, mission_duration, communication_range = (
            11,
            11,
            4,
            2,
            35,
            5,
        )
    elif map_type == "empty-16":
        rows, cols, max_gps, num_agents, mission_duration, communication_range = (
            16,
            16,
            5,
            3,
            50,
            5,
        )
    elif map_type == "empty-32":
        rows, cols, max_gps, num_agents, mission_duration, communication_range = (
            32,
            32,
            10,
            4,
            100,
            5,
        )
    elif map_type == "maze-32":
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
    elif map_type == "dense":
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
    return (
        rows,
        cols,
        max_gps,
        num_agents,
        mission_duration,
        communication_range,
        maze,
    )


def generate_agent_locations(
    num_agents: int,
    rows: int,
    cols: int,
    communication_range: int,
    maze: Union[NDArray[np.bool_], None] = None,
) -> List[Tuple[int, int]]:
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
    return list(agent_locations)


def generate_gp_locations(
    num_phenomenon: int,
    rows: int,
    cols: int,
    maze: Union[NDArray[np.bool_], None],
    agent_locations: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:

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
    return list(gp_locations)


def setup_vulcan_agents(
    agent_locations: List[Tuple[int, int]],
    grid: Grid,
    reward_map: RewardMap,
    mission_duration: int,
) -> Tuple[List[Agent], Grid]:

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
    return (vulcan_agents, vulcan_grid)


def execute_sample(parameter: Dict[str, Any], sample_id: int) -> SampleStats:

    logging.info("Running sample: %d", sample_id + 1)

    # Sample the number of phenomenons to generate
    num_phenomenon = np.random.randint(
        parameter["num_agents"], parameter["max_gps"] + 1
    )

    # Spawn the agents and the phenomenons such that the phenomenons are not spawned on the agents
    agent_locations = generate_agent_locations(
        parameter["num_agents"],
        parameter["rows"],
        parameter["cols"],
        parameter["communication_range"],
        parameter["maze"],
    )
    gp_locations = generate_gp_locations(
        num_phenomenon,
        parameter["rows"],
        parameter["cols"],
        parameter["maze"],
        agent_locations,
    )

    # Generate the map
    grid, reward_map = generate_map(
        parameter["rows"],
        parameter["cols"],
        grid=parameter["maze"],
        agent_locations=agent_locations,
        gp_means=np.ones(num_phenomenon).tolist(),
        gp_locations=gp_locations,
        parameters=parameter["params"],
    )

    # Object to store the statistics of this iteration
    sample_stats = SampleStats(
        nodes_expanded=0,
        nodes_generated=0,
        multi_agent_stats=[],
        single_agent_stats=[],
        single_agent_collision_avoidance_stats=[],
        gp_locations=gp_locations,
        agent_locations=agent_locations,
    )

    vulcan_agents, vulcan_grid = setup_vulcan_agents(
        agent_locations, grid, reward_map, parameter["mission_duration"]
    )

    logging.info("Running multi-agent vulcan")
    rh_ma_vulcan = MultiAgentVulcan(
        grid=vulcan_grid,
        reward_map=reward_map,
        agents=vulcan_agents,
        communication_range=parameter["communication_range"],
    )
    rh_ma_vulcan.planner()

    sample_stats.nodes_expanded = rh_ma_vulcan.nodes_expanded
    sample_stats.nodes_generated = rh_ma_vulcan.nodes_generated

    # Extract the paths of the agents after running multi-agent vulcan
    for idx, agent in enumerate(vulcan_agents):
        vulcan_agent_stats = VulcanStats(
            path=[],
            phenomenons_discovered=set(),
        )
        for v_location in agent.visited_locations:
            v_coord = vulcan_grid.get_coordinate(v_location)
            v_coord_compare = (v_coord[0], v_coord[1])
            if v_coord_compare in gp_locations:
                vulcan_agent_stats.phenomenons_discovered.add(v_coord_compare)
            vulcan_agent_stats.path.append(v_coord)

        sample_stats.multi_agent_stats.append(vulcan_agent_stats)

    # At this point multi-agent vulcan has been run. Now need to run single agent vulcan

    # Repeat the same process for single agent vulcan
    vulcan_agents, vulcan_grid = setup_vulcan_agents(
        agent_locations, grid, reward_map, parameter["mission_duration"]
    )

    # Extract the paths of the agents after running single-agent vulcan
    for idx, agent in enumerate(vulcan_agents):
        logging.info("Running single-agent vulcan for agent: %d", idx)
        agent.adaptive_search()

    # Extract the paths of the agents after running multi-agent vulcan
    for idx, agent in enumerate(vulcan_agents):
        vulcan_agent_stats = VulcanStats(
            path=[],
            phenomenons_discovered=set(),
        )
        for v_location in agent.visited_locations:
            v_coord = vulcan_grid.get_coordinate(v_location)
            v_coord_compare = (v_coord[0], v_coord[1])
            if v_coord_compare in gp_locations:
                vulcan_agent_stats.phenomenons_discovered.add(v_coord_compare)
            vulcan_agent_stats.path.append(v_coord)

        sample_stats.single_agent_stats.append(vulcan_agent_stats)

    """
    At this point multi-agent vulcan and single-agent vulcan without collision avoidance has been run.
    Now need to run single agent vulcan with collision avoidance
    """

    vulcan_agents, vulcan_grid = setup_vulcan_agents(
        agent_locations, grid, reward_map, parameter["mission_duration"]
    )

    logging.info("Running multi-agent vulcan")
    rh_sa_vulcan = SingleAgentVulcan(
        grid=vulcan_grid,
        reward_map=reward_map,
        agents=vulcan_agents,
    )
    rh_sa_vulcan.planner()

    # Extract the paths of the agents after running multi-agent vulcan
    for idx, agent in enumerate(vulcan_agents):
        vulcan_agent_stats = VulcanStats(
            path=[],
            phenomenons_discovered=set(),
        )
        for v_location in agent.visited_locations:
            v_coord = vulcan_grid.get_coordinate(v_location)
            v_coord_compare = (v_coord[0], v_coord[1])
            if v_coord_compare in gp_locations:
                vulcan_agent_stats.phenomenons_discovered.add(v_coord_compare)
            vulcan_agent_stats.path.append(v_coord)

        sample_stats.single_agent_collision_avoidance_stats.append(vulcan_agent_stats)

    filename = (
        parameter["results_base_path"] + "results_" + parameter["map_type"] + ".pkl"
    )
    if not os.path.isfile(filename):
        statistics = Statistics(
            rows=parameter["rows"],
            cols=parameter["cols"],
            max_gps=parameter["max_gps"],
            num_agents=parameter["num_agents"],
            mission_duration=parameter["mission_duration"],
            communication_range=parameter["communication_range"],
            stats=[sample_stats],
        )
        with open(
            parameter["results_base_path"]
            + "results_"
            + parameter["map_type"]
            + ".pkl",
            "wb",
        ) as f:
            pickle.dump(statistics, f)
        logging.info("Results saved. Size of results: %d", len(statistics.stats))

    else:
        with open(
            parameter["results_base_path"]
            + "results_"
            + parameter["map_type"]
            + ".pkl",
            "rb",
        ) as f:
            statistics = pickle.load(f)
            statistics.stats.append(sample_stats)
        with open(
            parameter["results_base_path"]
            + "results_"
            + parameter["map_type"]
            + ".pkl",
            "wb",
        ) as f:
            pickle.dump(statistics, f)
        logging.info(
            "Results loaded and saved. Size of results: %d", len(statistics.stats)
        )

    return sample_stats


if __name__ == "__main__":
    results_base_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/../data/updated_ablations/"
    )
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
    rows, cols, max_gps, num_agents, mission_duration, communication_range, maze = (
        setup_experiment_parameters(args.map_type)
    )
    arguments = {
        "params": params,
        "maze": maze,
        "rows": rows,
        "cols": cols,
        "max_gps": max_gps,
        "num_agents": num_agents,
        "map_type": args.map_type,
        "mission_duration": mission_duration,
        "results_base_path": results_base_path,
        "communication_range": communication_range,
    }

    for i in range(NUM_SAMPLES):
        execute_sample(arguments, i)

    print("All results saved")

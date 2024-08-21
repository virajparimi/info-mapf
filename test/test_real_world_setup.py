import os
import sys
import pickle
import logging
import numpy as np
from copy import deepcopy
from numpy.typing import NDArray
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import List, Tuple, Union, Dict, Any
from test_mapf_suite import (
    VulcanStats,
    SampleStats,
)
from utils import (
    load_data_to_pandas,
    extract_grid_from_data,
    generate_map_from_data,
    extract_rows_and_cols_from_data,
)


MAX_GROUP_SIZE=3

@dataclass
class RealWorldStatistics:
    rows: int
    cols: int
    max_gps: int
    num_agents: int
    dataset_name: str
    mission_duration: int
    obstacle_threshold: int
    communication_range: int
    stats: List[SampleStats]
    cell_size_degrees: float
    bounds: Tuple[float, float, float, float]


NUM_SAMPLES = 100

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from agent import Agent  # NOQA
from utils import generate_map  # NOQA
from rh_ma_vulcan import MultiAgentVulcan  # NOQA
from rh_sa_vulcan import SingleAgentVulcan  # NOQA
from map import Grid, RewardMap, Parameters  # NOQA


def setup_experiment_parameters(
    dataset_name: str,
) -> Tuple[int, int, int, int, int, List[str], Tuple[float, float, float, float]]:
    if dataset_name == "boston-harbor.xyz":
        (
            max_gps,
            num_agents,
            mission_duration,
            communication_range,
            obstacle_threshold,
            header,
            bounds,
        ) = (
            10,
            4,
            100,
            5,
            15,
            ["SURVEY", "LON", "LAT", "DEPTH"],
            (42.344, 42.355, -70.89, -70.876),
        )
    elif dataset_name == "galveston-bay.xyz":
        (
            max_gps,
            num_agents,
            mission_duration,
            communication_range,
            obstacle_threshold,
            header,
            bounds,
        ) = (
            30,
            10,
            100,
            5,
            2,
            ["SURVEY", "LAT", "LON", "DEPTH", "QUALITY_CODE", "ACTIVE"],
            (29.295284, 29.383748, -94.889612, -94.832103),
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is either not supported or invalid")
    return (
        max_gps,
        num_agents,
        mission_duration,
        communication_range,
        obstacle_threshold,
        header,
        bounds,
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


def generate_all_agent_locations(
        num_agents: int,
        rows: int,
        cols: int,
        communication_range: int,
        maze: Union[NDArray[np.bool_], None] = None,
) -> List[Tuple[int, int]]:

    if num_agents <= MAX_GROUP_SIZE:
        return generate_agent_locations(num_agents, rows, cols, communication_range, maze)
    else:
        groups = num_agents // MAX_GROUP_SIZE
        remainder = num_agents % MAX_GROUP_SIZE
        agent_locations = []
        for i in range(groups):
            agent_locations += generate_agent_locations(MAX_GROUP_SIZE, rows, cols, communication_range, maze)
        if remainder > 0:
            agent_locations += generate_agent_locations(remainder, rows, cols, communication_range, maze)
        return agent_locations


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

    rows, cols = extract_rows_and_cols_from_data(
        parameter["dataframe"], parameter["bounds"], parameter["cell_size_degrees"]
    )

    grid = extract_grid_from_data(
        parameter["dataframe"],
        parameter["bounds"],
        parameter["cell_size_degrees"],
        parameter["obstacle_threshold"],
    )

    # Spawn the agents and the phenomenons such that the phenomenons are not spawned on the agents
    agent_locations = generate_all_agent_locations(
        parameter["num_agents"],
        rows,
        cols,
        parameter["communication_range"],
        grid.obstacle_map,
    )
    gp_locations = generate_gp_locations(
        num_phenomenon,
        rows,
        cols,
        grid.obstacle_map,
        agent_locations,
    )

    # Generate the map
    grid, reward_map = generate_map_from_data(
        parameter["dataframe"],
        parameter["bounds"],
        parameter["cell_size_degrees"],
        parameter["obstacle_threshold"],
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
        multi_agent_mcts_stats=[],
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

    # At this point multi-agent vulcan has been run. Now need to run multi-agent vulcan with MCTS

    vulcan_agents, vulcan_grid = setup_vulcan_agents(
        agent_locations, grid, reward_map, parameter["mission_duration"]
    )

    logging.info("Running multi-agent vulcan with MCTS")
    rh_ma_mcts_vulcan = MultiAgentVulcan(
        grid=vulcan_grid,
        reward_map=reward_map,
        agents=vulcan_agents,
        communication_range=parameter["communication_range"],
        use_mcts=True,
    )
    rh_ma_mcts_vulcan.planner()

    logging.info("Number of MCTS nodes generated: %d", rh_ma_mcts_vulcan.num_mcts_nodes)

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

        sample_stats.multi_agent_mcts_stats.append(vulcan_agent_stats)

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

    logging.info("Running single-agent with collision avoidance vulcan")
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

    filename = parameter["results_base_path"] + parameter["dataset_name"] + ".pkl"
    if not os.path.isfile(filename):
        statistics = RealWorldStatistics(
            rows=rows,
            cols=cols,
            bounds=parameter["bounds"],
            max_gps=parameter["max_gps"],
            num_agents=parameter["num_agents"],
            dataset_name=parameter["dataset_name"],
            mission_duration=parameter["mission_duration"],
            cell_size_degrees=parameter["cell_size_degrees"],
            obstacle_threshold=parameter["obstacle_threshold"],
            communication_range=parameter["communication_range"],
            stats=[sample_stats],
        )
        with open(
            filename,
            "wb",
        ) as f:
            pickle.dump(statistics, f)
        logging.info("Results saved. Size of results: %d", len(statistics.stats))

    else:
        with open(
            filename,
            "rb",
        ) as f:
            statistics = pickle.load(f)
            statistics.stats.append(sample_stats)
        with open(
            filename,
            "wb",
        ) as f:
            pickle.dump(statistics, f)
        logging.info(
            "Results loaded and saved. Size of results: %d", len(statistics.stats)
        )

    return sample_stats


if __name__ == "__main__":
    dataset_base_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/maps/"
    results_base_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/results/"
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="boston-harbor.xyz",
        choices=["boston-harbor.xyz", "galveston-bay.xyz"],
        help="Which real-world dataset to use",
    )

    parser.add_argument(
        "--cell_size_degrees",
        type=float,
        default=0.0003,
        help="Size of each cell in degrees",
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
        theta_1=np.float64(1.25),
        theta_2=np.float64(args.cell_size_degrees * 4),
        u_tilde=np.float64(1.4),
        P_1=np.float64(0.98),
        P_2=np.float64(0.002),
        J=np.int64(5),
        measurement_noise=np.float64(0.2),
        distance_simplification=True,
    )

    # Extract the map parameters
    (
        max_gps,
        num_agents,
        mission_duration,
        communication_range,
        obstacle_threshold,
        header,
        bounds,
    ) = setup_experiment_parameters(args.dataset_name)

    if "boston" in args.dataset_name:
        delimiter = "\t"
    else:
        delimiter = ","
    dataframe = load_data_to_pandas(
        dataset_base_path + args.dataset_name, header, delimiter
    )

    for head in header:
        if head != "SURVEY":
            dataframe[head] = dataframe[head].astype(float)

    arguments = {
        "params": params,
        "bounds": bounds,
        "max_gps": max_gps,
        "dataframe": dataframe,
        "num_agents": num_agents,
        "dataset_name": args.dataset_name,
        "mission_duration": mission_duration,
        "results_base_path": results_base_path,
        "obstacle_threshold": obstacle_threshold,
        "communication_range": communication_range,
        "cell_size_degrees": args.cell_size_degrees,
    }

    start = 0
    filename = results_base_path + args.dataset_name + ".pkl"
    if os.path.isfile(filename):
        with open(
            filename,
            "rb",
        ) as f:
            statistics = pickle.load(f)
            start = len(statistics.stats)

    for i in range(start, NUM_SAMPLES):
        execute_sample(arguments, i)

    print("All results saved")

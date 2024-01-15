import numpy as np
from agent import Agent
from map import Map, Parameters
from numpy.typing import NDArray
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from typing import Union, List, Tuple


def generate_maze(
    rows: int,
    columns: int,
    agent_locations: Union[List[List[int]], None] = None,
    gp_means: Union[List[int], None] = None,
    gp_locations: Union[List[List[int]], None] = None,
    parameters: Union[Parameters, None] = None,
) -> Tuple[NDArray[np.bool8], Map]:
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
    return maze, map


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="single-small",
        options=["single-small", "single-large", "multi-small", "multi-large"],
        help="Type of Vulcan testing to perform",
    )

    parser.add_argument(
        "--distance_simplification",
        type=bool,
        default=True,
        help="Whether to use the distance simplification",
    )

    args = parser.parse_args()

    params = Parameters(
        theta_1=np.float64(0.4),
        theta_2=np.float64(0.01),
        u_tilde=np.float64(1.4),
        P_1=np.float64(0.98),
        P_2=np.float64(0.002),
        J=np.int64(5),
        distance_simplification=args.distance_simplification,
    )

    if args.type == "single-small":
        maze, map = generate_maze(5, 5, parameters=params)
    elif args.type == "single-large":
        maze, map = generate_maze(11, 11, parameters=params)
    elif args.type == "multi-small":
        maze, map = generate_maze(
            5, 5, gp_means=[1, 1], gp_locations=[[1, 1], [4, 4]], parameters=params
        )
    elif args.type == "multi-large":
        maze, map = generate_maze(
            11, 11, gp_means=[1, 1], gp_locations=[[1, 1], [10, 10]], parameters=params
        )
    else:
        raise ValueError("Invalid type")

    single_agent = Agent(id=1, start_location=0, map=map, mission_duration=20)

    single_agent.adaptive_search()

    path = []
    for observation in single_agent.mdp_handle.observations:
        location = observation.location
        path.append(map.get_coordinate(location))

    plt.plot([x[0] for x in path], [x[1] for x in path], "r--")
    plt.imshow(map.grid)
    plt.show()

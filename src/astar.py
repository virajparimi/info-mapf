# Credit for this: Nicholas Swift
# as found at https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
# Also credit to Ryan Collingwood for bug fixes
# Can be found here: https://gist.github.com/ryancollingwood/32446307e976a11a1185a5394d6657bc
# Modified by Jake Olkin for reward problem version

from __future__ import annotations

import heapq
import numpy as np
from map import Map
from warnings import warn
from typing import List, Union
from utils import get_nearest_locations, generate_map


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(
        self,
        t: int,
        parent: Union[Node, None] = None,
        position: Union[int, None] = None,
    ):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0
        self.t = t

    def __eq__(self, other: Node) -> bool:
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # Defining less than for purposes of heap queue
    def __lt__(self, other: Node) -> bool:
        return self.f < other.f

    # Defining greater than for purposes of heap queue
    def __gt__(self, other: Node) -> bool:
        return self.f > other.f


def return_path(current_node: Node) -> List[int]:
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def astar(
    map: Map,
    start: int,
    horizon: int,
    max_iterations: int = 500,
) -> Union[List[int], None]:
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param map: The map that we are planning on
    :param start: The start location from where the planning begins
    :param end: The end location of where we want to go
    """

    # Create start and end node
    start_node = Node(0, None, start)
    start_node.g = start_node.h = start_node.f = 0

    # We dont have end nodes so we do not need to create a node for that

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and add the start node
    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        time = current_node.t
        print(f"t: {time} h: {horizon}")
        if time == horizon:
            # If we hit our planning horizon then we return the path that we found till now
            return return_path(current_node)

        # Generate children
        children = []
        for neighbor in map.get_neighbors(current_node):
            # Get node position
            node_position = neighbor.location

            # Create new node
            new_node = Node(current_node.t + 1, current_node, node_position)
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if (
                len(
                    [
                        closed_child
                        for closed_child in closed_list
                        if (closed_child == child and closed_child.t == child.t)
                    ]
                )
                > 0
            ):
                continue

            # Create the f, g, and h values
            child.g = -1.0 * map.get_reward_from_traj([return_path(current_node)])
            post_traversal = map.get_grid_after_traj([return_path(current_node)])

            best_rewards = []
            max_range = horizon - child.t
            nearest_locations = get_nearest_locations([child.position], map, max_range)
            for location in nearest_locations:
                location_coords = map.get_coordinate(location)
                best_rewards.append(
                    post_traversal[location_coords[0], location_coords[1]]
                )

            best_rewards.sort()
            best_rewards = best_rewards[::-1]
            max_reward = sum(best_rewards[:max_range])

            # What we're doing isn't technically admissable but an h based off the best
            # reward we can get next isn't going to be too shabby for the sake of testing
            child.h = -max_reward
            child.f = child.g + child.h

            # Child is already in the open list
            if (
                len(
                    [
                        open_node
                        for open_node in open_list
                        if child.position == open_node.position
                        and child.t == open_node.t
                        and child.g > open_node.g
                    ]
                )
                > 0
            ):
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    warn("Couldn't get a path to destination")
    return None


def example(print_maze=True):
    maze = [
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
        ]
        * 2,
        [
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ]
        * 2,
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
        ]
        * 2,
    ]

    maze = np.array(maze)
    maze = 1 - maze  # Walls are 0 and free space is 1

    start = (0, 0)
    end = (len(maze) - 1, len(maze[0]) - 1)

    map = generate_map(maze.shape[0], maze.shape[1], maze, [start])
    linear_start = map.linearize_coordinate(start[0], start[1])
    linear_end = map.linearize_coordinate(end[0], end[1])

    path = astar(map, linear_start, linear_end)

    if print_maze and path is not None:
        path_coords = []
        for p in path:
            path_coords.append(map.get_coordinate(p))

        for step in path_coords:
            maze[step[0], step[1]] = 2

        for row in maze:
            line = []
            for col in row:
                if col == 1:
                    line.append("\u2588")
                elif col == 0:
                    line.append(" ")
                elif col == 2:
                    line.append(".")
            print("".join(line))

    print(path)

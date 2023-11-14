# Information Driven MAPF

This repo is for the information driven MAPF problem, where, given a set of agents, some phenomenon of interest, and a map
of the area, we calculate the path over all agents that gathers the most information.

##Overview of Method

This method uses a simple A* style search, however the innovation lies in how we calculate the heuristic functions, 
and how we can exploit the structure of the problem to prune unnecessary paths.

Simply put, we want our path to maximize the information gain from the observations taken along the way.
A given node in the search tree represents a path prefix for each agent. We can use this to calculate the 
information gain of these partial paths. Then, we can create an optimistic heuristic for the maximum information
gain from calculating the best information gain for each agent individually, and then just summing them together.


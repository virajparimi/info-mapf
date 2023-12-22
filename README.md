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

##Testing Information

In our tests, we will have a discritized grid for the agents to traverse, moving in the cardinal directions.
To represent the relationship between the phenomenon and the measured features, we will use gaussian processes as 
specified in Ben Ayton's Master's Thesis

This makes the assumption that, when the mean value for a gaussian process at a given location is above a certain threshold,
we will model that the probability that the phenomenon also appears at that location is high. Whereas, if we do not meet that threshold,
we will model that the probability that the phenomenon also appears as quite low (see section 4.5.2).

We will therefore supply a map with at least one phenomenon placed on it, and then used as the center for a 
gaussian process. The feature at each cell will be a function of the distance between the phenomenon and the measured cell.
Then we will add measurement noise on top of that when we take a measurement to get the measured value.

We will also be assuming, for our testing, that we are dealing with continuous measurements. This is what allows us to 
use the gaussian process assumptions listed prior.  

You can find the working draft at --> https://www.overleaf.com/5562241342qvdbbcvmnvsk#fb90e0 
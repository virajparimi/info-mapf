# Information Driven Multi-Agent Path Finding

Jake Olkin\*, [Viraj Parimi](https://people.csail.mit.edu/vparimi/)\*, [Brian Williams](https://www.csail.mit.edu/person/brian-williams)
Massachusetts Institute of Technology  
\*Equal contribution  
**[IROS 2024](https://iros2024-abudhabi.org/)**

**Project:** [info-mapf-mers.csail.mit.edu](https://info-mapf-mers.csail.mit.edu/) • **Paper:** [arXiv:2409.13065](https://arxiv.org/abs/2409.13065)

This repo is for the information driven MAPF problem, where, given a set of agents, some phenomenon of interest, and a map of the area, we compute the path over all agents that gathers the most information.

## Overview of Method

This method uses a simple A* style search, however the innovation lies in how we estimate the heuristic functions, and how we can exploit the structure of the problem to prune unnecessary paths.

Simply put, we want our path to maximize the information gain from the observations taken along the way.
A given node in the search tree represents a path prefix for each agent. We can use this to calculate the 
information gain of these partial paths. Then, we can create an optimistic heuristic for the maximum information gain from calculating the best information gain for each agent individually, and then just summing them together.

## Testing Information

In our tests, we will have a discritized grid for the agents to traverse, moving in the cardinal directions. To represent the relationship between the phenomenon and the measured features, we will use gaussian processes.

This makes the assumption that, when the mean value for a gaussian process at a given location is above a certain threshold, we will model that the probability that the phenomenon also appears at that location is high. Whereas, if we do not meet that threshold, we will model that the probability that the phenomenon also appears as quite low.

We will therefore supply a map with at least one phenomenon placed on it, and then use it as the center for a gaussian process. The feature at each cell will be a function of the distance between the phenomenon and the measured cell. Then we will add measurement noise on top of that when we take a measurement to get the measured value.

We will also be assuming, for our testing, that we are dealing with continuous measurements. This is what allows us to use the gaussian process assumptions listed prior.

## Requirements

```sh
conda env create -f environment.yml
```

## Experiments

### Small Examples

```sh
cd test
python test_rh_ma_vulcan.py --type <problem_type>
```

### MAPF Suite

```sh
cd test
python test_mapf_suite --map_type <map_type>
```

### Real World Datasets

```sh
cd test
python test_real_world_setup.py --dataset_name <name_of_dataset> --cell_size_degrees <discretization_factor>
```

Note: Information about the program options can be found in the corresponding files.

import matplotlib.pyplot as plt
from simple_multiagent_scenario import *
from astar import astar
from multi_agent_search import multi_agent_search

def check_maps(width, height, means, centers):
    reward_map = GaussianRewardMap(width, height, means, centers)
    im = plt.imshow(reward_map.grid, cmap='hot')
    plt.colorbar(im)
    for i in range(height):
        for j in range(width):
            text = plt.text(j, i, reward_map.grid[i, j],
                           ha="center", va="center", color="b")
    #plt.show()
    return  reward_map

def test_astar(map, init_pos, horizon):
    path = astar(map, init_pos, horizon)
    print(path)
    print(map.get_reward_from_traj([path]))


def action_model(act, init_pos):
    if act == "left":
        return [init_pos[0]-1, init_pos[1]]
    if act == "right":
        return [init_pos[0]+1, init_pos[1]]
    if act == "up":
        return [init_pos[0], init_pos[1]+1]
    if act == "down":
        return [init_pos[0], init_pos[1]-1]

agent_actions = ["left", "right", "up", "down"]
map = check_maps(5, 5, [2, 1], [[0, 0], [4, 4]])
test_astar(map, [3, 3], 3)
init_poses = [[1,1],[3,3]]

multi_path = multi_agent_search(agent_actions, map, init_poses, 5, 2, action_model)
print(multi_path)

plt.show()
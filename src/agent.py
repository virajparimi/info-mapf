# Default agent class with 4 cardinal deterministic actions

class Agent(object):
    def __init__(self, start_location):
        self.start = start_location
        self.observations = []
        self.actions = ["left", "right", "up", "down"]

    def take_action(self, state, action):
        assert(action in self.actions)
        
        if action == "left":
            return [state[0]-1, state[1]]
        elif action == "right":
            return [state[0]+1, state[1]]
        elif action == "up":
            return [state[0], state[1]+1]
        elif action == "down":
            return [state[0], state[1]-1]
        else:
            return state

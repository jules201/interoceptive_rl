# Grid world environment for homeostatic reinforcement learning

# 8x8 grid world with 4 possible actions (up, down, left, right)
import numpy as np

class GridWorld:
    # action -> (d_row, d_col); row 0 is the top, so "up" is -1
    ACTIONS = {0: (-1, 0),   # up
             1: (1, 0),    # down
             2: (0, -1),   # left
             3: (0, 1),    # right
             4: (0, 0)}    # wait

    def __init__(self, grid_size=8, 
                 n_stats=2,
                 spawn="random",
                 spawn_fixed=None,
                 resource_placement="corners", 
                 resource_fixed=None,
                 resample_resources=False,
                 episode_length=1000,
                 seed=None):
        self.grid_size = grid_size
        self.n_stats = n_stats
        self.spawn = spawn                      # "random" | "middle" | "fixed"
        self.spawn_fixed = spawn_fixed
        self.resource_placement = resource_placement  # "corners" | "random" | "fixed"
        self.resource_fixed = resource_fixed
        self.resample_resources = resample_resources  # new layout every episode?
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

        self.resource_pos = self._place_resources()
        self.reset()

    def _place_resources(self):
        n = self.grid_size
        if self.resource_placement == "corners":
            corners = [(0, 0), (n-1, n - 1), (n - 1, 0), (0, n - 1)]
            if self.n_stats > len(corners):
                raise ValueError("corners supports at most 4 stats")
            return np.array(corners[:self.n_stats])
        if self.resource_placement == "random":
            # distinct cells, so resources can't overlap
            idx = self.rng.choice(n * n, size=self.n_stats, replace=False)
            return np.array(np.unravel_index(idx, (n, n))).T   # shape (n_stats, 2)
        if self.resource_placement == "fixed":
            return np.array(self.resource_fixed)
        raise ValueError(f"unknown resource_placement: {self.resource_placement}")

    def _spawn_loc(self):
        n = self.grid_size
        if self.spawn == "random":
            return self.rng.integers(0, n, size=2)
        if self.spawn == "middle":
            return np.array([n // 2, n // 2])   # (4, 4) on an 8x8 grid
        if self.spawn == "fixed":
            return np.array(self.spawn_fixed)
        raise ValueError(f"unknown spawn: {self.spawn}")

    def reset(self):
        self.time_step = 0
        self.done = False
        if self.resample_resources:
            self.resource_pos = self._place_resources()
        self.agent_pos = self._spawn_loc()
        self.stats = np.zeros(self.n_stats)  # placeholder

    def _move(self, action):
        """
        move sthe agent 1 step in the env if possible (no wall)
        
        Returns new agent position"""
        new = self.agent_pos + np.array(self.ACTIONS[action])
        if np.all((new >= 0) & (new < self.grid_size)):
            self.agent_pos = new
        # otherwise: bump into the edge, stay in place

    def step(self, action):
        """ 
        after an action is chosen, this function should be called to move the agent 1 step (_move is called for that)
        Returns: agents new position, 
                on_resource (boolean describing whether the agent is at a resource)(on_resource.legth = stats) [false, True] woudl mean the agent is at the second resource
        """
        self._move(action)
        self.time_step += 1
        
        on_resource = np.all(self.agent_pos == self.resource_pos, axis=1)   # shape (n_stats,)
        done = self.time_step >= self.episode_length
        agent_pos_return = self.agent_pos.copy()  # return a copy to avoid external mutation
        return agent_pos_return, on_resource, done
import numpy as np
"""
I have a multiple agent

where each sub agent is 'responsible' for one stat. 
Each sub agent has its own drive function and reward function. and a value map is computed based on the rewards etc. 

"""

class Agent:
    def __init__(self,
                 env,
                 grid_size=5,
                 lr=0.1,
                 gamma = 0.99, #discount factor for future rewards
                 stat_decrease_per_step=np.array([0.02,0.02]), #how much the internal stats decrease per time step
                 stat_replenish_per_resource=np.array([0.4,0.4]),
                 m=2, #exponent for drive function
                 n=4, #exponent for drive function
                 n_stats=2, #number of stats to track 
                 set_point=1.0, #desired level of stat
                 n_actions = 5,
                 n_bins =20,
                 epsilon =0.5,
                 seed = None,
                 h_min = -1,
                 h_max = 3,
                 boundary_penalty = 0.01,
                 storable = [False, False],
                 storable_reward = 0.02 #reward for being above the set point for storable stats
                 




        ):
    
        self.env = env #grid environment
        self.lr = lr #learning rate
        self.gamma = gamma #discount factor for future rewards
        self.epsilon = epsilon 
        self.stat_decrease_per_step = stat_decrease_per_step
        self.stat_replenish_per_resource = stat_replenish_per_resource
        self.m = m
        self.n = n
        self.n_stats = n_stats
        self.set_point = set_point
        self.n_bins = n_bins
        self.grid_size, self.n_actions = grid_size, n_actions
        self.h_min = h_min
        self.h_max =   h_max

        self.rng = np.random.default_rng(seed)
        self.dims = (grid_size, grid_size) + (n_bins,)*n_stats
        self.n_states = int(np.prod(self.dims))
        self.Q = np.zeros((n_stats, self.n_states, n_actions))
        self.boundary_penalty = boundary_penalty
        self.dead = False
        self.storable = storable #list of booleans for each stat; if True the stat is storable and can be replenishedby collecting resources; if False the stat is not storable and cannot be replenished by collecting resources
        self.storable_reward = storable_reward #reward for being above the set point for storable stats



    def make_stats(self):
        self.stats = self.rng.uniform(self.set_point - 0.4, self.set_point + 0.4, size=self.n_stats)
        self.dead = False
        return self.stats

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    
    def select_action(self, location, stats, training= True):
        """
        argmax (sum of Q(state,action) for internal sat i)
        
        Equation 7 in Dulberg et"""
        if training and self.rng.random()<self.epsilon:
            return int(self.rng.integers(self.n_actions)) 
        state = self.encode(location, stats) #state is a combination of location(agent location) and internal stats
        q_total=self.Q[:,state].sum(axis=0) #sum over all stats/subagents

        best= np.flatnonzero(q_total ==q_total.max())
        
        return int(self.rng.choice(best))

    #TODO mach ich hier alle stats rein oder nur dass für das module/den einen agent? oder list an stats? aber dann brauch ich ja pro 
    # das is ja quatsch so ich will für den gesamten agent die stats updaten; aber dann pro sub module einen unterschiedlichen drive/reward berechnen
    #also new interna stats:; dann for i in sub agents(n_stats); compute drive and reward
    def organism_step(self, on_resource):
        """
        is called after the agent was moved in the environment to update the internal stats and compute reward
        on_resource boolean for collecting or not collecting reward for that resource || on_resource[i] for stat i 

        all stats are decreased each time step 
        and replenished if on_resource[i] = true 

        then I clip the new internal stats, so it stays in range.. 

        reward is positive for drive reduction (moving closer to set point) and negative for an increasing drive (further away from set point)
        """
        h_prev = self.stats.copy() #saving current stats to compute drive

        h_new = self.stats - self.stat_decrease_per_step 
        h_new = h_new + on_resource * self.stat_replenish_per_resource

        dead_per_stat = (h_new <=self.h_min) | (h_new >=self.h_max)
        self.dead = dead_per_stat.any()
        #h_new = np.clip(h_new, self.h_min, self.h_max) # If I clip here and the value is at the min or max the reward is zero because there is no drive reduction? or increasion??

        drive_prev = self.compute_drive(h_prev)
        drive_curr = self.compute_drive(h_new)



        rewards = drive_prev - drive_curr

        # If a stat is storable and the new value is above the set point, give a positive reward for that module. This encourages the agent to collect resources even if it is above the set point
        for i in range(self.n_stats):
            if self.storable[i] and h_new[i] > self.set_point:
                 rewards[i] = self.storable_reward  # Give a small positive reward for being above the set point for storable stats
               # rewards[i] = abs(rewards[i])   # Give a small positive reward for being above the set point for storable stats
               # if on_resource[i]:  # Only give a bonus if the drive is reduced
                #    rewards[i] = 0.04  # Give a bonus reward for reducing drive
               # else: 
                 #   rewards[i] = 0  

        #additional penalty for dead stats; if a stat is dead, the reward for that stat is reduced by the current drive value; this encourages the agent to avoid dying
        for i in range(self.n_stats):
            if dead_per_stat[i]:
                rewards[i] -= drive_curr[i]

        # Additional penalty for being stuck at a boundary
      #  stuck = (h_new == h_prev)
       # rewards[stuck] -= self.boundary_penalty

        self.stats = h_new #update stats variable 

        return h_prev, rewards, self.dead, dead_per_stat

    def compute_drive(self, h ):
        """
        quadratic drive function with m =4, n =2 
        absolute of (stats - set_point); the higher the bigger the drive 
        
        """
        
        drives = (np.abs(h - self.set_point) ** self.n  )

        return drives ** (1.0/self.m)

    def update_value_map(self, location, stats_old, stats_new, action, rewards, location_new, done):
        """
        takes self 
        agent location, 
        the old internal stats before taking a step
        new internal stats
        the action that was taken
        the rewards for that action
        new agent location

        max Q-learning algorithm 
        """
        
        state = self.encode(location, stats_old) 
        next_state = self.encode(location_new, stats_new) # here I encode the next state; the new agent ocation with the internal stats; but the new stats

        deltas = np.zeros(self.n_stats) #TD errors
        for i in range(self.n_stats):
            if done:
                target= rewards[i]
            else:
                target = rewards[i] + self.gamma * np.max(self.Q[i, next_state]) # np.max(qi) picks best action for  stat[i], next state
            deltas[i]= target - self.Q[i,state,action] # TD error
            self.Q[i, state, action] += self.lr *deltas[i]
        return deltas

   
    def encode(self, loc, h):
        #creates one number for each location; internal state combo; 
        bins = np.clip(((h - self.h_min) / (self.h_max - self.h_min) * self.n_bins).astype(int),
                    0, self.n_bins - 1)
        return int(np.ravel_multi_index((*loc, *bins), self.dims))



    def value_map(self, h_ref):
        v = np.zeros((self.n_stats, self.grid_size, self.grid_size))
        for i in range(self.n_stats):
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    v[i, x, y] = np.max(self.Q[i, self.encode((x, y), h_ref)])
        return v

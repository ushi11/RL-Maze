import numpy as np

# ==========================================
# PART 1: THE ENVIRONMENT (GridWorld Class)
# ==========================================
class GridWorld:
    def __init__(self, tot_row, tot_col):
        self.action_space_size = 4
        self.world_row = tot_row
        self.world_col = tot_col
        self.transition_matrix = np.ones((self.action_space_size, self.action_space_size))/self.action_space_size
        self.reward_matrix = np.zeros((tot_row, tot_col))
        self.state_matrix = np.zeros((tot_row, tot_col))
        self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]

    def setTransitionMatrix(self, transition_matrix):
        if(transition_matrix.shape != self.transition_matrix.shape):
            raise ValueError('The shape of the two matrices must be the same.')
        self.transition_matrix = transition_matrix

    def setRewardMatrix(self, reward_matrix):
        if(reward_matrix.shape != self.reward_matrix.shape):
            raise ValueError('The shape of the two matrices must be the same.')
        self.reward_matrix = reward_matrix

    def setStateMatrix(self, state_matrix):
        if(state_matrix.shape != self.state_matrix.shape):
            raise ValueError('The shape of the two matrices must be the same.')
        self.state_matrix = state_matrix

    def step(self, action):
        # 1. Determine the ACTUAL action based on the "Slippery" Transition Matrix
        # The transition matrix defines the probability that the agent moves 
        # in a different direction than intended.
        probabilities = self.transition_matrix[action, :]
        actual_action = np.random.choice(self.action_space_size, p=probabilities)

        # 2. Calculate new position based on actual_action
        # 0: Up, 1: Down, 2: Left, 3: Right
        new_position = list(self.position) # Copy current position
        if actual_action == 0: # Up
            if new_position[0] > 0: new_position[0] -= 1
        elif actual_action == 1: # Down
            if new_position[0] < self.world_row - 1: new_position[0] += 1
        elif actual_action == 2: # Left
            if new_position[1] > 0: new_position[1] -= 1
        elif actual_action == 3: # Right
            if new_position[1] < self.world_col - 1: new_position[1] += 1

        # Check if we hit a wall/obstacle (represented by -1 in state_matrix generally, 
        # but here logic is implicit based on grid bounds). 
        # Note: In this simple version, walls are just boundaries.
        
        self.position = new_position
        
        # 3. Get Reward and Check if Done
        reward = self.reward_matrix[self.position[0], self.position[1]]
        
        # Check if terminal state (Goal or Pit)
        # In the main script, goal is (0,3) and pit is (1,3)
        done = False
        if self.state_matrix[self.position[0], self.position[1]] != 0:
            done = True
            
        return self.position, reward, done

    def reset(self, exploring_starts=False):
        if exploring_starts:
            while True:
                row = np.random.randint(self.world_row)
                col = np.random.randint(self.world_col)
                # Don't start on a terminal state (Goal/Pit)
                if self.state_matrix[row, col] == 0:
                    self.position = [row, col]
                    break
        else:
            self.position = [self.world_row-1, 0] # Bottom left corner usually
        return self.position

# ==========================================
# PART 2: THE ALGORITHM (Actor-Critic)
# ==========================================

def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def update_critic(utility_matrix, observation, new_observation, 
                   reward, alpha, gamma, done):
    u = utility_matrix[observation[0], observation[1]]
    u_t1 = utility_matrix[new_observation[0], new_observation[1]]
    
    # If done, the value of next state is 0 (game over)
    if done: u_t1 = 0 
        
    delta = reward + ((gamma * u_t1) - u) #temporal difference error
    utility_matrix[observation[0], observation[1]] += alpha * delta
    return utility_matrix, delta

def update_actor(state_action_matrix, observation, action, delta):
    col = observation[1] + (observation[0]*4)
    state_action_matrix[action, col] += delta # Simple update
    return state_action_matrix 

def main():
    env = GridWorld(3, 4)

    # Define the state matrix (0=Empty, 1=Terminal/Goal, -1=Terminal/Pit)
    state_matrix = np.zeros((3,4))
    state_matrix[0, 3] = 1 # Goal
    state_matrix[1, 3] = 1 # Pit (treated as terminal)
    # state_matrix[1, 1] = -1 # Wall
    
    # Define the reward matrix
    reward_matrix = np.full((3,4), -0.04) # Step cost
    reward_matrix[0, 3] = 1.0  # Goal Reward
    reward_matrix[1, 3] = -1.0 # Pit Reward

    # Define the transition matrix (0.8 prob of success, 0.2 prob of slip)
    # [Up, Down, Left, Right]
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])

    state_action_matrix = np.random.random((4,12))  #policy matrix
    
    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    utility_matrix = np.zeros((3,4))

    gamma = 0.999
    
    # ==========================================
    # EXPERIMENT 1: CHANGING THIS VALUE
    # Try 0.001 (stable) vs 0.5 (unstable)
    # ==========================================
    alpha = 0.5 
    # ==========================================
    
    tot_epoch = 10000 
    print_epoch = 1000

    print("Starting training...")

    for epoch in range(tot_epoch):
        observation = env.reset(exploring_starts=True)
        for step in range(1000):
            col = observation[1] + (observation[0]*4)
            
            # Actor chooses action
            action_array = state_action_matrix[:, col]
            action_distribution = softmax(action_array)
            action = np.random.choice(4, 1, p=action_distribution)[0]

            # Environment steps
            new_observation, reward, done = env.step(action)

            # Critic learns
            utility_matrix, delta = update_critic(utility_matrix, observation, 
                                                  new_observation, reward, alpha, gamma, done)
            
            # Actor learns
            state_action_matrix = update_actor(state_action_matrix, observation, 
                                               action, delta)
            
            observation = new_observation
            if done: break
            
        if(epoch % print_epoch == 0):
            print(f"\nEpoch: {epoch}")
            print("Utility Matrix (Critic's Values):")
            print(utility_matrix)

    print("\nFinal Utility Matrix:")
    print(utility_matrix)

if __name__ == "__main__":
    main()
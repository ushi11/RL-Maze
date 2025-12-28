import os
import time
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


# ----------------------------
# Slippery Maze Environment (solvable random maze)
# ----------------------------
class MazeEnv:
    def __init__(self, size=30, wall_prob=0.18, slip_prob=0.2, seed=0):
        self.size = size
        self.wall_prob = float(wall_prob)
        self.slip_prob = float(slip_prob)
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.rng = np.random.default_rng(seed)

        self.walls = self._generate_solvable_walls()
        self.reset()

    def _neighbors(self, pos):
        x, y = pos
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < self.size and 0 <= ny < self.size:
                yield (nx, ny)

    def _is_solvable(self, walls):
        q = deque([self.start])
        seen = {self.start}
        while q:
            p = q.popleft()
            if p == self.goal:
                return True
            for nb in self._neighbors(p):
                if nb not in walls and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        return False

    def _generate_solvable_walls(self):
        while True:
            walls = set()
            for x in range(self.size):
                for y in range(self.size):
                    if (x, y) in (self.start, self.goal):
                        continue
                    if self.rng.random() < self.wall_prob:
                        walls.add((x, y))
            if self._is_solvable(walls):
                return walls

    def reset(self):
        self.agent_pos = self.start
        return self._get_state()

    def _get_state(self):
        x, y = self.agent_pos
        return np.array([x / self.size, y / self.size], dtype=np.float32)

    def step(self, intended_action):
        # slippery execution
        if random.random() < self.slip_prob:
            action = random.randint(0, 3)
        else:
            action = intended_action

        x, y = self.agent_pos
        if action == 0:      # up
            new_pos = (x - 1, y)
        elif action == 1:    # down
            new_pos = (x + 1, y)
        elif action == 2:    # left
            new_pos = (x, y - 1)
        else:                # right
            new_pos = (x, y + 1)

        if (0 <= new_pos[0] < self.size and
            0 <= new_pos[1] < self.size and
            new_pos not in self.walls):
            self.agent_pos = new_pos

        if self.agent_pos == self.goal:
            return self._get_state(), 1.0, True

        return self._get_state(), -0.01, False


# ----------------------------
# DQN model
# ----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=2, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Visual helpers
# ----------------------------
def moving_average(x, window=50):
    x = np.array(x, dtype=np.float32)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window, dtype=np.float32) / window, mode="valid")

def plot_maze(env, path=None, title="Maze"):
    grid = np.zeros((env.size, env.size), dtype=np.int32)
    for (x, y) in env.walls:
        grid[x, y] = 1

    plt.imshow(grid, cmap="gray_r", origin="upper")
    plt.xticks([]); plt.yticks([])
    plt.title(title)

    sx, sy = env.start
    gx, gy = env.goal
    plt.scatter([sy], [sx], marker="s", s=70, label="Start")
    plt.scatter([gy], [gx], marker="*", s=120, label="Goal")

    if path is not None:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(ys, xs, linewidth=2, label="Path")

    plt.legend(loc="best")

def greedy_run(env, policy_net, max_steps=4000):
    state = env.reset()
    path = [env.agent_pos]
    done = False
    steps = 0

    while not done and steps < max_steps:
        steps += 1
        with torch.no_grad():
            q_values = policy_net(torch.tensor(state))
            intended_action = q_values.argmax().item()
        state, _, done = env.step(intended_action)
        path.append(env.agent_pos)

    return path, (env.agent_pos == env.goal)


# ----------------------------
# Output directory
# ----------------------------
SAVE_DIR = os.path.join(os.path.dirname(__file__), "figures_30x30_slippery")
os.makedirs(SAVE_DIR, exist_ok=True)
print("Saving figures to:", SAVE_DIR)


# ----------------------------
# Training setup
# ----------------------------
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = MazeEnv(size=30, wall_prob=0.18, slip_prob=0.2, seed=seed)

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
replay_buffer = deque(maxlen=100000)

gamma = 0.99
batch_size = 128

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9985

target_update = 800
num_episodes = 3000
max_steps_per_episode = 1200

step_count = 0

ep_rewards = []
ep_steps = []
ep_success = []

early_path = None
late_path = None


# ----------------------------
# Save maze layout
# ----------------------------
plt.figure(figsize=(6, 6))
plot_maze(
    env,
    title=f"Slippery Maze Layout (30x30)\nwall_prob={env.wall_prob}, slip_prob={env.slip_prob}"
)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "maze_layout.png"), dpi=200)
plt.close()


# ----------------------------
# Training loop
# ----------------------------
start_time = time.time()

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_steps_per_episode:
        steps += 1
        step_count += 1

        if random.random() < epsilon:
            intended_action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = policy_net(torch.tensor(state))
                intended_action = q_values.argmax().item()

        next_state, reward, done = env.step(intended_action)
        replay_buffer.append((state, intended_action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.from_numpy(np.array(states, dtype=np.float32))
            next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            q_sa = policy_net(states).gather(1, actions).squeeze(1)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]

            targets = rewards + gamma * max_next_q * (1 - dones)
            loss = nn.MSELoss()(q_sa, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
            optimizer.step()

        if step_count % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    success = (env.agent_pos == env.goal)
    ep_rewards.append(total_reward)
    ep_steps.append(steps)
    ep_success.append(1 if success else 0)

    if early_path is None and episode == 500:
        early_path, _ = greedy_run(env, policy_net)

    if episode % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Episode {episode}, Reward {total_reward:.2f}, ε {epsilon:.3f}, "
              f"Steps {steps}, Success {success}, Time {elapsed:.1f}s")

late_path, late_success = greedy_run(env, policy_net)

print("Training finished.")


# ----------------------------
# Plots
# ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(ep_rewards, alpha=0.35)
ma = moving_average(ep_rewards)
plt.plot(range(len(ma)), ma, linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title("Reward vs Episode (30x30 slippery)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "reward_vs_episode.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(ep_steps, alpha=0.35)
ma_s = moving_average(ep_steps)
plt.plot(range(len(ma_s)), ma_s, linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Steps vs Episode (30x30 slippery)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "steps_vs_episode.png"), dpi=200)
plt.close()

window = 100
rolling_sr = [
    sum(ep_success[max(0, i - window + 1):i + 1]) /
    (i - max(0, i - window + 1) + 1)
    for i in range(len(ep_success))
]

plt.figure(figsize=(8, 5))
plt.plot(rolling_sr, linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Success rate")
plt.title(f"Rolling Success Rate (window={window})")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "success_rate.png"), dpi=200)
plt.close()

plt.figure(figsize=(6, 6))
plot_maze(env, path=late_path,
          title=f"Final Greedy Path (success={late_success}) (30x30 slippery)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "final_path.png"), dpi=200)
plt.close()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_maze(env, path=early_path, title="Early Greedy Path")
plt.subplot(1, 2, 2)
plot_maze(env, path=late_path, title="Late Greedy Path")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "early_vs_late_paths.png"), dpi=200)
plt.close()

print("Saved report figures to:", SAVE_DIR)

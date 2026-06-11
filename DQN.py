import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# ------------------------- 神经网络定义 -------------------------
class DQN(nn.Module):
    """简单的全连接Q网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)   # 输出每个动作的Q值

# ------------------------- 经验回放缓冲区 -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state),
                torch.LongTensor(action).unsqueeze(1),
                torch.FloatTensor(reward).unsqueeze(1),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)

# ------------------------- DQN Agent -------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_capacity=10000, batch_size=64, target_update=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_cnt = 0

        # 双重网络
        self.q_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 探索参数
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # 经验回放
        self.memory = ReplayBuffer(buffer_capacity)

    def choose_action(self, state, eval_mode=False):
        """epsilon-greedy动作选择"""
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        # 采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 当前Q值
        q_values = self.q_net(states).gather(1, actions)

        # 目标Q值
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # 损失和优化
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        self.update_cnt += 1
        if self.update_cnt % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # 探索率衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

# ------------------------- 训练主函数 -------------------------
def train_dqn(env_name='CartPole-v1', episodes=500, render=False):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        if (ep+1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {ep+1}, Avg Reward (last 50): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    return rewards_history

if __name__ == "__main__":
    rewards = train_dqn()
    plt.plot(rewards)
    plt.title("DQN Training Rewards")
    plt.show()
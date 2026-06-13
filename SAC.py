import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# ------------------------- 经验回放 -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state),
                torch.FloatTensor(action),
                torch.FloatTensor(reward).unsqueeze(1),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)

# ------------------------- 网络定义 -------------------------
class SoftQNetwork(nn.Module):
    """双Q网络之一"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class PolicyNetwork(nn.Module):
    """输出动作的均值和log标准差，使用重参数化采样"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=1.0):
        super().__init__()
        self.action_scale = action_scale
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)   # 限制标准差范围
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # 重参数化采样
        z = normal.rsample()
        action = torch.tanh(z) * self.action_scale
        # 计算对数概率（考虑tanh变换）
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# ------------------------- SAC Agent -------------------------
class SAC:
    def __init__(self, state_dim, action_dim, action_scale, lr=3e-4, gamma=0.99,
                 tau=0.005, alpha=0.2, buffer_capacity=1e6, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # 熵温度系数
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 网络: 两个Q网络，一个策略网络
        self.q1 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.q2 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.target_q1 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.target_q2 = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.policy = PolicyNetwork(state_dim, action_dim, action_scale=action_scale).to(self.device)

        # 拷贝目标网络
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # 优化器
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 经验池
        self.memory = ReplayBuffer(int(buffer_capacity))

        # 自动调节alpha（可选）
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if eval:
            with torch.no_grad():
                mean, _ = self.policy.forward(state)
                action = torch.tanh(mean) * self.policy.action_scale
            return action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action, _ = self.policy.sample(state)
            return action.cpu().numpy()[0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # 采样
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # ----- 更新Q网络 -----
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_state)
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward + (1 - done) * self.gamma * target_q

        q1_pred = self.q1(state, action)
        q2_pred = self.q2(state, action)
        q1_loss = F.mse_loss(q1_pred, target_value)
        q2_loss = F.mse_loss(q2_pred, target_value)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ----- 更新策略网络 -----
        new_action, log_prob = self.policy.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ----- 自适应alpha -----
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ----- 软更新目标网络 -----
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ------------------------- 训练示例（HalfCheetah）-------------------------
def train_sac(env_name='HalfCheetah-v4', episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = env.action_space.high[0]  # 假设对称

    agent = SAC(state_dim, action_dim, action_scale)

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_reward += reward
            if done:
                break
        if (ep+1) % 20 == 0:
            print(f"Episode {ep+1}, Reward: {ep_reward:.2f}, Alpha: {agent.alpha:.3f}")
    env.close()

if __name__ == "__main__":
    train_sac()
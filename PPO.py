import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# ------------------------- 网络架构 -------------------------
class RolloutBuffer:
    """存储一个episode的transition用于PPO更新"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]

class ActorCritic(nn.Module):
    """Actor输出连续动作的均值和标准差，Critic输出状态价值V(s)"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Actor头
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))  # 可学习的标准差
        # Critic头
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self):
        raise NotImplementedError

    def get_value(self, state):
        x = self.shared(state)
        return self.critic(x)

    def get_action_and_value(self, state, action=None):
        x = self.shared(state)
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(x)
        return action, logprob, entropy, value

# ------------------------- PPO Agent -------------------------
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, K_epochs=4, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, logprob, _, value = self.policy.get_action_and_value(state)
        self.buffer.states.append(state.cpu().numpy().squeeze())
        self.buffer.actions.append(action.cpu().numpy())
        self.buffer.logprobs.append(logprob.cpu().item())
        return action.cpu().numpy()[0]

    def update(self):
        # 计算GAE优势函数
        rewards = np.array(self.buffer.rewards)
        dones = np.array(self.buffer.dones)
        values = []
        with torch.no_grad():
            for s in self.buffer.states:
                s_tensor = torch.FloatTensor(s).unsqueeze(0).to(self.device)
                v = self.policy.get_value(s_tensor)
                values.append(v.cpu().item())
        values = np.array(values).flatten()

        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards)-1:
                next_value = 0
            else:
                next_value = values[t+1] * (1 - dones[t+1])
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换为tensor
        old_states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        old_actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(self.buffer.logprobs).to(self.device)
        old_advantages = torch.FloatTensor(advantages).to(self.device)
        old_returns = torch.FloatTensor(rewards + self.gamma * values * (1-dones)).to(self.device)

        # 多轮更新
        for _ in range(self.K_epochs):
            # 重新计算新策略下的logprob, entropy, value
            _, logprobs, entropy, values = self.policy.get_action_and_value(old_states, old_actions)
            # 计算比率
            ratio = torch.exp(logprobs - old_logprobs)
            # PPO clip损失
            surr1 = ratio * old_advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * old_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            # Critic损失（MSE）
            critic_loss = nn.MSELoss()(values.squeeze(), old_returns)
            # 熵奖励（鼓励探索）
            entropy_loss = -entropy.mean() * 0.01

            total_loss = actor_loss + critic_loss + entropy_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.buffer.clear()

# ------------------------- 训练示例（Pendulum连续控制）-------------------------
def train_ppo(env_name='Pendulum-v1', episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim)

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)
            state = next_state
            ep_reward += reward
            if done:
                agent.update()
                break
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {ep_reward:.2f}")
    env.close()

if __name__ == "__main__":
    train_ppo()
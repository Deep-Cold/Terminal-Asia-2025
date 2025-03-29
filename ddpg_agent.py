import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

# Replay buffer for experience replay.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# Actor network: maps state to a continuous action vector.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)  # action_dim should be 5.
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Tanh bounds outputs between -1 and 1.
        return torch.tanh(self.fc3(x))

# Critic network: estimates Q-value for a given state and action.
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DDPG Agent using an actor and a critic.
class DDPGAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.memory = ReplayMemory(10000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005  # For soft updates

    def select_action(self, state, noise_scale=0.1):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        noise = torch.randn_like(action) * noise_scale
        action = action + noise
        return action.clamp(-1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        if non_final_mask.sum() > 0:
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        else:
            non_final_next_states = torch.empty(0, self.state_dim, device=self.device)
        
        with torch.no_grad():
            next_actions = self.actor_target(non_final_next_states) if non_final_next_states.nelement() > 0 else torch.empty(0, self.action_dim, device=self.device)
            next_Q = torch.zeros(self.batch_size, device=self.device)
            if non_final_next_states.nelement() > 0:
                next_Q[non_final_mask] = self.critic_target(non_final_next_states, next_actions).squeeze(1)
            target_Q = reward_batch + self.gamma * next_Q
        current_Q = self.critic(state_batch, action_batch).squeeze(1)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
    
    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)

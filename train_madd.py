#!/usr/bin/env python
import os
import torch
import numpy as np
import random
import time
import math

from MADDPG import MADDPG  # Your provided MADDPG class
from maddpg_terminal_env import TerminalEnv

# ---------------------------
# Simple Replay Buffer for Multi-Agent MADDPG
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    def add(self, states, actions, rewards, next_states, done):
        self.buffer.append((states, actions, rewards, next_states, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones
    @property
    def size(self):
        return len(self.buffer)

def evaluate(env, maddpg, episode_length=40, n_episode=10):
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        states = env.reset()
        ep_return = np.zeros(len(env.agents))
        for t in range(episode_length):
            actions = maddpg.take_action(states, explore=False)
            states, rewards, done, info = env.step(actions)
            ep_return += np.array(rewards)
            if done:
                break
        returns += ep_return
    returns /= n_episode
    return returns.tolist()

# ---------------------------
# Main Training Loop
# ---------------------------
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Instantiate the Terminal environment.
    env = TerminalEnv(project_root, state_dim=424, max_turns=40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assume we have 3 agents.
    env.agents = [0, 1, 2]
    state_dims = [env.state_dim for _ in env.agents]
    # For example, the action dimensions for your three agents.
    action_dims = [12, 10, 5]  
    critic_input_dim = sum(state_dims) + sum(action_dims)
    hidden_dim = 256
    actor_lr = 1e-4
    critic_lr = 1e-3
    gamma = 0.95
    tau = 0.01

    maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim, gamma, tau)
    
    replay_buffer = ReplayBuffer(capacity=100000)
    batch_size = 1024
    minimal_size = 4000
    update_interval = 100
    total_steps = 0
    num_episodes = 5000
    episode_length = env.max_turns  # turns per episode
    return_list = []
    
    for ep in range(num_episodes):
        states = env.reset()  # initial observations for each agent
        ep_rewards = np.zeros(len(env.agents))
        for t in range(episode_length):
            # Get actions from the MADDPG agent.
            actions = maddpg.take_action(states, explore=True)
            # Wait for the engine (or algo_strategy) to write a new line to action.txt,
            # then get the observation, reward, done flag, and info.
            next_states, rewards, done, info = env.step(actions)
            replay_buffer.add(states, actions, rewards, next_states, done)
            states = next_states
            ep_rewards += np.array(rewards)
            total_steps += 1
            if replay_buffer.size >= minimal_size and total_steps % update_interval == 0:
                batch = replay_buffer.sample(batch_size)
                for a_i in range(len(env.agents)):
                    maddpg.update(batch, a_i)
                maddpg.update_all_targets()
            if done:
                break
        return_list.append(ep_rewards.tolist())
        if (ep + 1) % 100 == 0:
            eval_returns = evaluate(env, maddpg, episode_length=episode_length, n_episode=10)
            print(f"Episode {ep+1}, Eval Returns: {eval_returns}, Training Returns: {ep_rewards.tolist()}")
    
    # Save each agent's actor model.
    for i, agent in enumerate(maddpg.agents):
        torch.save(agent.actor.state_dict(), f"ddpg_actor_model_agent{i}.pth")
    print("Training complete. Models saved.")

if __name__ == "__main__":
    main()

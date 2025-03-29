import os
import torch
import numpy as np
from ddpg_agent import DDPGAgent
from ddpg_terminal_env import TerminalEnv
import random
import math

# --- Setup paths ---
# Assume train_ddpg.py is in "python-algo" and project_root is one level up.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
algo_path = os.path.join(project_root, "python-algo", "run.sh")

# --- Initialize the Environment ---
env = TerminalEnv(project_root, algo_path, algo_path)

# --- Initialize DDPG Agent ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = env.state_dim   # 2 (p1_points, p2_points)
action_dim = env.action_dim  # 5
agent = DDPGAgent(state_dim, action_dim, device)

num_episodes = 100

for i_episode in range(num_episodes):
    obs = env.reset()  # Initial observation.
    state = torch.tensor([obs], dtype=torch.float32, device=device)
    
    action = agent.select_action(state)
    next_obs, reward, done, info = env.step()
    if next_obs is None:
        next_obs = [0.0] * env.state_dim
    next_state = torch.tensor([next_obs], dtype=torch.float32, device=device)
    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
    
    agent.memory.push(state, action, next_state, reward_tensor)
    agent.optimize_model()
    
    print(f"Episode {i_episode} completed with reward {reward}")
    
torch.save(agent.actor.state_dict(), "ddpg_actor_model.pth")
print("Training complete. Model saved to ddpg_actor_model.pth")

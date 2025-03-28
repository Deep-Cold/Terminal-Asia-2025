import os
import torch
import numpy as np
from dqn_agent import DQNAgent
from terminal_env import TerminalEnv
import random
import math

# --- Setup paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
algo_path = os.path.join(project_root, "python-algo", "run.sh")

# --- Initialize the Environment ---
env = TerminalEnv(project_root, algo_path, algo_path)

# --- Initialize DQN Agent ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = env.state_dim 
output_dim = len(env.action_space) 

agent = DQNAgent(input_dim, output_dim, device)

num_episodes = 500

for i_episode in range(num_episodes):
    # Reset environment and get initial state.
    state = env.reset()
    state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
    
    # Agent selects an action.
    action = agent.select_action(state_tensor)
    
    # Run one full match with the chosen action.
    next_state, reward, done, info = env.step(action.item())
    
    if next_state is None:
        next_state = [0.0] * env.state_dim
    next_state_tensor = torch.tensor([next_state], dtype=torch.float32, device=device)
    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
    
    # Save the transition in replay memory.
    agent.memory.push(state_tensor, action, next_state_tensor, reward_tensor)
    
    # Optimize the model.
    agent.optimize_model()
    
    # Update target network every 10 episodes.
    if i_episode % 10 == 0:
        agent.update_target_network()
    
    print(f"Episode {i_episode} completed with reward {reward}")
    
# Save the trained model.
torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
print("Training complete. Model saved to dqn_model.pth")

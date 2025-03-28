# baseline_train.py
import torch
import numpy as np
import random
import math
from sys import maxsize
from dqn_agent import DQNAgent

# A simple dummy environment for baseline training.
class DummyEnv:
    def __init__(self, state_dim=10, action_space=[0, 1, 2], max_steps=50):
        self.state_dim = state_dim
        self.action_space = action_space
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None

    def reset(self):
        self.current_step = 0
        # Return an initial state vector (e.g. random values between 0 and 1)
        self.state = np.random.rand(self.state_dim)
        return self.state

    def step(self, action):
        self.current_step += 1
        # Simulate a state transition by generating a new random state.
        self.state = np.random.rand(self.state_dim)
        # Define a simple reward function:
        # For example, let action 0 be "good" (+1), action 1 neutral (0), action 2 "bad" (-1)
        if action == 0:
            reward = 1.0
        elif action == 1:
            reward = 0.0
        else:
            reward = 2.0
        # Episode is done after max_steps
        done = self.current_step >= self.max_steps
        return self.state, reward, done, {}

if __name__ == "__main__":
    # Initialize the dummy environment and DQN agent.
    env = DummyEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = env.state_dim
    output_dim = len(env.action_space)
    agent = DQNAgent(input_dim, output_dim, device)

    num_episodes = 500  # Increase if needed for better training

    for i_episode in range(num_episodes):
        state = env.reset()
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        total_reward = 0
        while True:
            # Select an action (returns a tensor)
            action_tensor = agent.select_action(state_tensor)
            action = action_tensor.item()
            
            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            next_state_tensor = torch.tensor([next_state], dtype=torch.float32, device=device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            
            # Save the transition in replay memory
            agent.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
            agent.optimize_model()
            
            state_tensor = next_state_tensor
            if done:
                break

        # Update the target network every 10 episodes.
        if i_episode % 10 == 0:
            agent.update_target_network()
        print(f"Episode {i_episode} completed with total reward {total_reward}")

    # Save the trained model weights.
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("Training complete. Model saved to dqn_model.pth")

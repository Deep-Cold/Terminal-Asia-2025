# terminal_env.py
import os
import glob
import json
import time

class TerminalEnv:
    """
    A Terminal environment that communicates via three files:
      - observation.txt: updated externally each turn with a JSON string containing "obs"
      - action.txt: our step() method appends the agents’ action vectors here (one line per agent)
      - rewards.txt: updated externally with a JSON string containing "reward" (and optionally "done")
      
    The environment maintains pointers to process each new line in FIFO order.
    """
    def __init__(self, project_root, state_dim=424, max_turns=40):
        self.project_root = project_root
        self.state_dim = state_dim
        self.max_turns = max_turns
        self.agents = [0, 1, 2]
        self.obs_file = os.path.join(self.project_root, "observation.txt")
        self.reward_file = os.path.join(self.project_root, "rewards.txt")
        self.action_file = os.path.join(self.project_root, "action.txt")
        self.obs_idx = 0
        self.rew_idx = 0
        self.action_idx = 0

    def reset(self):
        """
        Resets the environment for a new episode.
        Deletes the existing observation.txt, rewards.txt, and action.txt files (if any)
        and resets the internal pointers.
        Returns initial observations (here, placeholder zero vectors) for each agent.
        """
        for f in [self.obs_file, self.reward_file, self.action_file]:
            if os.path.exists(f):
                os.remove(f)
        self.obs_idx = 0
        self.rew_idx = 0
        self.action_idx = 0
        return [[0.0] * self.state_dim for _ in self.agents]

    def _wait_for_new_line(self, file_path, current_idx):
        """
        Blocks until file_path exists and contains more than current_idx lines.
        Returns the next unprocessed line.
        """
        while True:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    lines = f.readlines()
                if len(lines) > current_idx:
                    return lines[current_idx].strip()
            time.sleep(0.1)

    def _get_next_obs_line(self):
        line = self._wait_for_new_line(self.obs_file, self.obs_idx)
        self.obs_idx += 1
        return line

    def _get_next_reward_line(self):
        line = self._wait_for_new_line(self.reward_file, self.rew_idx)
        self.rew_idx += 1
        return line

    def _append_actions(self, actions):
        """
        Appends the provided actions to the action.txt file.
        Each action vector (one per agent) is written on its own line.
        """
        with open(self.action_file, "a") as f:
            for agent_action in actions:
                f.write(",".join(str(v) for v in agent_action) + "\n")
        self.action_idx += len(actions)

    def step(self, actions):
        """
        Processes one turn.
        
        Workflow:
          1. Appends the provided actions (a list of 3 action vectors) to action.txt.
          2. Waits for a new line in observation.txt, parses it as JSON to obtain the observation.
          3. Waits for a new line in rewards.txt, parses it as JSON to obtain the reward and done flag.
          
        Returns:
          - observations: a list of observation vectors (one per agent).
          - rewards: a list of reward values (one per agent).
          - done: Boolean flag (from the rewards file or if max_turns reached).
          - info: A dictionary with extra information (e.g. current turn number).
        """
        # Append the actions to action.txt.
        self._append_actions(actions)
        print("Appended actions to action.txt")
        
        # Wait for a new observation line.
        obs_line = self._get_next_obs_line()
        try:
            obs_data = json.loads(obs_line)
        except Exception as e:
            print("Error parsing observation line:", obs_line, e)
            obs_data = {}
        obs = obs_data.get("obs", [0.0] * self.state_dim)
        
        # Wait for a new reward line.
        rew_line = self._get_next_reward_line()
        try:
            rew_data = json.loads(rew_line)
        except Exception as e:
            print("Error parsing reward line:", rew_line, e)
            rew_data = {}
        done = rew_data.get("done", False)
        reward = rew_data.get("reward", 0)
        
        info = self.obs_idx
        return [obs for _ in self.agents], [reward for _ in self.agents], done, info

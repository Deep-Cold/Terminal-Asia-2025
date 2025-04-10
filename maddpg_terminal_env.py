# terminal_env.py
import os
import glob
import json
import time
import sys
import subprocess


class TerminalEnv:
    """
    A Terminal environment that communicates via three files:
      - observation.txt: updated externally each turn with a JSON string containing "obs"
      - action.txt: our step() method appends the agents’ action vectors here (one line per agent)
      - rewards.txt: updated externally with a JSON string containing "reward" (and optionally "done")
      
    The environment maintains pointers to process each new line in FIFO order.
    """
    def __init__(self, project_root, state_dim=424, max_turns=100):
        self.project_root = project_root
        self.state_dim = state_dim
        self.max_turns = max_turns
        self.agents = [0, 1, 2]
        self.obs_file = os.path.join(self.project_root, "python-algo", "observation.txt")
        self.reward_file = os.path.join(self.project_root, "python-algo", "reward.txt")
        self.action_file = os.path.join(self.project_root, "python-algo", "action.txt")
        self.obs_idx = 0
        self.rew_idx = 0
        self.action_idx = 0
        self.flag = 0
        self.algo1 = os.path.join(self.project_root, "python-algo", "run.sh")
        self.algo2 = os.path.join(self.project_root, "python-algo", "run1.sh")
        self.first = True

    def reset(self):
        """
        Resets the environment for a new episode.
        Deletes the existing observation.txt, rewards.txt, and action.txt files (if any)
        and resets the internal pointers.
        Returns initial observations (here, placeholder zero vectors) for each agent.
        """
        for f in [self.obs_file, self.reward_file, self.action_file]:
            with open(f, "w") as file:
                file.write("")
        self.obs_idx = 0
        self.rew_idx = 0
        self.action_idx = 0
        self.first = True
        self.flag = 0
        return [[0.0] * self.state_dim for _ in self.agents]

    def _wait_for_new_line(self, file_path, current_idx):
        """
        Blocks until file_path exists and contains more than current_idx lines.
        Returns the next unprocessed line.
        """
        time_start = time.time()
        while True:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    lines = f.readlines()
                if len(lines) > current_idx:
                    return lines[current_idx].strip()
            time.sleep(0.1)
            if time.time() - time_start > 5:
                print(f"Timeout waiting for new line in {file_path}.")
                break

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

    def run_single_game(self):
        """
        Executes the engine command to run a single game.
        The command is constructed using the project_root and algo1/algo2 paths.
        """
        cmd = f"cd {self.project_root} && java -jar engine.jar work {self.algo1} {self.algo2}"
        print("Executing command:", cmd)
        p = subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        p.daemon = 1
        print("Finished running match")
        return p

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
        reward = 0
        if self.first == True:
            self.first = False
        else:
            # Wait for a new reward line.
            rew_line = self._get_next_reward_line()
            try:
                rew_data = json.loads(rew_line)
            except Exception as e:
                print("Error parsing reward line:", rew_line, e)
                rew_data = {}
                self.flag += 1
            reward = rew_data.get("reward", 0)

        # Wait for a new observation line.
        obs_line = self._get_next_obs_line()
        try:
            obs_data = json.loads(obs_line)
        except Exception as e:
            print("Error parsing observation line:", obs_line, e)
            obs_data = {}
            self.flag += 1
        obs = obs_data.get("obs", [0.0] * self.state_dim)

        # Append the actions to action.txt.
        self._append_actions(actions)
        print("Appended actions to action.txt")
    
        if self.flag > 2:
            print("Error in the game. Ending the game.")
            done = True
            info = self.obs_idx
            return [obs for _ in self.agents], [reward for _ in self.agents], done, info

        done = False
        info = self.obs_idx
        return [obs for _ in self.agents], [reward for _ in self.agents], done, info

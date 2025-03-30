# terminal_env.py
import os
import glob
import json
import random
from queue import Queue, Empty
import threading
import time
import numpy as np

class TerminalEnv:
    """
    A Terminal environment that receives external observation and reward signals via
    an action file. Each new line in the file corresponds to one turn.
    
    This version processes turns in FIFO order and, additionally, writes a file
    "action_output.txt" that logs the actions passed to step(). Specifically, the
    first line in action_output.txt is actions[0] and the second is actions[1].
    
    Attributes:
      - state_dim: dimension of each observation vector (default 424)
      - max_turns: maximum turns per episode
      - turn: current turn counter
      - agents: list of agent indices (e.g., [0,1,2])
      - action_file: path to the file that the game engine/appends turn data
    """
    def __init__(self, project_root, state_dim=424, max_turns=40):
        self.state_dim = state_dim
        self.max_turns = max_turns
        self.project_root = project_root
        self.action_file = os.path.join(self.project_root, "action.txt")
        self.line_idx = 0
        self.agents = [0, 1, 2]

    def reset(self):
        """
        Resets the environment by deleting the old action file (if any) and resetting the line index.
        Returns initial observations (here, zero vectors) for each agent.
        """
        if os.path.exists(self.action_file):
            os.remove(self.action_file)
        self.line_idx = 0
        return [[0.0] * self.state_dim for _ in self.agents]
    
    def _wait_for_new_line(self):
        """
        Blocks until a new line is available in action.txt beyond self.line_idx.
        Returns that line.
        """
        while True:
            if os.path.exists(self.action_file):
                with open(self.action_file, "r") as f:
                    lines = f.readlines()
                if len(lines) > self.line_idx:
                    line = lines[self.line_idx].strip()
                    self.line_idx += 1
                    return line
            time.sleep(0.1)
    
    def step(self, actions):
        """
        Processes one turn.
        
        It waits for a new line in action.txt (each line should be a JSON string
        with keys "obs" and "reward"; the final line has "done": true).
        After reading the line, it writes the actions (a 2D list, one per agent)
        to a new file "action_output.txt", where line 1 is actions[0] and line 2 is actions[1].
        
        Returns:
          - observations: List of observation vectors (one per agent)
          - rewards: List of rewards (one per agent)
          - done: True if the JSON contains "done": true or max_turns reached
          - info: Dictionary with extra info (e.g. current turn number)
        """
        # Wait for the next line in action.txt.
        line = self._wait_for_new_line()
        try:
            data = json.loads(line)
        except Exception as e:
            print("Error parsing line:", line, e)
            data = {}
        done = data.get("done", False)
        obs = data.get("obs", [0.0] * self.state_dim)
        reward = data.get("reward", 0)
        
        # Write a new file "action_output.txt" with the first two agents' actions.
        output_file = os.path.join(self.project_root, "action_output.txt")
        try:
            with open(output_file, "w") as f:
                if len(actions) >= 2:
                    f.write(",".join(str(v) for v in actions[0]) + "\n")
                    f.write(",".join(str(v) for v in actions[1]))
                else:
                    f.write("Insufficient actions")
            print("Wrote action_output.txt")
        except Exception as e:
            print("Error writing action_output.txt:", e)
        
        info = {"turn": self.line_idx}
        return [obs for _ in self.agents], [reward for _ in self.agents], done, info

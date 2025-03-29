import os
import subprocess
import sys
import glob
import json
import torch

class TerminalEnv:
    """
    An environment that runs a full Terminal match as one episode.
    The agent's action is a continuous vector of dimension 5:
      - Component 0: Attack unit type (maps to {0,1,2})
      - Component 1: Attack location index (maps to an integer in [0,27])
      - Component 2: Defence unit type (maps to {0,1,2})
      - Component 3: Defence x coordinate (maps to integer in [0,27])
      - Component 4: Defence y coordinate (maps to integer in [0,13])
    
    The action is written to an action file ("action.txt") as:
        "attack_unit_type,attack_x,attack_y\n
         defence_unit_type,defence_x,defence_y"
    
    After running the engine, the replay is parsed to extract an observation vector:
      [p1_points, p2_points]
    and the reward is computed as p1_points - p2_points.
    """
    def __init__(self, project_root, algo1, algo2, replay_dir="replays"):
        self.project_root = project_root
        self.algo1 = algo1
        self.algo2 = algo2
        self.replay_dir = replay_dir
        # Observation: [p1_points, p2_points]
        self.state_dim = 2
        # Action: 5-dimensional.
        self.action_dim = 5

    def run_single_game(self):
        # Execute the engine command.
        cmd = f"cd {self.project_root} && java -jar engine.jar work {self.algo1} {self.algo2}"
        print("Executing command:", cmd)
        p = subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        p.daemon = 1
        p.wait()
        print("Finished running match")

    def reset(self):
        replay_path = os.path.join(self.project_root, self.replay_dir)
        old_replays = glob.glob(os.path.join(replay_path, "*.replay"))
        for f in old_replays:
            os.remove(f)
        return [0.0, 0.0]

    def extract_observation(self, replay_file):
        json_objects = []
        with open(replay_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        json_objects.append(obj)
                    except json.JSONDecodeError as e:
                        print("Error decoding line:", line, e)
        for obj in reversed(json_objects):
            if 'endStats' in obj:
                endStats = obj['endStats']
                p1 = endStats.get("player1", {})
                p2 = endStats.get("player2", {})
                p1_points = p1.get("points_scored", 0)
                p2_points = p2.get("points_scored", 0)
                return [p1_points, p2_points]
        return [0.0, 0.0]

    def step(self):
        self.run_single_game()
        
        replay_path = os.path.join(self.project_root, self.replay_dir)
        replay_files = glob.glob(os.path.join(replay_path, "*.replay"))
        if not replay_files:
            print("No replay files found. Something went wrong.")
            return None, 0, True, {}
        newest_replay = max(replay_files, key=os.path.getctime)
        
        observation = self.extract_observation(newest_replay)
        reward = observation[0] - observation[1]
        done = True
        info = {"replay_path": newest_replay}
        return observation, reward, done, info

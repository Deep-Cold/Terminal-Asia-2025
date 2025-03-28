import os
import subprocess
import sys
import glob
import json

class TerminalEnv:
    """
    An environment that runs a full Terminal match as one episode.
    The agent's action (an integer, e.g., 0, 1, 2) is passed to the engine.
    After the match, the replay file is parsed to compute a reward.
    """
    def __init__(self, project_root, algo1, algo2, replay_dir="replays"):
        """
        :param project_root: Absolute path to the project root folder.
                             (This folder contains 'scripts', 'python_algo', and 'replays'.)
        :param algo1: Path to the first algo's run file.
        :param algo2: Path to the second algo's run file.
        :param replay_dir: Folder name (inside project_root) where .replay files are stored.
        """
        self.project_root = project_root
        self.algo1 = algo1
        self.algo2 = algo2
        self.replay_dir = replay_dir

        # Define dummy state vector size and discrete actions.
        self.state_dim = 10
        self.action_space = [0, 1, 2]  # e.g., 0: stall, 1: demolisher, 2: scout

    def run_single_game(self, process_command):
        """
        Runs one match using the Terminal engine.
        The chosen action is passed to the engine as an extra parameter.
        (Ensure your engine/run files are set up to accept this parameter.)
        """
        print("Start run a match")
        p = subprocess.Popen(
            process_command,
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr
            )
        # daemon necessary so game shuts down if this script is shut down by user
        p.daemon = 1
        p.wait()
        print("Finished running match")


    def reset(self):
        """
        Resets the environment before an episode.
        Cleans old replays and returns an initial dummy state.
        """
        replay_path = os.path.join(self.project_root, self.replay_dir)
        old_replays = glob.glob(os.path.join(replay_path, "*.replay"))
        for f in old_replays:
            os.remove(f)
        # Return an initial dummy state vector.
        return [0.0] * self.state_dim

    def step(self, action):
        """
        Runs one full match (episode) with the given action.
        :param action: An integer (0, 1, or 2) representing the chosen strategy.
        :return: (observation, reward, done, info)
                 observation: Dummy observation (state vector)
                 reward: Computed reward (e.g., final health difference)
                 done: True, since one match is one episode.
                 info: Extra information (e.g., replay path)
        """
        # Run the match with the chosen action.
        self.run_single_game(f"cd {self.project_root} && java -jar engine.jar work {self.algo1} {self.algo2}")

        # Look for the newest replay file in project_root/replays.
        

        replay_path = os.path.join(self.project_root, self.replay_dir)
        print(replay_path)
        replay_files = glob.glob(os.path.join(replay_path, "*.replay"))
        print(replay_files)
        if not replay_files:
            print("No replay files found. Something went wrong.")
            return None, 0, True, {}

        newest_replay = max(replay_files, key=os.path.getctime)

        import json

        json_objects = []
        with open(newest_replay, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        obj = json.loads(line)
                        json_objects.append(obj)
                    except json.JSONDecodeError as e:
                        print("Error decoding line:", line, e)

        # Now, if one of these objects contains the key "endStats", extract points:
        for obj in json_objects:
            if 'endStats' in obj:
                player1_points = obj["endStats"]["player1"]["points_scored"]
                player2_points = obj["endStats"]["player2"]["points_scored"]
                print("Player 1 points_scored:", player1_points)
                print("Player 2 points_scored:", player2_points)


            # Compute reward. (Adjust this function as needed.)
        reward = player1_points - player2_points

        done = True
        observation = [0.0] * self.state_dim
        info = {"replay_path": newest_replay}
        return observation, reward, done, info

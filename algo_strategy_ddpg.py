import gamelib
import torch
import random
import math
import json
import os
from ddpg_agent import DDPGAgent  # updated DDPG agent

# Set up the device for PyTorch.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        from sys import maxsize
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write("Random seed: {}".format(seed))

        # Define state and action dimensions.
        # Here, we create a live observation vector with 10 features.
        self.state_dim = 10  
        self.action_dim = 5  # as defined for our DDPG model

        self.agent = DDPGAgent(self.state_dim, self.action_dim, device)

        # Load updated DDPG model.
        model_path = os.path.join(os.path.dirname(__file__), "ddpg_actor_model.pth")
        try:
            self.agent.actor.load_state_dict(torch.load(model_path, map_location=device))
            self.agent.actor.eval()
            gamelib.debug_write("Loaded DDPG model successfully from {}".format(model_path))
        except Exception as e:
            gamelib.debug_write("DDPG model not found or failed to load. Using default strategy. Error: {}".format(e))
        
        self.scored_on_locations = []

    def on_game_start(self, config):
        gamelib.debug_write("Configuring your custom DDPG strategy...")
        self.config = config

        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        self.scored_on_locations = []

    def get_index_mapping(self):
        """
        Builds and returns a mapping array for the diamond-shaped board.
        Each element in the returned list is a tuple (row, col) corresponding to the
        cell's location on a 28x28 board that is part of the 420 valid cells (the diamond).
        
        Mapping details:
        - For rows 0 to 13 (upper half):
        Valid columns run from (13 - row) to (14 + row) inclusive.
        For example, row 0 has cells at columns 13 and 14.
            -> mapping[0] = (0, 13)
            -> mapping[1] = (0, 14)
        Row 1 has cells at columns 12 to 15, and so on.
        
        - For rows 14 to 27 (lower half):
        Let mirror = row - 14.
        Valid columns run from mirror to (27 - mirror) inclusive.
        For example, row 14 (mirror=0) covers columns 0 to 27.
        
        The mapping array is ordered row-by-row.
        """
        mapping = []
        for row in range(28):
            if row <= 13:
                start_col = 13 - row
                end_col = 14 + row  # inclusive
            else:
                mirror = row - 14
                start_col = mirror
                end_col = 27 - mirror  # inclusive

            for col in range(start_col, end_col + 1):
                mapping.append((row, col))
                
        # Ensure that the mapping array has exactly 420 elements.
        assert len(mapping) == 420, f"Mapping length is {len(mapping)}, but expected 420."
        return mapping

    def get_state_vector(self, game_state):
        """
        Extracts a state vector from the current game state.
        """
        vector = []

        unit_max_health = {WALL: 55, SUPPORT: 10, TURRET:75}

        # return a vector of size 420 + 4
        mapping = self.get_index_mapping()
        for i in mapping:
            unit = game_state.game_map[i]
            unit_type = unit.unit_type if unit else None
            health = unit.health if unit else 0
            if unit_type == SUPPORT:
                vector.append(health)
            elif unit_type == TURRET:
                vector.append(health + unit_max_health[SUPPORT])
            elif unit_type == WALL:
                if unit.upgraded:
                    vector.append(health + unit_max_health[SUPPORT] + unit_max_health[TURRET] + unit_max_health[WALL])
                else:
                    vector.append(health + unit_max_health[SUPPORT] + unit_max_health[TURRET])
            else:
                vector.append(0)
            
        vector.append(game_state.get_resource(MP))
        vector.append(game_state.get_resource(SP))
        vector.append(game_state.my_health)
        vector.append(game_state.enemy_health)

        return vector

    def on_turn(self, turn_state):
        """
        Called every turn with the current turn's state.
        Extracts a live observation vector from the board state, uses the DDPG actor
        to select a 5-dimensional continuous action, maps that action into two commands:
        - Attack: defined by attack unit type and attack location (derived from an index 0-27)
        - Defence: defined by defence unit type, defence x, and defence y.
        The commands are written to an action file, the corresponding units are spawned,
        and then the turn is submitted.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write("Turn {} using DDPG strategy".format(game_state.turn_number))
        
        # Get live state vector.
        state_vector = self.get_state_vector(game_state)
        state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=device)
        
        # Select a 5-dimensional action.
        with torch.no_grad():
            action = self.agent.actor(state_tensor)
        action_values = action.squeeze(0).tolist()
        gamelib.debug_write("DDPG selected action: {}".format(action_values))
        
        # --- Attack Mapping ---
        # Component 0: Attack unit type.
        attack_unit_type = int(round(((action_values[0] + 1) / 2) * 2))
        attack_unit_type = max(0, min(2, attack_unit_type))
        
        # Component 1: Attack location index in [0,27].
        attack_index = int(round(((action_values[1] + 1) / 2) * 27))
        attack_index = max(0, min(27, attack_index))
        # Map the index to a spawn coordinate:
        if attack_index <= 13:
            ax = attack_index
            ay = 13 - attack_index
        else:
            ax = attack_index
            ay = attack_index - 14
        
        # --- Defence Mapping ---
        # Component 2: Defence unit type.
        defence_unit_type = int(round(((action_values[2] + 1) / 2) * 2))
        defence_unit_type = max(0, min(2, defence_unit_type))
        
        # Component 3: Defence x coordinate in [0,27].
        dx = int(round(((action_values[3] + 1) / 2) * 27))
        dx = max(0, min(27, dx))
        
        # Component 4: Defence y coordinate in [0,13].
        dy = int(round(((action_values[4] + 1) / 2) * 13))
        dy = max(0, min(13, dy))
        
        # Write two lines to the action file.
        action_file = os.path.join(os.path.dirname(__file__), "action.txt")
        try:
            with open(action_file, "w") as f:
                # First line: attack command.
                f.write(f"{attack_unit_type},{ax},{ay}\n")
                # Second line: defence command.
                f.write(f"{defence_unit_type},{dx},{dy}")
            gamelib.debug_write("Action written to file: {} -> Attack: {0},{1},{2} | Defence: {3},{4},{5}".format(
                action_file, attack_unit_type, ax, ay, defence_unit_type, dx, dy))
        except Exception as e:
            gamelib.debug_write("Error writing action file: " + str(e))
        
        # Spawn the corresponding units immediately.
        # For attack units:
        if attack_unit_type == 0:
            unit_attack = SCOUT
        elif attack_unit_type == 1:
            unit_attack = DEMOLISHER
        else:
            unit_attack = INTERCEPTOR
        gamelib.debug_write("Spawning attack unit {} at [{},{}]".format(unit_attack, ax, ay))
        game_state.attempt_spawn(unit_attack, [ax, ay])
        
        # For defence structures:
        if defence_unit_type == 0:
            unit_defence = WALL
        elif defence_unit_type == 1:
            unit_defence = TURRET
        else:
            unit_defence = SUPPORT
        gamelib.debug_write("Spawning defence unit {} at [{},{}]".format(unit_defence, dx, dy))
        game_state.attempt_spawn(unit_defence, [dx, dy])
        
        # Submit turn.
        game_state.submit_turn()


    def on_action_frame(self, turn_string):
        """
        Processes action frames for reactive defense.
        """
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))

    def detect_enemy_unit(self, game_state, unit_type=None, valid_x=None, valid_y=None):
        total_units = 0
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if (unit.player_index == 1 and 
                        (unit_type is None or unit.unit_type == unit_type) and
                        (valid_x is None or location[0] in valid_x) and
                        (valid_y is None or location[1] in valid_y)):
                        total_units += 1
        return total_units

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()

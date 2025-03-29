import gamelib
import torch
import random
import math
import json
import os
from MADDPG import MADDPG  # updated MADDPG agent

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
        self.attack_action_dim = 10  # as defined for our DDPG model
        self.turret_action_dim = 6  # as defined for our DDPG model
        self.wall_booster_action_dim = 13  # as defined for our DDPG model

        self.agent = MADDPG(, self.state_dim, self.action_dim, device)

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

    def get_state_vector(self, game_state):
        """
        Constructs a live observation vector from the current game state.
        
        Features (example):
          0. Normalized turn number (assuming a max of 100 turns)
          1. Normalized MP (e.g., MP/10)
          2. Normalized SP (e.g., SP/10)
          3. Count of your turrets (normalized by an assumed max, e.g., /10)
          4. Count of your walls (normalized, e.g., /20)
          5. Count of enemy turrets in front (y in [14,15]) normalized (/10)
          6. Count of enemy walls in front (y in [14,15]) normalized (/10)
          7. Count of enemy supports normalized (/10)
          8. Count of enemy demolishers normalized (/10)
          9. Bias term (1.0)
        Adjust these features and normalization factors as needed.
        """
        vector = []
        # 0. Normalized turn number.
        vector.append(game_state.turn_number / 100.0)
        
        # 1. Normalized MP.
        mp = game_state.get_resource(MP)
        vector.append(mp / 10.0)
        
        # 2. Normalized SP.
        sp = game_state.get_resource(SP)
        vector.append(sp / 10.0)
        
        # 3. Count of your turrets.
        my_turrets = sum(
            1 for loc in game_state.game_map
            if game_state.contains_stationary_unit(loc)
            for unit in game_state.game_map[loc]
            if unit.player_index == 0 and unit.unit_type == TURRET
        )
        vector.append(my_turrets / 10.0)
        
        # 4. Count of your walls.
        my_walls = sum(
            1 for loc in game_state.game_map
            if game_state.contains_stationary_unit(loc)
            for unit in game_state.game_map[loc]
            if unit.player_index == 0 and unit.unit_type == WALL
        )
        vector.append(my_walls / 20.0)
        
        # 5. Enemy turret count in front (y in [14,15]).
        enemy_turrets = self.detect_enemy_unit(game_state, unit_type=TURRET, valid_y=[14, 15])
        vector.append(enemy_turrets / 10.0)
        
        # 6. Enemy wall count in front (y in [14,15]).
        enemy_walls = sum(
            1 for loc in game_state.game_map
            if game_state.contains_stationary_unit(loc)
            for unit in game_state.game_map[loc]
            if unit.player_index == 1 and unit.unit_type == WALL and loc[1] in [14, 15]
        )
        vector.append(enemy_walls / 10.0)
        
        # 7. Enemy supports count.
        enemy_supports = self.detect_enemy_unit(game_state, unit_type=SUPPORT)
        vector.append(enemy_supports / 10.0)
        
        # 8. Enemy demolishers count.
        enemy_demolishers = self.detect_enemy_unit(game_state, unit_type=DEMOLISHER)
        vector.append(enemy_demolishers / 10.0)
        
        # 9. Bias term.
        vector.append(1.0)
        
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
            attack_action, turret_action, wall_booster_action = self.agent.actor(state_tensor)
        attack_action_values = attack_action.squeeze(0).tolist()
        turret_action_values = turret_action.squeeze(0).tolist()
        wall_booster_action_values = wall_booster_action.squeeze(0).tolist()
        gamelib.debug_write("DDPG selected action: {}".format(attack_action_values, turret_action_values, wall_booster_action_values))

        # --- Attack Mapping ---
        attack_regions = [[(3, 10), (4, 9)], 
                          [(5, 8), (6, 7)], 
                          [(7, 6), (8, 5)], 
                          [(9, 4), (10, 3)], 
                          [(11, 2), (12, 1)], 
                          [(13, 0), (14, 1)], 
                          [(15, 2), (16, 3)], 
                          [(17, 4), (18, 5)], 
                          [(19, 6), (20, 7)], 
                          [(21, 8), (22, 9)], 
                          [(23, 10), (24, 11)], 
                          [(25, 12), (26, 13)]]

        for i in range(10):
            attack_action_values[i] = max(min(attack_action_values[i], 1), 0)
            attack_unit_type = min(int(attack_action_values[i] * 4), 3)
            if attack_unit_type == 0:
                unit_attack = SCOUT
            elif attack_unit_type == 1:
                unit_attack = DEMOLISHER
            else:
                unit_attack = INTERCEPTOR
            attack_region = attack_regions[i]
            ax, ay = random.choice(attack_region)
            
            gamelib.debug_write("Spawning attack unit {} at [{},{}]".format(unit_attack, ax, ay))
            game_state.attempt_spawn(unit_attack, [ax, ay])


        # --- Turret Mapping ---
        turret_regions = [[],[]]



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

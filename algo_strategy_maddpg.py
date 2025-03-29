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
        self.state_dim = 424
        self.attack_action_dim = 10  # as defined for our DDPG model
        self.turret_action_dim = 6  # as defined for our DDPG model
        self.wall_action_dim = 13  # as defined for our DDPG model

        self.agent = MADDPG(, self.state_dim, self.action_dim, device)

        # Load updated MADDPG model.
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
            unit = game_state.game_map[i[0]][i[1]]
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
        print(len(vector))

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
            attack_action, turret_action, wall_action = self.agent.actor(state_tensor)
        attack_action_values = attack_action.squeeze(0).tolist()
        turret_action_values = turret_action.squeeze(0).tolist()
        wall_action_values = wall_action.squeeze(0).tolist()
        gamelib.debug_write("MADDPG selected action: {}".format(attack_action_values, turret_action_values, wall_action_values))

        # --- Attack Mapping ---
        attack_regions = [[(3, 10), (4, 9), (5, 8), (6, 7)], 
                          [(7, 6), (8, 5), (9, 4)], 
                          [(10, 3), (11, 2), (12, 1), (13, 0)], 
                          [(14, 0), (15, 1), (16, 2), (17, 3)], 
                          [(18, 4), (19, 5), (20, 6)], 
                          [(21, 7), (22, 8), (23, 9), (24, 10)]]
        
        total_unit_number_cost = 0

        MP_this_turn = game_state.get_resource(MP)
        
        for i in range(6):
            unit_number_index = 2 * i + 1
            attack_action_values[unit_number_index] = max(min(attack_action_values[unit_number_index], 1), 0)
            total_unit_number_cost += attack_action_values[unit_number_index]
        
        MP_per_unit_cost = MP_this_turn / total_unit_number_cost 

        for i in range(6):
            unit_type_index = 2 * i
            unit_number_index = unit_type_index + 1
            attack_action_values[unit_type_index] = max(min(attack_action_values[unit_type_index], 1), 0)
            attack_unit_type = min(int(attack_action_values[unit_type_index] * 4), 3)
            if attack_unit_type == 1:
                unit_attack = SCOUT
            elif attack_unit_type == 2:
                unit_attack = DEMOLISHER
            elif attack_unit_type == 3:
                unit_attack = INTERCEPTOR
            else:
                continue
            attack_region = attack_regions[i]
            ax, ay = random.choice(attack_region)

            attack_action_values[unit_number_index] = max(min(attack_action_values[unit_number_index], 1), 0)
            number_of_units = int((attack_action_values[unit_number_index] * MP_per_unit_cost) // unit_attack["cost"][1])
            
            gamelib.debug_write("Spawning attack unit {} at [{},{}]".format(unit_attack, ax, ay))
            for _ in range(number_of_units):
                game_state.attempt_spawn(unit_attack, [ax, ay])


        # --- Turret Mapping ---
                # --- Turret Mapping ---
        turret_regions = [[(1, 12), (2, 12), (3, 12), (4, 12), (2, 11), (3, 11), (4, 11)],
                          [(5, 10), (6, 10), (7, 10), (6, 9), (7, 9), (7, 8)],
                          [(8, 7), (9, 7), (10, 7), (9, 6), (10, 6), (10, 5)],
                          [(11, 4), (12, 4), (13, 4), (12, 3), (13, 3), (13, 2)],
                          [(14, 4), (15, 4), (16, 4), (14, 3), (15, 3), (14, 2)],
                          [(17, 7), (18, 7), (19, 7), (17, 6), (18, 6), (17, 5)],
                          [(20, 10), (21, 10), (22, 10), (20, 9), (21, 9), (20, 8)],
                          [(23, 12), (24, 12), (25, 12), (26, 12), (23, 11), (24, 11), (25, 11)]]
        
        wall_regions = [[(0, 13), (1, 13), (2, 13), (3, 13), (4, 13)],
                        [(5, 12), (5, 11), (6, 11), (7, 11)],
                        [(8, 10), (8, 9), (8, 8), (9, 8), (10, 8)],
                        [(11, 7), (11, 6), (11, 5), (12, 5), (13, 5)],
                        [(14, 5), (15, 5), (16, 5), (16, 6), (16, 7)],
                        [(17, 8), (18, 8), (19, 8), (19, 9), (19, 10)],
                        [(20, 11), (21, 11), (22, 11), (22, 12)],
                        [(23, 13), (24, 13), (25, 13), (26, 13), (27, 13)]]
        
        total_value = 0

        SP_this_tern = game_state.get_resource(SP)

        for i in range(8):
            index = i * 2
            turret_action_values[index] = max(min(turret_action_values[index], 1), 0)
            turret_action_values[index + 1] = max(min(turret_action_values[index + 1]), 0)
            wall_action_values[index] = max(min(wall_action_values[index], 1), 0)
            wall_action_values[index + 1] = max(min(wall_action_values[index + 1], 1), 0)
            total_value += turret_action_values[index] * 4 + turret_action_values[index + 1] * 6
            total_value += wall_action[index] * 3 + wall_action[index + 1] * 2

        MP_per_unit_cost = SP_this_tern / total_value

        for i in range(8):
            index = i * 2
            number_of_units = int((turret_action_values[index] * MP_per_unit_cost))
            for (ax, ay) in turret_regions[i]:
                if number_of_units <= 0:
                    break
                number_of_units -= game_state.attempt_spawn(TURRET, [ax, ay])
            
            number_of_updates = int((turret_action_values[index + 1] * MP_per_unit_cost))
            for (ax, ay) in turret_regions[i]:
                if number_of_updates <= 0:
                    break
                number_of_updates -= game_state.attempt_upgrade([ax, ay])
            
        for i in range(8):
            index = i * 2
            number_of_units = int((wall_action_values[index] * MP_per_unit_cost))
            for (ax, ay) in wall_regions[i]:
                if number_of_units <= 0:
                    break
                number_of_units -= game_state.attempt_spawn(WALL, [ax, ay])
            
            number_of_updates = int((wall_action[index + 1] * MP_per_unit_cost))
            for (ax, ay) in wall_regions[i]:
                if number_of_updates <= 0:
                    break
                number_of_updates -= game_state.attempt_upgrade([ax, ay])



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

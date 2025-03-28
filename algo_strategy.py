# algo_strategy.py
import gamelib
import random
import math
import json
import torch
from dqn_agent import DQNAgent
import random
from sys import maxsize
import os

# If you haven't defined these globally yet:
MP = 1
SP = 0

# Set up the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

        # We'll define a 10-dimensional state vector and 3 discrete actions.
        self.state_dim = 10
        self.action_space = 3

        # Initialize the DQN agent (no training done here).
        self.agent = DQNAgent(self.state_dim, self.action_space, device)

        # Attempt to load a pre-trained model
        model_path = os.path.join(os.path.dirname(__file__), "dqn_model.pth")
        try:
            self.agent.policy_net.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            self.agent.policy_net.eval()
            gamelib.debug_write("Loaded DQN model successfully.")
        except Exception as e:
            gamelib.debug_write(
                f"DQN model not found or failed to load. Using default strategy. Error: {e}"
            )
        # The rest of your original code
        self.scored_on_locations = []

    def on_game_start(self, config):
        gamelib.debug_write("Configuring your custom DQN strategy...")
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
        Extract a 10-dimensional feature vector from the current game state.
        """
        vector = []
        # Feature 0: normalized turn number
        vector.append(game_state.turn_number / 100.0)

        # Feature 1: normalized MP resource
        vector.append(game_state.get_resource(MP) / 10.0)

        # Feature 2: normalized SP resource
        vector.append(game_state.get_resource(SP) / 10.0)

        # Feature 3: count of your turrets (normalized)
        my_turrets = sum(
            1
            for loc in game_state.game_map
            if game_state.contains_stationary_unit(loc)
            for unit in game_state.game_map[loc]
            if unit.player_index == 0 and unit.unit_type == TURRET
        )
        vector.append(my_turrets / 10.0)

        # Feature 4: count of your walls (normalized)
        my_walls = sum(
            1
            for loc in game_state.game_map
            if game_state.contains_stationary_unit(loc)
            for unit in game_state.game_map[loc]
            if unit.player_index == 0 and unit.unit_type == WALL
        )
        vector.append(my_walls / 20.0)

        # Feature 5: enemy turret count in the front (y in [14,15])
        enemy_turrets = self.detect_enemy_unit(
            game_state, unit_type=TURRET, valid_y=[14, 15]
        )
        vector.append(enemy_turrets / 10.0)

        # Feature 6: enemy wall count in the front (y in [14,15])
        enemy_walls = sum(
            1
            for loc in game_state.game_map
            if game_state.contains_stationary_unit(loc)
            for unit in game_state.game_map[loc]
            if unit.player_index == 1 and unit.unit_type == WALL and loc[1] in [14, 15]
        )
        vector.append(enemy_walls / 10.0)

        # Feature 7: enemy supports (assume max 10)
        enemy_supports = self.detect_enemy_unit(game_state, unit_type=SUPPORT)
        vector.append(enemy_supports / 10.0)

        # Feature 8: enemy demolishers (assume max 10)
        enemy_demolishers = self.detect_enemy_unit(game_state, unit_type=DEMOLISHER)
        vector.append(enemy_demolishers / 10.0)

        # Feature 9: bias term
        vector.append(1.0)

        return vector

    def on_turn(self, turn_state):
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write(f"Turn {game_state.turn_number} using DQN strategy")

        # Always ensure basic defenses are built
        self.build_dynamic_defences(game_state)
        self.build_reactive_defense(game_state)

        # Extract state features and convert to a tensor for the DQN
        state_vector = self.get_state_vector(game_state)
        state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=device)

        # Use the policy network to select an action
        with torch.no_grad():
            action = self.agent.policy_net(state_tensor).max(1)[1].item()

        gamelib.debug_write(f"DQN selected action: {action}")

        # Map the selected action to a strategy:
        # 0: Stall with interceptors, 1: Demolisher line attack, 2: Scout attack
        if action == 0:
            self.stall_with_interceptors(game_state)
        elif action == 1:
            self.demolisher_line_strategy(game_state)
        elif action == 2:
            # For the scout attack, only deploy scouts on every other turn.
            if game_state.turn_number % 2 == 1:
                scout_spawn_location_options = [[13, 0], [14, 0]]
                best_location = self.least_damage_spawn_location(
                    game_state, scout_spawn_location_options
                )
                game_state.attempt_spawn(SCOUT, best_location, 1000)

        game_state.submit_turn()

    def build_defences(self, game_state):
        turret_locations = [[0, 13], [27, 13], [8, 11], [19, 11], [13, 11], [14, 11]]
        game_state.attempt_spawn(TURRET, turret_locations)
        wall_locations = [[8, 12], [19, 12], [1, 13], [2, 13], [3, 13], [4, 13], [5, 13], [6, 13], [7, 13], [9, 13], [11, 13], [12, 13], [13, 13], [14, 13], [15, 13], [16, 13], [17, 13], [18, 13], [20, 13], [21, 13], [22, 13], [23, 13], [24, 13], [25, 13], [26, 13]]
        game_state.attempt_spawn(WALL, wall_locations)
        game_state.attempt_upgrade(wall_locations)


    def build_dynamic_turrets(self, game_state):
        """
        Dynamically place turrets where our defense is weakest.
        Candidate positions are generated within a defined range in our half of the map (y from 0 to 13).
        Each candidate's defense score is computed based on how many friendly turrets
        are nearby. Positions with lower scores are considered weak.
        """
        # Optionally, call your base defence routine first if desired.
        self.build_defences(game_state)

        # Only build dynamic turrets if we have sufficient SP.
        if game_state.get_resource(SP) > 30:
            # Define candidate positions: x from 3 to 24, and y from 0 to 13 (our half).
            candidate_positions = []
            for x in range(3, 25):
                for y in range(0, 14):
                    candidate_positions.append([x, y])
            
            def defense_score(pos):
                score = 0
                # Examine a 3x3 block around the candidate position.
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        neighbor = (pos[0] + dx, pos[1] + dy)
                        if game_state.contains_stationary_unit(neighbor):
                            # Use membership check instead of .get()
                            if neighbor in game_state.game_map:
                                units = game_state.game_map[neighbor]
                            else:
                                units = []
                            for unit in units:
                                if unit.player_index == 0 and unit.unit_type == TURRET:
                                    score += 1
                return score
            
            # Sort candidates by defense score (lower means fewer turrets nearby).
            candidate_positions.sort(key=lambda pos: defense_score(pos))
            # Select a number of positions (e.g., 5) to place turrets.
            best_positions = candidate_positions[:5]
            gamelib.debug_write("Dynamic turret positions chosen: " + str(best_positions))
            game_state.attempt_spawn(TURRET, best_positions)


    def build_dynamic_defences(self, game_state):
        """
        Build the overall dynamic defenses.
        This method first places walls to block enemy paths and then places turrets
        to cover weak points.
        """
        self.build_dynamic_turrets(game_state)



    def build_reactive_defense(self, game_state):
        for location in self.scored_on_locations:
            build_location = [location[0], location[1] + 1]
            game_state.attempt_spawn(TURRET, build_location)

    def stall_with_interceptors(self, game_state):
        friendly_edges = (
            game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT)
            + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
        )
        deploy_locations = self.filter_blocked_locations(friendly_edges, game_state)
        while (
            game_state.get_resource(MP) >= game_state.type_cost(INTERCEPTOR)[MP]
            and len(deploy_locations) > 0
        ):
            deploy_index = random.randint(0, len(deploy_locations) - 1)
            deploy_location = deploy_locations[deploy_index]
            game_state.attempt_spawn(INTERCEPTOR, deploy_location)

    def demolisher_line_strategy(self, game_state):
        stationary_units = [WALL, TURRET, SUPPORT]
        cheapest_unit = WALL
        for unit in stationary_units:
            unit_class = gamelib.GameUnit(unit, game_state.config)
            if unit_class.cost[game_state.MP] < gamelib.GameUnit(
                cheapest_unit, game_state.config
            ).cost[game_state.MP]:
                cheapest_unit = unit

        for x in range(27, 5, -1):
            game_state.attempt_spawn(cheapest_unit, [x, 11])
        game_state.attempt_spawn(DEMOLISHER, [24, 10], 1000)

    def least_damage_spawn_location(self, game_state, location_options):
        damages = []
        # Evaluate damage risk for each location
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            if path is None:
                # If no path is found, assign a very high damage value
                damage = float('inf')
            else:
                damage = 0
                for path_location in path:
                    attackers = game_state.get_attackers(path_location, 0)
                    turret_damage = gamelib.GameUnit(TURRET, game_state.config).damage_i
                    damage += len(attackers) * turret_damage
            damages.append(damage)
        # Return the location corresponding to the minimal damage
        return location_options[damages.index(min(damages))]


    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for loc in locations:
            if not game_state.contains_stationary_unit(loc):
                filtered.append(loc)
        return filtered

    def detect_enemy_unit(self, game_state, unit_type=None, valid_x=None, valid_y=None):
        total_units = 0
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if (
                        unit.player_index == 1
                        and (unit_type is None or unit.unit_type == unit_type)
                        and (valid_x is None or location[0] in valid_x)
                        and (valid_y is None or location[1] in valid_y)
                    ):
                        total_units += 1
        return total_units

    def on_action_frame(self, turn_string):
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            if not unit_owner_self:
                gamelib.debug_write(f"Got scored on at: {location}")
                self.scored_on_locations.append(location)
                gamelib.debug_write(f"All locations: {self.scored_on_locations}")

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()

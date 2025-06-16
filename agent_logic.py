
import numpy as np
import os

from constants import *

SCORE_SCALE = 5.0    
ENERGY_SCALE = -0.001
DIST_SCALE = -0.2

class Agent:
    def __init__(self):
        """Initialize the agent with weights and step counter from checkpoint or defaults."""
        reset_flag = os.environ.get("FORCE_RESET", "0") == "1"
        self.weights, self.step_counter = self.load_checkpoint()
        if reset_flag:
            self.weights = self.initialize_weights()
            self.step_counter = 0
        # Previous variables for Q-learning update
        self.prev_features = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_game_over = None

        # Current variables for Q-value calculation
        self.current_features = None
        self.current_action = None
        self.max_q_value = None
    @staticmethod  
    def nearest_loot_distance(local_map):
        """Manhattan distance from agent (center of 9×9 view) to closest gold or loot container."""
        targets = {STONE_GOLD, DEEPSLATE_GOLD, CHEST, BARREL}
        ys, xs = np.where(np.isin(local_map, list(targets)))
        if xs.size == 0:
            return 0                
        dists = np.abs(xs - 4) + np.abs(ys - 4)  
        return dists.min()
        #USE manhat to get closest 

    def initialize_weights(self):
        """
        Initialize weights randomly for linear Q-learning.
        Shape: [VIEW_WIDTH * VIEW_HEIGHT * NUM_BLOCK_TYPES + 16 (energy) + 16 (score) + 32 (gold) + 1, NUM_ACTIONS]
        """
        return np.random.randn(FEATURE_DIM, NUM_ACTIONS) * 0.01

    def load_checkpoint(self):
        """
        Load weights and step counter from checkpoint.
        Returns default values if no checkpoint exists or loading fails.
        """
        if os.path.exists(CHECKPOINT_PATH):
            try:
                checkpoint = np.load(CHECKPOINT_PATH, allow_pickle=False)
                print(f"Successfully loaded checkpoint, step: {checkpoint['step_counter']}")
                weights = checkpoint.get('weights')
                if weights is None:
                    weights = self.initialize_weights()
                return weights, int(checkpoint['step_counter'])
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
        # If checkpoint does not exist or fails, initialize weights
        return self.initialize_weights(), 0

    def save_checkpoint(self, force=False):
        """
        Save weights and step counter to checkpoint file.
        
        Args:
            force: If True, save regardless of step counter
        """
        if force or self.step_counter % SAVE_INTERVAL == 0:
            try:
                np.savez(CHECKPOINT_PATH, weights=self.weights, step_counter=self.step_counter)
                print(f"Checkpoint saved successfully, step: {self.step_counter}")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")

    def extract_features(self, local_map, position, energy, score, gold_count):
        """
        Convert state to one-hot encoded feature vector.
        
        Args:
            local_map: 9x9 map view
            position: Agent's position in the map
            energy: Current energy value
            score: Current score value
            gold_count: Dictionary with counts of gold in each direction, keys=WASD
        Returns:
            Flattened feature vector of length W*H*T + 16 (energy) + 16 (score) + 32 (gold count) + 1
        """
        # TODO: Explore more effective feature design - still need to experiement with this
        features = np.zeros(FEATURE_DIM, dtype=int)
        # Constant feature
        features[-1] = 1
        
        # Process the map features (block types)
        indices = NUM_BLOCK_TYPES * np.arange(VIEW_HEIGHT * VIEW_WIDTH)
        indices += local_map.flat
        features[indices] = 1

        # Add energy as 16-bit binary representation
        energy_int = int(min(energy, 2**16 - 1))  # Cap at max 16-bit value
        energy_binary = format(energy_int, '016b')  # Convert to 16-bit binary string
        for i in range(16):
            if energy_binary[i] == '1':
                features[BASE_FEATURE_DIM + i] = 1
        
        # Add score as 16-bit binary representation
        score_int = int(min(score, 2**16 - 1))  # Cap at max 16-bit value
        score_binary = format(score_int, '016b')  # Convert to 16-bit binary string
        for i in range(16):
            if score_binary[i] == '1':
                features[BASE_FEATURE_DIM + 16 + i] = 1
        
        # Add gold_count as 32-bit binary representation (8 bits per direction)
        directions = [('W', 0), ('A', 8), ('S', 16), ('D', 24)]  # Direction and bit offset
        for direction, offset in directions:
            assert direction in gold_count
            # Cap at max 8-bit value
            gold_val = int(min(gold_count[direction], 2**8 - 1))
            gold_binary = format(gold_val, '08b')  # Convert to 8-bit binary string
            for i in range(8):
                if gold_binary[i] == '1':
                    features[BASE_FEATURE_DIM + 16 + 16 + offset + i] = 1
        
        return features

    def calculate_q_value(self, features, action):
        """
        Calculate Q-value using linear approximation.
        """
        # TODO: Implement this #DONE
        return float(np.dot(features, self.weights[:, action]))

    def get_epsilon(self):
        """
        Linearly decay epsilon from EPSILON_START to EPSILON_END based on step counter.
        """
        decay_rate = (EPSILON_START - EPSILON_END) / DECAY_STEPS
        epsilon = max(EPSILON_END, EPSILON_START - decay_rate * self.step_counter)
        return epsilon
        # τ = 60000  # decay time constant; tweak between 30k–100k
        # return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-self.step_counter / τ)

    def agent_logic(self, local_map, position, energy, score, gold_count, training):
        """
        Main logic for the agent.
        Args:
            local_map: 9x9 map view
            position: Agent's position in the map
            energy: Current energy value
            score: Current score value
            gold_count: Dictionary with counts of gold in each direction {'W': w, 'A': a, 'S': s, 'D': d}
            training: Boolean indicating if in training mode
        Returns:
            action (str): 'W', 'A', 'S', 'D', or 'I'
        """
        # Step 1: Feature extraction
        self.current_features = self.extract_features(local_map, position, energy, score, gold_count)
        # Step 2: Q-value calculation
        # TODO: Implement this - DONE
        q_vec = self.current_features @ self.weights 
        self.curr_dist = self.nearest_loot_distance(local_map)   
        self.max_q_value = q_vec.max()

        # Step 3: Epsilon-greedy action selection with decaying epsilon. You might modify the epsilon calculation by changing the constant variables in constants.py to achieve better performance
        epsilon = self.get_epsilon()
        # TODO: Implement this - DONE
        if training and np.random.rand() < epsilon:
            self.current_action = np.random.randint(NUM_ACTIONS)
        else:
            self.current_action = int(np.argmax(q_vec))
    
        # self.current_action = 4
        #comment out ^^ bc its a constant var
        # action_char = ACTION_TO_CHAR[self.current_action]
        # return action_char
        return ACTION_TO_CHAR[self.current_action]

    def update_q_learning(self, delta_energy, delta_score, game_over):
        """
        Update Q-learning weights based on the current state and action.
        Args:
            delta_energy: Change in energy
            delta_score: Change in score
            game_over: Boolean indicating if the game is over
        """
        if self.prev_features is None:
            reward = SCORE_SCALE * delta_score + ENERGY_SCALE * delta_energy
            self.prev_features  = self.current_features
            self.prev_action    = self.current_action
            self.prev_reward    = reward
            self.prev_dist      = self.curr_dist
            self.prev_game_over = game_over
            self.step_counter  += 1
            self.save_checkpoint()
            return
        # Step 4: Update weights (Q-learning update for previous step)
        if self.prev_features is not None:
            assert None not in [self.prev_action, self.prev_reward, self.prev_game_over]
            # Calculate Q(s, a) for previous state/action

            # TODO: Implement this - DONE
            prev_q = float(np.dot(self.prev_features, self.weights[:, self.prev_action]))
            # weights update calculation
            # TODO: Implement this, delta_w means the change of weights - DONE
            target = self.prev_reward + (0 if self.prev_game_over else DISCOUNT_FACTOR * self.max_q_value)
            td_error = target - prev_q
            delta_w = LEARNING_RATE * td_error * self.prev_features


            # Gradient clipping for stability
            delta_w = np.clip(delta_w, -0.01, 0.01)
            self.weights[:, self.prev_action] += delta_w

        # Step 5: Calculate reward
        # TODO: Implement the reward function. It may include delta_score, delta_energy, and penalize repeated actions
        # The easiest way is simply define reward = a * delta_score + b * delta_energy
        # You can also try more complicated reward functions
        # reward = None
        # pass
        reward = SCORE_SCALE * delta_score + ENERGY_SCALE * delta_energy

        if hasattr(self, "prev_dist"):
            reward += DIST_SCALE * (self.curr_dist - self.prev_dist)

        if self.prev_action is not None and self.prev_action == self.current_action:
            reward -= 0.5
        # Step 6: Update previous variables for next step
        self.prev_features = self.current_features
        self.prev_action = self.current_action
        self.prev_reward = reward
        self.prev_dist = self.curr_dist
        self.prev_game_over = game_over

        # Step 7: Increment step counter and save checkpoint
        self.step_counter += 1
        self.save_checkpoint()
    #teststest
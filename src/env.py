import numpy as np
from gymnasium import Env, spaces
from ray.rllib.env.env_context import EnvContext

class YahtzeeEnv(Env):
    def __init__(self, config: EnvContext):
        """
        Yahtzee custom environment for RLlib.

        Args:
            config (EnvContext): RLlib configuration.
        """
        super(YahtzeeEnv, self).__init__()
        self.num_dice = 5 # Cant handle more dice in _calculate_score function 
        self.sides_per_die = 6
        self.max_rolls_per_turn = 3
        self.leaderboard = {
                0: False,
                1: False,
                2: False,
                3: False,
                4: False,
                5: False,
                6: False,
                7: False,
                8: False,
                9: False,
                10: False,
                11: False,
                12: False,
                13: False,
            }
        self.num_categories = len(self.leaderboard)


        # Action space
        # 0: ones
        # 1: twos
        # 2: threes
        # 3: fours
        # 4: fives
        # 5: sixes
        # 6: pair
        # 7: 3 of a kind
        # 8: 4 of a kind
        # 9: double pair
        # 10: full house
        # 11: small straight
        # 12: large straight
        # 13: yathzee
        # 14: reroll all dice combination Cn1 + Cn2 + Cn3 + Cn4 + Cn5 + ... = 2^n      
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.num_categories + 2 ** self.num_dice)  # Actions: select a category or reroll
        self.observation_space = spaces.Dict({
            "dice": spaces.MultiDiscrete([self.sides_per_die] * self.num_dice),  # The dice values
            "score_sheet": spaces.MultiDiscrete([1] * self.num_categories),  # Scored/not scored
            "current_rolls": spaces.Discrete(self.max_rolls_per_turn + 1),  # Rolls remaining
        })

        # Initialize the game state
        self.dice = None
        self.score_sheet = None
        self.current_rolls = None

    def reset(self):
        """Resets the environment to the starting state."""
        self.dice = np.random.randint(1, self.sides_per_die + 1, size=self.num_dice)
        self.score_sheet = np.zeros(self.num_categories, dtype=int)  # 0 = not scored
        self.current_rolls = self.max_rolls_per_turn

        return self._get_obs()

    def _get_obs(self):
        """Construct the observation from the current state."""
        return {
            "dice": self.dice,
            "score_sheet": self.score_sheet,
            "current_rolls": self.current_rolls,
        }

    def step(self, action):
        """
        Takes an action in the environment.

        Args:
            action (int): The action to take. Either reroll specific dice or score in a category.

        Returns:
            obs: Updated observation.
            reward: Reward for the action.
            done: Whether the game is over.
            info: Extra information.
        """
        done = False
        reward = 0
        info = {}

        if action == 0:  # Example: "reroll"
            if self.current_rolls > 0:
                self.dice = np.random.randint(1, self.sides_per_die + 1, size=self.num_dice)
                self.current_rolls -= 1
            else:
                reward = -1  # Penalty for attempting a reroll with no rolls left
        else:  # Score in a category
            category_idx = action - 1
            if self.score_sheet[category_idx] == 0:  # Category not yet scored
                self.score_sheet[category_idx] = 1  # Mark as scored
                reward = self._calculate_score(category_idx)
            else:
                reward = -1  # Penalty for trying to score in a used category

        # Check if game is over (all categories scored or no rolls left)
        if np.all(self.score_sheet) or self.current_rolls == 0:
            done = True

        return self._get_obs(), reward, done, info

    def _calculate_score(self, category_idx):
        """Calculate score based on the selected category and dice."""
        if self.leaderboard[category_idx]:
            # If the category is already scored, return 0 to indicate an invalid action
            return 0

        dice_counts = np.bincount(self.dice, minlength=7)  # Count occurrences of each die face (1-6)
        score = 0

        if category_idx == 0:  # Ones
            score = dice_counts[1] * 1
        elif category_idx == 1:  # Twos
            score = dice_counts[2] * 2
        elif category_idx == 2:  # Threes
            score = dice_counts[3] * 3
        elif category_idx == 3:  # Fours
            score = dice_counts[4] * 4
        elif category_idx == 4:  # Fives
            score = dice_counts[5] * 5
        elif category_idx == 5:  # Sixes
            score = dice_counts[6] * 6
        elif category_idx == 6:  # Pair
            pairs = np.where(dice_counts >= 2)[0]  # Find all numbers with at least two occurrences
            if len(pairs) > 0:
                score = max(pairs) * 2  # Take the highest pair
        elif category_idx == 7:  # Three of a Kind
            threes = np.where(dice_counts >= 3)[0]
            if len(threes) > 0:  # Check for three occurrences
                score = max(threes) * 3
        elif category_idx == 8:  # Four of a Kind
            fours = np.where(dice_counts >= 4)[0]
            if len(fours) > 0:  # Check for four occurrences
                score = max(fours) * 4
        elif category_idx == 9:  # Double Pair
            pairs = np.where(dice_counts >= 2)[0]  # Find all numbers with at least two occurrences
            if len(pairs) >= 2:
                score = np.sum(pairs[:2]) * 2  # Sum the top two pairs
        elif category_idx == 10:  # Full House
            threes = np.where(dice_counts >= 3)[0]
            pairs = np.where(dice_counts >= 2)[0]
            if len(threes) > 0 and len(pairs) > 0:
                score = 25
        elif category_idx == 11:  # Small Straight
            # Check for sequence [1, 2, 3, 4, 5]
            if (all(dice_counts[i] >= 1 for i in [1, 2, 3, 4]) or
                all(dice_counts[i] >= 1 for i in [2, 3, 4, 5]) or
                all(dice_counts[i] >= 1 for i in [3, 4, 5, 6])):
                score = 30
        elif category_idx == 12:  # Large Straight
            # Check for sequence [2, 3, 4, 5, 6]
            if (all(dice_counts[i] >= 1 for i in [1, 2, 3, 4, 5]) or
                all(dice_counts[i] >= 1 for i in [2, 3, 4, 5, 6])):
                score = 40
        elif category_idx == 13:  # Yahtzee
            if np.any(dice_counts == 5):  # Check if any number appears exactly five times
                score = 50

        self.leaderboard[category_idx] = True
        return score

    def render(self, mode="human"):
        """Render the current state (for debugging)."""
        print(f"Dice: {self.dice}, Rolls left: {self.current_rolls}, Score Sheet: {self.score_sheet}")

# Example RLlib Configuration
from ray.rllib.algorithms.ppo import PPOConfig

if __name__ == "__main__":
    # Register the environment
    from ray.tune.registry import register_env
    def env_creator(env_config):
        return YahtzeeEnv(env_config)

    register_env("YahtzeeEnv", env_creator)

    # Train with RLlib PPO
    config = PPOConfig().environment("YahtzeeEnv").rollouts(num_rollout_workers=0)
    algo = config.build()

    # Train for a few iterations
    for i in range(10):
        result = algo.train()
        print(f"Iteration {i}: reward = {result['episode_reward_mean']}")

    algo.stop()
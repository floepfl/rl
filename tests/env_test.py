import unittest
import numpy as np
from src.env import YahtzeeEnv

class TestYahtzeeScoreCalculation(unittest.TestCase):
    def setUp(self):
        """Set up the environment before each test."""
        self.env = YahtzeeEnv()

    def test_ones(self):
        self.env.dice = [1, 1, 2, 3, 4]
        self.assertEqual(self.env._calculate_score(0), 2)

    def test_twos(self):
        self.env.dice = [2, 2, 2, 3, 4]
        self.assertEqual(self.env._calculate_score(1), 6)

    def test_pair(self):
        self.env.dice = [3, 3, 4, 5, 6]
        self.assertEqual(self.env._calculate_score(6), 6)  # Highest pair is 3

        self.env.dice = [5, 5, 4, 4, 6]
        self.assertEqual(self.env._calculate_score(6), 10)  # Highest pair is 5

    def test_three_of_a_kind(self):
        self.env.dice = [4, 4, 4, 2, 3]
        self.assertEqual(self.env._calculate_score(7), 12)

        self.env.dice = [1, 1, 1, 5, 6]
        self.assertEqual(self.env._calculate_score(7), 3)

    def test_four_of_a_kind(self):
        self.env.dice = [6, 6, 6, 6, 2]
        self.assertEqual(self.env._calculate_score(8), 24)

        self.env.dice = [3, 3, 3, 3, 3]
        self.assertEqual(self.env._calculate_score(8), 12)

    def test_double_pair(self):
        self.env.dice = [3, 3, 4, 4, 5]
        self.assertEqual(self.env._calculate_score(9), 14)

        self.env.dice = [1, 1, 2, 2, 3]
        self.assertEqual(self.env._calculate_score(9), 6)

    def test_full_house(self):
        self.env.dice = [3, 3, 3, 5, 5]
        self.assertEqual(self.env._calculate_score(10), 25)

        self.env.dice = [2, 2, 2, 4, 4]
        self.assertEqual(self.env._calculate_score(10), 25)

    def test_small_straight(self):
        self.env.dice = [1, 2, 3, 4, 6]
        self.assertEqual(self.env._calculate_score(11), 30)

        self.env.dice = [2, 3, 4, 5, 1]
        self.assertEqual(self.env._calculate_score(11), 30)

    def test_large_straight(self):
        self.env.dice = [2, 3, 4, 5, 6]
        self.assertEqual(self.env._calculate_score(12), 40)

        self.env.dice = [1, 2, 3, 4, 5]
        self.assertEqual(self.env._calculate_score(12), 40)

    def test_yahtzee(self):
        self.env.dice = [6, 6, 6, 6, 6]
        self.assertEqual(self.env._calculate_score(13), 50)

        self.env.dice = [3, 3, 3, 3, 3]
        self.assertEqual(self.env._calculate_score(13), 50)

    def test_already_scored(self):
        self.env.dice = [1, 1, 1, 2, 2]
        self.env.leaderboard[0] = True  # Mark "ones" as already scored
        self.assertEqual(self.env._calculate_score(0), 0)

    def test_invalid_category(self):
        with self.assertRaises(KeyError):
            self.env._calculate_score(14)  # Invalid category index

if __name__ == "__main__":
    unittest.main()
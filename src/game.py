import random
from sys import exit
import time
from dataclasses import dataclass
from typing import Tuple
import signal

from .game_config import GameConfig
from .algorithms.abstract_search import SearchAlgorithm, SearchResult
from .state import State


@dataclass
class ResultRecord(object):
    """
    Represents the result of a search algorithm.

    Attributes:
    - `path_length (int)`: The length of the path from the start node to the goal node, if found.
    - `time_cp (int)`: The time complexity of the search algorithm.
    - `space_cp (int)`: The space complexity of the search algorithm.
    - `time (float)`: The runtime of the search algorithm.
    """

    path_length: int
    time_cp: int
    space_cp: int
    time: float


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out")


class Game(object):
    """
    Represents an instance of the n-puzzle game.

    Attributes:
    - `game_config` (`GameConfig`): The configuration of the game, including the start and goal states.
    - `algorithm` (`SearchAlgorithm`): The search algorithm to use to solve the game.

    Methods:
    - `solve()`: Solves the given game configuration using the given algorithm.
    """

    __slots__ = ["game_config", "algorithm", "ignore_solvability"]

    def __init__(
        self,
        game_config: GameConfig,
        algorithm: SearchAlgorithm,
        ignore_solvability: bool = True,
    ) -> None:
        self.game_config = game_config
        self.algorithm = algorithm
        self.ignore_solvability = ignore_solvability

    def play(self) -> ResultRecord:
        # print(f"Algorithm: {self.algorithm.__class__.__name__}")
        start_state = self.game_config.start_state
        goal_state = self.game_config.goal_state

        if not self.ignore_solvability:
            if not self.game_config.is_solvable():
                exit("Solvability is False. Game terminated.")
        else:
            print("Ignoring solvability...")

        start_time = time.time()

        # Set a timeout of 60 seconds (adjust as needed)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # Set the alarm to trigger after 60 seconds

        try:
            search_result = self.algorithm.solve(start_state, goal_state)
        except TimeoutError as e:
            # print("Function took too long to execute. Stopping...")
            search_result = None  # Handle the timeout situation as needed
        finally:
            signal.alarm(0)  # Cancel the alarm

        end_time = time.time()
        execution_time = end_time - start_time

        path_length = len(search_result.path) - 1 if search_result else -1
        time_cp = search_result.time_cp if search_result else -1
        space_cp = search_result.space_cp if search_result else -1

        return ResultRecord(path_length, time_cp, space_cp, execution_time)


def game_generator(n: int = 3) -> GameConfig:
    """
    Generates a new game configuration for the 8-puzzle game.

    Args:
    - `n` (`int`): The size of the puzzle. Default is 3.

    Returns:
    - `GameConfig`: A new game configuration object containing the start and goal states.

    Example:
    ```
    game_config = game_generator(n = 4)
    ```
    """
    try:
        start = [i for i in range(n**2)]
        goal = [i for i in range(n**2)]

        random.shuffle(start)
        random.shuffle(goal)

        start_state = State(*start)
        goal_state = State(*goal)

        game_config = GameConfig(start_state=start_state, goal_state=goal_state)

    except ValueError as e:
        print(f"An error occurred: {e}")
        return None

    return game_config

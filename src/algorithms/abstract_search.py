from typing import List, Union
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..heuristics import CallableHeuristicClass

from ..node import Node
from ..state import State


class Result(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CUTOFF = "cutoff"


@dataclass
class SearchResult(object):
    """
    Represents the result of a search algorithm.

    Attributes:
    - `result (Result)`: The result of the search.
    - `path (Union[List[Node], None])`: The path from the start node to the goal node, if found.
    - `time_cp (int)`: The time complexity of the search algorithm.
    - `space_cp (int)`: The space complexity of the search algorithm.

    Methods:
    - `print_result()`: Print the result. If a path is found, print the path.
    """

    result: Result
    path: Union[List[Node], None]
    time_cp: int
    space_cp: int

    def print_result(self) -> None:
        print(self.result)
        if self.path is not None:
            print("Path:")
            print(f"\tLength: {len(self.path)-1}")
            for p in self.path:
                print(f"\t{p}")
        else:
            print("No Path is found")
        print(f"Space: {self.space_cp}")
        print(f"Time: {self.time_cp}")


class SearchAlgorithm(ABC):
    """
    An abstract class representing a search algorithm. Subclasses should implement the `search`
    method to define the specific search algorithm.

    Methods:
    - `solve(start_state: State, goal_state: State, print: bool = True) -> None`:
        Set the goal state of the search algorithm, then search using `_search()`
    - `_search(start: Node) -> SearchResult`:
        Searches for a path from the start node to the goal node. Returns a `SearchResult` object.
        Must be implemented.
    """

    def _is_goal(self, node: Node) -> bool:
        return node.state == self.goal_state

    def _reconstruct_path(self, node: Node) -> List[Node]:
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]

    def _set_goal(self, goal_state: State) -> None:
        self.goal_state = goal_state

    @abstractmethod
    def _search(self, start_state: State) -> SearchResult:
        ...

    def solve(self, start_state: State, goal_state: State, print: bool = False) -> SearchResult:
        self._set_goal(goal_state)
        search_result = self._search(start_state)
        if print:
            search_result.print_result()
        return search_result


class UninformedSearchAlgorithm(SearchAlgorithm):
    """
    An abstract class representing an uninformed search algorithm.

    This class inherits from the `SearchAlgorithm` class and provides a template for implementing
    uninformed search algorithms. Subclasses should implement the `search` method to define the
    specific search algorithm.
    """

    def __init__(self) -> None:
        super().__init__()


class InformedSearchAlgorithm(SearchAlgorithm):
    """
    An abstract class representing an informed search algorithm.

    This class inherits from the `SearchAlgorithm` class and provides a template for implementing
    informed search algorithms. Subclasses should implement the `search` method to define the
    specific search algorithm.

    Attributes:
    - `heuristic` (`CallableHeuristicClass`): The heuristic function.
    """

    def __init__(self, heuristic: CallableHeuristicClass) -> None:
        super().__init__()
        self.heuristic = heuristic

    def h_cost(self, node: Node) -> int:
        return self.heuristic(node.state, self.goal_state)

from typing import Set
from queue import PriorityQueue

from ..heuristics import CallableHeuristicClass
from ..algorithms.abstract_search import Result, SearchResult, InformedSearchAlgorithm
from ..node import PNode
from ..state import State


class GBFS(InformedSearchAlgorithm):
    """
    GBFS search algorithm implementation.

    Args:
    - `heuristic` (`CallableHeuristicClass`): The heuristic function.
    """

    def __init__(self, heuristic: CallableHeuristicClass) -> None:
        super().__init__(heuristic)

    def _search(self, start_state: State) -> SearchResult:
        start = PNode(start_state)
        start.f_cost = self.h_cost(start)

        frontier: PriorityQueue[PNode] = PriorityQueue()
        frontier.put(start)

        closed: Set[State] = set()

        time_cp = 0
        space_cp = len(closed) + frontier.qsize()

        while frontier:
            node = frontier.get()
            closed.add(node.state)

            time_cp += 1
            space_cp = max(space_cp, len(closed) + frontier.qsize())

            if self._is_goal(node):
                path = self._reconstruct_path(node)
                return SearchResult(
                    result=Result.SUCCESS, path=path, time_cp=time_cp, space_cp=space_cp
                )

            for child in node.expand():
                if child.state not in closed:
                    child.f_cost = self.h_cost(child)
                    frontier.put(child)
                    time_cp += 1
            space_cp = max(space_cp, len(closed) + frontier.qsize())

        return SearchResult(
            result=Result.FAILURE, path=None, time_cp=time_cp, space_cp=space_cp
        )


__all__ = ["GBFS"]

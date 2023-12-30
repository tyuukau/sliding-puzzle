from typing import Tuple

from ..heuristics import CallableHeuristicClass
from ..algorithms.abstract_search import Result, SearchResult, InformedSearchAlgorithm
from ..node import PNode
from ..state import State


class RBFS(InformedSearchAlgorithm):
    """
    Recursive Best First Search (RBFS) algorithm implementation.

    Args:
    - `heuristic` (`CallableHeuristicClass`): The heuristic function.
    - `limit` (`int`): Limit for the f_cost of nodes.
    """

    def __init__(self, heuristic: CallableHeuristicClass, limit: int = 1e15) -> None:
        super().__init__(heuristic)
        self.limit = limit

    def _rbfs(
        self, node: PNode, limit: int, time_cp: int = 1, space_cp: int = 1
    ) -> Tuple[SearchResult, int]:
        if self._is_goal(node):
            path = self._reconstruct_path(node)
            return SearchResult(Result.SUCCESS, path, time_cp, space_cp), limit

        children = []

        for child in node.expand():
            child.f_cost = max(child.cost + self.h_cost(child), node.f_cost)
            if node.parent is not None:
                if child.state == node.parent.state:
                    child.f_cost = 1e20
            children.append(child)

        while True:
            children.sort(key=lambda node: node.f_cost)

            best = children[0]
            if best.f_cost > limit:
                return SearchResult(Result.FAILURE, None, time_cp, space_cp), best.f_cost

            alternative = children[1]
            search_result, best.f_cost = self._rbfs(
                best, min(limit, alternative.f_cost), time_cp, space_cp
            )
            if search_result.result != Result.FAILURE:
                return search_result, best.f_cost

    def _search(self, start_state: State) -> SearchResult:
        start = PNode(start_state)
        start.f_cost = start.cost + self.h_cost(start)

        search_result, _ = self._rbfs(start, self.limit)
        return search_result


__all__ = ["RBFS"]

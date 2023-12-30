from typing import Iterator, Optional, Self
from enum import Enum

from .state import State


class Action(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class Node(object):
    """
    Represents a node in a search tree.

    Attributes:
    - `state` (`State`): The state of the node.
    - `parent` (`Node`): The parent node of the node.
    - `action` (`str`): The action taken to reach the node.
    - `cost` (`int`): The cost of the path from the initial state to the node.

    Methods:
    - `expand() -> List[Node]`:
        Returns a list of neighbors expanded from the node.
    """

    __slots__ = ["state", "parent", "action", "cost"]

    def __init__(
        self,
        state: State,
        parent: Optional[Self] = None,
        action: Optional[Action] = None,
        cost: int = 0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def __repr__(self) -> str:
        if self.ư is not None:
            return f"Node(state={self.state}, parent.state={self.parent.state}, action={self.action}, cost={self.cost})"
        else:
            return (
                f"Node(state={self.state}, parent=None, action={self.action}, cost={self.cost})"
            )

    def __str__(self) -> str:
        if self.parent is not None:
            return f"Node(state={self.state}, parent.state={self.parent.state}, action={self.action}, cost={self.cost})"
        else:
            return (
                f"Node(state={self.state}, parent=None, action={self.action}, cost={self.cost})"
            )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return False
        return (
            self.state == other.state
            and self.parent == other.parent
            and self.action == other.action
            and self.cost == other.cost
        )

    def __lt__(self, other) -> bool:
        return self.cost < other.cost

    def __hash__(self) -> int:
        return hash((self.state, self.parent, self.action, self.cost))

    def expand(self) -> Iterator[Self]:
        i_old, j_old = self.state.get_blank_tile()
        action_dict = [
            (i_old - 1, j_old, Action.UP),
            (i_old + 1, j_old, Action.DOWN),
            (i_old, j_old - 1, Action.LEFT),
            (i_old, j_old + 1, Action.RIGHT),
        ]
        for i_new, j_new, action in action_dict:
            try:
                new_state = self.state.swap(i_old, j_old, i_new, j_new)
                yield Node(state=new_state, parent=self, action=action, cost=self.cost + 1)
            except AssertionError:
                continue


class PNode(Node):
    """
    Represents a node in a search tree.

    Attributes:
    - `state` (`State`): The state of the node.
    - `parent` (`Node`): The parent node of the node.
    - `action` (`str`): The action taken to reach the node.
    - `cost` (`int`): The cost of the path from the initial state to the node.
    - `f_cost` (`int`): The f-cost of the node.

    Methods:
    - `expand() -> Iterator[Node]`:
        Returns a list of neighbors expanded from the node.
    """

    __slots__ = ["f_cost"]

    def __init__(
        self,
        state: State,
        parent: Optional[Self] = None,
        action: Optional[Action] = None,
        cost: int = 0,
        f_cost: int = 0,
    ):
        super().__init__(state, parent, action, cost)
        self.f_cost = f_cost

    def __repr__(self) -> str:
        if self.parent is not None:
            return f"Node(state={self.state}, parent.state={self.parent.state}, action={self.action}, cost={self.cost}, path={self.f_cost})"
        else:
            return f"Node(state={self.state}, parent=None, action={self.action}, cost={self.cost}, path={self.f_cost})"

    def __str__(self) -> str:
        if self.parent is not None:
            return f"Node(state={self.state}, parent.state={self.parent.state}, action={self.action}, cost={self.cost}, path={self.f_cost})"
        else:
            return f"Node(state={self.state}, parent=None, action={self.action}, cost={self.cost}, path={self.f_cost})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, PNode):
            return False
        return (
            self.state == other.state
            and self.parent == other.parent
            and self.action == other.action
            and self.cost == other.cost
            and self.f_cost == self.f_cost
        )

    def __lt__(self, other) -> bool:
        if self.f_cost < other.f_cost:
            return True
        elif self.f_cost == other.f_cost:
            if self.cost > other.cost:
                return True
        return False

    def __hash__(self) -> int:
        return hash((self.state, self.parent, self.action, self.cost, self.f_cost))

    def expand(self) -> Iterator[Self]:
        i_old, j_old = self.state.get_blank_tile()
        action_dict = [
            (i_old - 1, j_old, Action.UP),
            (i_old + 1, j_old, Action.DOWN),
            (i_old, j_old - 1, Action.LEFT),
            (i_old, j_old + 1, Action.RIGHT),
        ]
        for i_new, j_new, action in action_dict:
            try:
                new_state = self.state.swap(i_old, j_old, i_new, j_new)
                yield PNode(state=new_state, parent=self, action=action, cost=self.cost + 1)
            except AssertionError:
                continue

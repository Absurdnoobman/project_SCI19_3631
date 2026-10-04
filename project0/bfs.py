from collections import deque
from typing import Deque, List, Optional, Set, Tuple, Union

from pacman_module.game import Agent, Grid
from pacman_module.pacman import Directions, GameState


GridLikeTuple = Tuple[Tuple[bool, ...], ...]
Vector2i = Tuple[int, int]
FoodState = Optional[Union[GridLikeTuple, Grid]]
Key = Tuple[Vector2i, FoodState, Tuple[Vector2i, ...]]


def key(state: GameState) -> Key:
    """Return a hashable key that uniquely identifies a game state."""
    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules()),
    )


class PacmanAgent(Agent):
    """A Pacman agent based on breadth-first search."""

    def __init__(self, args) -> None:
        """Initialize an agent from the command-line arguments namespace."""
        super().__init__()
        self.moves: List[str] = []

    def get_action(self, state: GameState) -> str:
        """Return the next legal move for the current game state."""
        if not self.moves:
            self.moves = self.bfs(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    def bfs(self, state: GameState) -> List[str]:
        """Return a shortest sequence of moves that solves the layout."""
        fringe: Deque[Tuple[GameState, List[str]]] = deque([(state, [])])
        closed: Set[Key] = set()

        while fringe:
            current, path = fringe.popleft()

            if current.isWin():
                return path

            current_key = key(current)
            if current_key in closed:
                continue

            closed.add(current_key)
            for next_state, action in current.generatePacmanSuccessors():
                fringe.append((next_state, path + [action]))

        return []


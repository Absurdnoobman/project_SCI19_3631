from pacman_module.game import Agent, Grid
from pacman_module.pacman import Directions, GameState

from collections import deque

type GridLikeTuple = tuple[tuple[bool, ...]]

type FoodState = None | GridLikeTuple | Grid
type Vector2i = tuple[int, int]

type Key = tuple[Vector2i, FoodState, tuple[Vector2i, ...]]
"""A hashable key object that uniquely identifies a Pacman game state."""

def key(state: GameState) -> Key:
    """
    Returns a key that uniquely identifies a Pacman game state.

    Arguments:
    ----------
    - `state`: the current game state. See FAQ and class
               `pacman.GameState`.

    Return:
    -------
    - A hashable key object that uniquely identifies a Pacman game state.
    """
    return (state.getPacmanPosition(), state.getFood(), tuple(state.getCapsules()))


class PacmanAgent(Agent):
    """
    A Pacman agent based on Breadth-First-Search.
    """

    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.moves = []

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        if not self.moves:
            self.moves = self.bfs(state)

        try:
            return self.moves.pop(0)

        except IndexError:
            return Directions.STOP

    def bfs(self, state: GameState) -> list[Directions]:
        """
        Given a pacman game state,
        returns a list of legal moves to solve the search layout.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A list of legal moves as defined in `game.Directions`.
        """
        path: list[Directions] = []
        fringe = deque([(state, path)])
        closed: set[Key] = set()

        while True:
            if len(fringe) == 0: 
                return []

            current, path = fringe.popleft()

            if current.isWin(): 
                return path

            current_key = key(current)

            if current_key not in closed:
                closed.add(current_key)

                for next_state, action in current.generatePacmanSuccessors():
                    fringe.append((next_state, path + [action]))

            

        

# Complete this class for all parts of the project

from collections import deque

from pacman_module.game import Agent, Grid
from pacman_module.pacman import Directions, GameState
from pacman_module import util
from bayesfilter import Vector2i

import numpy as np


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

    def get_action(self, state: GameState, 
                   belief_state: list[np.ndarray]):
        """
        Given a pacman game state and a belief state,
                returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.
        - `belief_state`: a list of probability matrices.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        # XXX: Your code here to obtain bonus
        pacman_pos_t = state.getPacmanPosition()
        pacman_position: Vector2i = Vector2i(pacman_pos_t[0], 
                                             pacman_pos_t[1])
        walls: Grid = state.getWalls()
        
        alive_ghosts: list[np.ndarray] = list(filter(
            lambda M: np.sum(M) > 0,
            belief_state
        ))

        target_positions: list[Vector2i] = []
        for ghost_belief in alive_ghosts:
            x, y = np.unravel_index(np.argmax(ghost_belief),
                                    ghost_belief.shape)
            target_positions.append(Vector2i(int(x), int(y)))

        target_pos = Vector2i(-1, -1)

        if len(target_positions) > 1:
            distants: dict[int, Vector2i] = dict(list(map(
                lambda pos: (pos.manhattan_to(pacman_position), pos), 
                target_positions
            )))

            target_key = min(distants.keys())

            target_pos = distants[target_key]

        elif len(target_positions) == 1:
            target_pos = target_positions[0]
        else:
            return Directions.STOP

        if target_pos.x == -1 or target_pos.y == -1:
            raise Exception("Fatal Error: no target position")
        
        legal_actions: list[str] = state.getLegalActions()
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        if pacman_position == target_pos:
            return legal_actions[0]

        queue: deque[tuple[Vector2i, str]] = deque([])
        visited: set[Vector2i] = set()

        for action in legal_actions:
            neighbours = {
                Directions.NORTH: pacman_position.north,
                Directions.SOUTH: pacman_position.south,
                Directions.EAST: pacman_position.east,
                Directions.WEST: pacman_position.west
            }
            next_pos = neighbours.get(action)
            if next_pos is None:
                raise Exception("Fatal Error: action maybe broken, next_pos failed.")

            if next_pos == target_pos:
                return action

            queue.append((next_pos, action))
            visited.add(next_pos)

        while queue:
            current_pos, first_action = queue.popleft()
            for nbr in current_pos.neighbours():
                if (0 <= nbr.x < walls.width
                    and 0 <= nbr.y < walls.height
                    and not walls[nbr.x][nbr.y]
                    and nbr not in visited):
                        if nbr == target_pos:
                            return first_action

                        visited.add(nbr)
                        queue.append((nbr, first_action))

        if legal_actions:
            return legal_actions[0]

        # XXX: End of your code here to obtain bonus

        return Directions.STOP

    
    

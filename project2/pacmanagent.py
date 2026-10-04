# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions

import numpy as np


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

    def get_action(self, state, belief_state: list[np.ndarray]):
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
        
        alive_ghosts: list[np.ndarray] = []
        for prob_mat in belief_state:
            if np.sum(prob_mat):
                


        # XXX: End of your code here to obtain bonus

        return Directions.STOP

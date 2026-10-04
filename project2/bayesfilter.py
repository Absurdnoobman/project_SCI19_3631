# Complete this class for all parts of the project

from __future__ import annotations
from pacman_module.game import Agent, Grid
import numpy as np
from pacman_module import util
from scipy.stats import binom

from typing import NamedTuple


class Vector2i(NamedTuple):
    """
    A 2D integer vector representing grid coordinates (x, y).

    Attributes:
    -----------
    - `x`: Integer horizontal coordinate.
    - `y`: Integer vertical coordinate.
    """
    x: int
    y: int

    @property
    def north(self) -> Vector2i:
        """Return the neighboring coordinate to the north (y + 1)."""
        return Vector2i(self.x, self.y + 1)

    @property
    def south(self) -> Vector2i:
        """Return the neighboring coordinate to the south (y - 1)."""
        return Vector2i(self.x, self.y - 1)

    @property
    def east(self) -> Vector2i:
        """Return the neighboring coordinate to the east (x + 1)."""
        return Vector2i(self.x + 1, self.y)

    @property
    def west(self) -> Vector2i:
        """Return the neighboring coordinate to the west (x - 1)."""
        return Vector2i(self.x - 1, self.y)

    def neighbours(self) -> list[Vector2i]:
        """Return a list of all 4 orthogonal neighboring coordinates."""
        return [self.north, self.south, self.east, self.west]

    def manhattan_to(self, other: tuple[int, int]) -> int:
        """
        Compute Manhattan distance to another coordinate.

        Arguments:
        ----------
        - `other`: (x, y) coordinate as a tuple or Vector2i.

        Return:
        -------
        - The L1 (Manhattan) distance.
        """
        return util.manhattanDistance(self, other)

    def __add__(self, rhs: tuple[int, int]) -> Vector2i:
        """Add another coordinate offset to this vector."""
        return Vector2i(self.x + rhs[0], self.y + rhs[1])


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

        """
            Variables to use in 'update_belief_state' method.
            Initialization occurs in 'get_action' method.

            XXX: DO NOT MODIFY THE DEFINITION OF THESE VARIABLES
            # Doing so will result in a 0 grade.
        """

        # Current list of belief states over ghost positions
        self.beliefGhostStates = None

        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None

        # Hyper-parameters
        self.ghost_type = self.args.ghostagent
        self.sensor_variance = self.args.sensorvariance

        self.p = 0.5
        self.n = int(self.sensor_variance/(self.p*(1-self.p)))

        # XXX: Your code here
        # NB: Adding code here is not necessarily useful, but you may.
        
        # XXX: End of your code
    

    def _try_get_walls(self) -> tuple[Grid, int, int]:
        """
        A function to safely retrieves `self.walls` and its dimensions.

        Return:
        -------
        - A `Grid` object representing the maze walls.
        - An integer representing the width of the maze.
        - An integer representing the height of the maze.
        """
        if self.walls is None or not isinstance(self.walls, Grid):
            raise ValueError("self.walls has not been initialized.")

        return self.walls, self.walls.width, self.walls.height
        
    def _get_sensor_model(self, 
                          pacman_position: Vector2i, 
                          evidence: float):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The sensor model represented as a 2D numpy array of
        size [width, height].
        The element at position (w, h) is the probability
        P(E_t=evidence | X_t=(w, h))
        """
        walls, width, height = self._try_get_walls()

        result = np.zeros((width, height))

        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    continue # if (x, y) is a wall then skip.

                this_position = Vector2i(x, y)

                distance: int = this_position.manhattan_to(pacman_position)
                noise = evidence - distance
                k = noise + (self.n * self.p)

                if 0 <= k <= self.n:
                    result[x, y] = binom.pmf(k, self.n, self.p)

        return result

    def _get_transition_model(self, pacman_position: Vector2i):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The transition model represented as a 4D numpy array of
        size [width, height, width, height].
        The element at position (w1, h1, w2, h2) is the probability
        P(X_t+1=(w1, h1) | X_t=(w2, h2))
        """
        walls, width, height = self._try_get_walls()

        transition_model = np.zeros((width, height, width, height))

        ghost_weights = {"scared": 8, "afraid": 2, "confused": 1}
        scare_weight = ghost_weights.get(self.ghost_type, 1)

        for w2 in range(width):
            for h2 in range(height):
                if walls[w2][h2]:
                    continue 
                # again, ignore the walls since ghost can not move 
                # to a wall. 
                # unless it that a null zone which lead to the backrooms.
                current_position = Vector2i(w2, h2)

                current_distant = current_position.manhattan_to(pacman_position)

                legal_neighbour: list[Vector2i] = []
                for direction in current_position.neighbours():
                    if (0 <= direction.x < width 
                        and 0 <= direction.y < height 
                        and not walls[direction.x][direction.y]):
                            legal_neighbour.append(direction)

                move_weights: dict[Vector2i, int] = {}

                for neighbour in legal_neighbour:
                    neighbour_distant = neighbour.manhattan_to(pacman_position)
                    
                    weight = scare_weight if \
                    neighbour_distant >= current_distant \
                    else 1 

                    move_weights[neighbour] = weight

                total_weight = sum(move_weights.values())

                if total_weight > 0:
                    for neighbour, weight in move_weights.items():
                        transition_model[
                            neighbour.x, neighbour.y, w2, h2
                            ] = weight / total_weight

        return transition_model

    def _get_updated_belief(self, 
                            belief: list[np.ndarray], 
                            evidences: list[int], 
                            pacman_position: Vector2i, 
                            ghosts_eaten: list[bool]):
        """
        Given a list of (noised) distances from pacman to ghosts,
        and the previous belief states before receiving the evidences,
        returns the updated list of belief states about ghosts positions

        Arguments:
        ----------
        - `belief`: A list of Z belief states at state x_{t-1}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze.
               Matrices filled with zeros must be returned for eaten ghosts.
        """
        _, width, height = self._try_get_walls()

        new_belief = []

        trans_mdl = self._get_transition_model(pacman_position)

        for ghost_i in range(len(belief)):
            if ghosts_eaten[ghost_i]:
                new_belief.append(np.zeros((width, height)))
                continue

            previous_belief = belief[ghost_i]

            predicted_belief = np.tensordot(trans_mdl, previous_belief, 
                                           axes=([2, 3], [0, 1]))

            sensor_mdl = self._get_sensor_model(pacman_position, 
                                                evidences[ghost_i])

            unnormalised = predicted_belief * sensor_mdl
            total = np.sum(unnormalised)
            if total > 0:
                new_belief.append(unnormalised / total)
            else:
                new_belief.append(predicted_belief)

        return new_belief

    def update_belief_state(self, evidences, pacman_position, ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        belief = self._get_updated_belief(self.beliefGhostStates, evidences, 
                                          pacman_position, ghosts_eaten)
        self.beliefGhostStates = belief
        return belief

    def _get_evidence(self, state):
        """
        Computes noisy distances between pacman and ghosts.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.


        Return:
        -------
        - A list of Z noised distances in real numbers
          where Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        pacman_position = state.getPacmanPosition()
        noisy_distances = []

        for pos in positions:
            true_distance = util.manhattanDistance(pos, pacman_position)
            noise = binom.rvs(self.n, self.p) - self.n*self.p
            noisy_distances.append(true_distance + noise)

        return noisy_distances

    def _record_metrics(self, belief_states, state):
        """
        Use this function to record your metrics
        related to true and belief states.
        Won't be part of specification grading.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.
        - `belief_states`: A list of Z
           N*M numpy matrices of probabilities
           where N and M are respectively width and height
           of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """
        pass

    def get_action(self, state):
        """
        Given a pacman game state, returns a belief state.

        Arguments:
        ----------
        - `state`: the current game state.
                   See FAQ and class `pacman.GameState`.

        Return:
        -------
        - A belief state.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()

        evidence = self._get_evidence(state)
        newBeliefStates = self.update_belief_state(evidence,
                                                   state.getPacmanPosition(),
                                                   state.data._eaten[1:])
        self._record_metrics(self.beliefGhostStates, state)

        return newBeliefStates, evidence

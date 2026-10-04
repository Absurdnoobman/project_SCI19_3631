from pacman_module.game import Agent, Directions


def key(state):
    """Returns a hashable key that uniquely identifies a game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key object.
    """
    return (
        state.getPacmanPosition(),
        state.getGhostPosition(1),
        state.getGhostDirection(1),
        state.getFood(),
    )


class PacmanAgent(Agent):
    """Pacman agent based on the Minimax algorithm."""

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        best_value = float('-inf')
        best_action = Directions.STOP
        path = {key(state)}

        for successor, action in state.generatePacmanSuccessors():
            value = self.min_value(successor, path)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def min_value(self, state, path):
        """Value of a ghost node (the ghost minimizes Pacman's score)."""
        if state.isWin() or state.isLose():
            return state.getScore()

        current = key(state)

        if current in path:
            return state.getScore()

        path.add(current)
        value = float('inf')

        for successor, _ in state.generateGhostSuccessors(1):
            value = min(value, self.max_value(successor, path))

        path.discard(current)

        return value

    def max_value(self, state, path):
        """Value of a Pacman node (Pacman maximizes his score)."""
        if state.isWin() or state.isLose():
            return state.getScore()

        current = key(state)

        if current in path:
            return state.getScore()

        path.add(current)
        value = float('-inf')

        for successor, _ in state.generatePacmanSuccessors():
            value = max(value, self.min_value(successor, path))

        path.discard(current)

        return value

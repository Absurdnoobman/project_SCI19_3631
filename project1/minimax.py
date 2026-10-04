from math import inf

from pacman_module.game import Agent, Directions


class PacmanAgent(Agent):
    """Pacman agent that uses exact Minimax search."""

    def __init__(self):
        super().__init__()
        self._cache = {}

    def get_action(self, state):
        """Return the Pacman action with the best minimax value."""
        legal = [
            action
            for action in state.getLegalActions(0)
            if action != Directions.STOP
        ]
        if not legal:
            return Directions.STOP

        self._cache.clear()
        best_value = -inf
        best_action = legal[0]

        for action in legal:
            successor = state.generateSuccessor(0, action)
            value = self._min_value(successor, 1)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _min_value(self, state, agent_index):
        """Return the minimum value for the current ghost."""
        if state.isWin() or state.isLose():
            return state.getScore()

        key = (state, agent_index)
        if key in self._cache:
            return self._cache[key]

        legal = [
            action
            for action in state.getLegalActions(agent_index)
            if action != Directions.STOP
        ]
        if not legal:
            return state.getScore()

        value = inf
        last_agent = state.getNumAgents() - 1
        next_index = agent_index + 1

        for action in legal:
            successor = state.generateSuccessor(agent_index, action)
            if agent_index == last_agent:
                child_value = self._max_value(successor)
            else:
                child_value = self._min_value(successor, next_index)
            value = min(value, child_value)

        self._cache[key] = value
        return value

    def _max_value(self, state):
        """Return the maximum value for Pacman."""
        if state.isWin() or state.isLose():
            return state.getScore()

        key = (state, 0)
        if key in self._cache:
            return self._cache[key]

        legal = [
            action
            for action in state.getLegalActions(0)
            if action != Directions.STOP
        ]
        if not legal:
            return state.getScore()

        value = -inf
        for action in legal:
            successor = state.generateSuccessor(0, action)
            value = max(value, self._min_value(successor, 1))

        self._cache[key] = value
        return value

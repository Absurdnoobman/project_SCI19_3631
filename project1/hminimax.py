"""Depth-limited minimax Pacman agent with alpha-beta pruning."""

from collections import deque
from math import inf

from pacman_module.game import Agent, Directions


Position = tuple[int, int]


class PacmanAgent(Agent):
    """Choose actions using heuristic minimax search."""

    def __init__(self):
        super().__init__()
        self._distances: dict[Position, dict[Position, int]] = {}
        self._visits = {}
        self._walls = None

    def get_action(self, state):
        """Return the best legal action found by H-Minimax."""
        if state.isWin() or state.isLose():
            return Directions.STOP

        self._prepare_maze(state)
        state_key = self._visit_key(state)
        self._visits[state_key] = self._visits.get(state_key, 0) + 1
        depth = self._search_depth(state)
        _, action = self._max_value(state, depth, -inf, inf)

        if action is not None:
            return action

        legal = state.getLegalPacmanActions()
        non_stop = [move for move in legal if move != Directions.STOP]
        return non_stop[0] if non_stop else Directions.STOP

    def _max_value(self, state, depth, alpha, beta):
        """Return the value and action of a Pacman (MAX) node."""
        if self._finished(state) or depth == 0:
            return self._evaluate(state), None

        successors = state.generatePacmanSuccessors()
        if not successors:
            return self._evaluate(state), None

        successors.sort(
            key=lambda item: self._evaluate(item[0]), reverse=True
        )
        best_value = -inf
        best_action = None

        for successor, action in successors:
            if successor.isWin() or successor.isLose():
                value = self._evaluate(successor)
            else:
                value = self._min_value(
                    successor, 1, depth, alpha, beta
                )

            value -= 35.0 * self._visits.get(
                self._visit_key(successor), 0
            )

            if value > best_value:
                best_value = value
                best_action = action

            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        return best_value, best_action

    def _min_value(self, state, agent_index, depth, alpha, beta):
        """Return the value of a ghost (MIN) node."""
        if self._finished(state):
            return self._evaluate(state)

        successors = state.generateGhostSuccessors(agent_index)
        if not successors:
            return self._evaluate(state)

        successors.sort(key=lambda item: self._evaluate(item[0]))
        best_value = inf

        for successor, _ in successors:
            next_agent = agent_index + 1
            if successor.isWin() or successor.isLose():
                value = self._evaluate(successor)
            elif next_agent == state.getNumAgents():
                value, _ = self._max_value(
                    successor, depth - 1, alpha, beta
                )
            else:
                value = self._min_value(
                    successor, next_agent, depth, alpha, beta
                )

            best_value = min(best_value, value)
            beta = min(beta, best_value)
            if beta <= alpha:
                break

        return best_value

    @staticmethod
    def _finished(state):
        """Return whether no more actions should be searched from state."""
        return state.isWin() or state.isLose()

    def _evaluate(self, state):
        """Estimate the utility of a non-terminal game state."""
        if state.isWin() or state.isLose():
            return state.getScore()

        pacman = self._integer_position(state.getPacmanPosition())
        food = state.getFood().asList()
        value = state.getScore()

        if food:
            food_distances = [self._distance(pacman, dot) for dot in food]
            nearest_food = min(food_distances)
            value -= 8.0 * nearest_food
            value -= 12.0 * len(food)

        value -= 25.0 * self._visits.get(self._visit_key(state), 0)

        ghost_distances = [
            self._distance(
                pacman, self._integer_position(state.getGhostPosition(index))
            )
            for index in range(1, state.getNumAgents())
        ]
        if ghost_distances:
            nearest_ghost = min(ghost_distances)
            if nearest_ghost <= 1:
                value -= 40.0
            elif nearest_ghost == 2:
                value -= 18.0
            elif nearest_ghost == 3:
                value -= 6.0
            else:
                value += min(nearest_ghost, 10) * 2.0

        legal_moves = [
            action
            for action in state.getLegalPacmanActions()
            if action != Directions.STOP
        ]
        value += 3.0 * len(legal_moves)
        return value

    def _visit_key(self, state):
        """Identify repeated positions that made no food progress."""
        return (
            self._integer_position(state.getPacmanPosition()),
            tuple(state.getFood().asList()),
        )

    @staticmethod
    def _integer_position(position):
        """Convert an agent position to a grid coordinate."""
        return tuple(int(coordinate) for coordinate in position)

    def _prepare_maze(self, state):
        """Clear cached distances when a different maze is loaded."""
        walls = state.getWalls()
        if walls is not self._walls:
            self._walls = walls
            self._distances.clear()

    def _distance(self, start, goal):
        """Return the shortest path length between two maze positions."""
        if start not in self._distances:
            self._distances[start] = self._distances_from(start)
        return self._distances[start].get(goal, inf)

    def _distances_from(self, start):
        """Compute shortest path lengths from one maze position."""
        distances = {start: 0}
        queue = deque([start])

        while queue:
            x, y = queue.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                neighbor = (x + dx, y + dy)
                if neighbor in distances:
                    continue
                if self._walls[neighbor[0]][neighbor[1]]:
                    continue
                distances[neighbor] = distances[(x, y)] + 1
                queue.append(neighbor)

        return distances

    @staticmethod
    def _search_depth(state):
        """Use an extra search round on very small mazes."""
        walls = state.getWalls()
        open_cells = sum(
            not walls[x][y]
            for x in range(walls.width)
            for y in range(walls.height)
        )
        return 4 if open_cells <= 20 else 3

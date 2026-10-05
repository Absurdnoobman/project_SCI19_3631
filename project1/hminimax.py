from collections import deque

from pacman_module.game import Agent, Directions

DEPTH = 4
GHOST_RADIUS = 3
GHOST_WEIGHT = 40
FOOD_WEIGHT = 1.5
FOOD_LEFT_WEIGHT = 10


def key(state):
  
    return (
        state.getPacmanPosition(),
        state.getGhostPosition(1),
        state.getGhostDirection(1),
        state.getFood(),
    )


def bfs_distances(walls, source):
   
    dist = {source: 0}
    fringe = deque([source])

    while fringe:
        x, y = fringe.popleft()

        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) not in dist and not walls[nx][ny]:
                dist[(nx, ny)] = dist[(x, y)] + 1
                fringe.append((nx, ny))

    return dist


class PacmanAgent(Agent):

    def __init__(self):
        super().__init__()
        self.cache = {}

    def get_action(self, state):
      
        self.cache = {}
        alpha = float('-inf')
        beta = float('inf')
        best_value = float('-inf')
        best_action = Directions.STOP
        path = {key(state)}

        for successor, action in state.generatePacmanSuccessors():
            value = self.min_value(successor, alpha, beta, DEPTH - 1, path)

            if value > best_value:
                best_value = value
                best_action = action

            alpha = max(alpha, best_value)

        return best_action

    def distances(self, walls, source):
        
        if source not in self.cache:
            self.cache[source] = bfs_distances(walls, source)

        return self.cache[source]

    def evaluate(self, state):

        walls = state.getWalls()
        pacman = state.getPacmanPosition()
        dist = self.distances(walls, pacman)
        value = state.getScore()

        foods = state.getFood().asList()

        if foods:
            nearest = min(dist.get(food, 1000) for food in foods)
            value -= FOOD_WEIGHT * nearest
            value -= FOOD_LEFT_WEIGHT * len(foods)

        ghost = tuple(int(round(c)) for c in state.getGhostPosition(1))
        gdist = dist.get(ghost, 1000)

        if gdist < GHOST_RADIUS:
            value -= GHOST_WEIGHT * (GHOST_RADIUS - gdist)

        return value

    def min_value(self, state, alpha, beta, depth, path):

        if state.isWin() or state.isLose():
            return state.getScore()

        if depth <= 0:
            return self.evaluate(state)

        current = key(state)

        if current in path:
            return state.getScore()

        path.add(current)
        value = float('inf')

        for successor, _ in state.generateGhostSuccessors(1):
            value = min(
                value, self.max_value(successor, alpha, beta, depth, path)
            )

            if value <= alpha:
                break

            beta = min(beta, value)

        path.discard(current)

        return value

    def max_value(self, state, alpha, beta, depth, path):

        if state.isWin() or state.isLose():
            return state.getScore()

        if depth <= 0:
            return self.evaluate(state)

        current = key(state)

        if current in path:
            return state.getScore()

        path.add(current)
        value = float('-inf')

        for successor, _ in state.generatePacmanSuccessors():
            value = max(
                value,
                self.min_value(successor, alpha, beta, depth - 1, path),
            )

            if value >= beta:
                break

            alpha = max(alpha, value)

        path.discard(current)

        return value
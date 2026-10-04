from heapq import heappop, heappush
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from pacman_module.game import Agent, Grid
from pacman_module.pacman import Directions, GameState


GridLikeTuple = Tuple[Tuple[bool, ...], ...]
Vector2i = Tuple[int, int]
FoodState = Optional[Union[GridLikeTuple, Grid]]
Key = Tuple[Vector2i, FoodState, Tuple[Vector2i, ...]]
DistanceMap = Dict[Vector2i, Dict[Vector2i, int]]
SearchNode = Tuple[int, int, int, GameState, List[str], int]


def key(state: GameState) -> Key:
    """Return a hashable key that uniquely identifies a game state."""
    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules()),
    )


class PacmanAgent(Agent):
    """A Pacman agent based on A-star search."""

    def __init__(self, args) -> None:
        """Initialize an agent from the command-line arguments namespace."""
        super().__init__()
        self.args = args
        self.moves: List[str] = []
        self._walls: Optional[Grid] = None
        self._dist_map: DistanceMap = {}
        self._mst_cache: Dict[Tuple[Vector2i, ...], int] = {}

    def get_action(self, state: GameState) -> str:
        """Return the next legal move for the current game state."""
        if not self.moves:
            self.moves = self.astar(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    def _init_maze(self, state: GameState) -> None:
        """Precompute shortest-path distances between passable cells."""
        walls = state.getWalls()
        self._walls = walls
        width = walls.width
        height = walls.height

        passable = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if not walls[x][y]
        ]

        self._dist_map = {}
        for source in passable:
            distances = {source: 0}
            queue = [source]

            for current in queue:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbor = (current[0] + dx, current[1] + dy)
                    x, y = neighbor
                    if not (0 <= x < width and 0 <= y < height):
                        continue
                    if walls[x][y] or neighbor in distances:
                        continue

                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

            self._dist_map[source] = distances

        self._mst_cache = {}

    def _compute_mst(self, food_list: Sequence[Vector2i]) -> int:
        """Return the minimum-spanning-tree weight of the food positions."""
        number_of_food = len(food_list)
        if number_of_food <= 1:
            return 0

        in_tree = [False] * number_of_food
        minimum_distances = [float("inf")] * number_of_food
        minimum_distances[0] = 0.0
        mst_cost = 0

        for _ in range(number_of_food):
            closest_index = -1
            closest_distance = float("inf")

            for index in range(number_of_food):
                if (
                    not in_tree[index]
                    and minimum_distances[index] < closest_distance
                ):
                    closest_distance = minimum_distances[index]
                    closest_index = index

            in_tree[closest_index] = True
            mst_cost += int(closest_distance)
            current_food = food_list[closest_index]
            current_distances = self._dist_map.get(current_food, {})

            for index in range(number_of_food):
                if in_tree[index]:
                    continue

                distance = current_distances.get(
                    food_list[index], float("inf")
                )
                if distance < minimum_distances[index]:
                    minimum_distances[index] = distance

        return mst_cost

    def h(self, state: GameState) -> int:
        """Estimate the remaining cost with nearest-food plus food MST."""
        walls = state.getWalls()
        if self._walls is not walls:
            self._init_maze(state)

        food_list = state.getFood().asList()
        if not food_list:
            return 0

        position = state.getPacmanPosition()
        food_tuple = tuple(sorted(food_list))
        if food_tuple not in self._mst_cache:
            self._mst_cache[food_tuple] = self._compute_mst(food_list)

        mst_cost = self._mst_cache[food_tuple]
        position_distances = self._dist_map.get(position, {})
        nearest_food = min(
            position_distances.get(food, float("inf"))
            for food in food_list
        )
        return int(nearest_food + mst_cost)

    def astar(self, state: GameState) -> List[str]:
        """Return an optimal sequence of moves that solves the layout."""
        walls = state.getWalls()
        if self._walls is not walls:
            self._init_maze(state)

        start_heuristic = self.h(state)
        counter = 0
        fringe: List[SearchNode] = []
        heappush(fringe, (start_heuristic, 0, counter, state, [], 0))
        closed: Set[Key] = set()
        best_cost: Dict[Key, int] = {key(state): 0}

        while fringe:
            _, _, _, current, path, cost = heappop(fringe)

            if current.isWin():
                return path

            current_key = key(current)
            if current_key in closed:
                continue
            closed.add(current_key)

            successors = current.generatePacmanSuccessors()
            if successors is None:
                continue

            for next_state, action in successors:
                next_key = key(next_state)
                if next_key in closed:
                    continue

                next_cost = cost + 1
                if best_cost.get(next_key, float("inf")) <= next_cost:
                    continue

                best_cost[next_key] = next_cost
                counter += 1
                heappush(
                    fringe,
                    (
                        next_cost + self.h(next_state),
                        -next_cost,
                        counter,
                        next_state,
                        path + [action],
                        next_cost,
                    ),
                )

        return []

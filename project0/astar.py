from collections.abc import Sequence
from heapq import heappop, heappush
from typing import Optional

from pacman_module.game import Agent, Grid
from pacman_module.pacman import Directions, GameState

type GridLikeTuple = tuple[tuple[bool, ...]]
type Vector2i = tuple[int, int]

type FoodState = None | GridLikeTuple | Grid
type Key = tuple[Vector2i, FoodState, tuple[Vector2i, ...]]

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
    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules()),
    )


class PacmanAgent(Agent):
    """
    A Pacman agent based on A* (A star) Search.
    """

    def __init__(self, args) -> None:
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        super().__init__()
        self.args = args
        self.moves: list[Directions] = []
        self._walls: Optional[Grid] = None
        self._dist_map: dict[Vector2i, dict[Vector2i, int]] = {}
        self._mst_cache: dict[tuple[Vector2i, ...], int] = {}

    def get_action(self, state: GameState):
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
            self.moves = self.astar(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    def _init_maze(self, state: GameState) -> None:
        """
        Precomputes all-pairs shortest path maze distances.

        Arguments:
        ----------
        - `state`: a game state used to extract maze walls.
        """
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
        for src in passable:
            d: dict[Vector2i, int] = {src: 0}
            queue = [src]
            for u in queue:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    v = (u[0] + dx, u[1] + dy)
                    if 0 <= v[0] < width and 0 <= v[1] < height:
                        if not walls[v[0]][v[1]] and v not in d:
                            d[v] = d[u] + 1
                            queue.append(v)
            self._dist_map[src] = d

        self._mst_cache = {}

    def _compute_mst(self, food_list: Sequence[Vector2i]) -> int:
        """
        Computes the weight of the Minimum Spanning Tree of food positions.

        Arguments:
        ----------
        - `food_list`: a sequence of food coordinates.

        Return:
        -------
        - The sum of edge weights in the MST.
        """
        n = len(food_list)
        if n <= 1:
            return 0

        dist_map = self._dist_map
        in_tree = [False] * n
        min_dist = [float("inf")] * n
        min_dist[0] = 0.0
        mst_cost = 0

        for _ in range(n):
            u = -1
            u_dist = float("inf")
            for i in range(n):
                if not in_tree[i] and min_dist[i] < u_dist:
                    u_dist = min_dist[i]
                    u = i

            in_tree[u] = True
            mst_cost += int(u_dist)
            u_pos = food_list[u]
            u_dists = dist_map.get(u_pos, {})

            for v in range(n):
                if not in_tree[v]:
                    d = u_dists.get(food_list[v], float("inf"))
                    if d < min_dist[v]:
                        min_dist[v] = d

        return mst_cost

    def h(self, state: GameState) -> int:
        """
        Computes an admissible and consistent heuristic for A* search.

        Estimates the minimum cost (steps) to collect all remaining food dots
        from the given state by combining the distance to the nearest food
        with the Minimum Spanning Tree (MST) weight of all remaining food.

        Arguments:
        ----------
        - `state`: the current game state.

        Return:
        -------
        - The heuristic estimate as an integer.
        """
        walls = state.getWalls()
        if self._walls is not walls:
            self._init_maze(state)

        food_list = state.getFood().asList()
        if not food_list:
            return 0

        pos = state.getPacmanPosition()
        food_tuple = tuple(sorted(food_list))
        if food_tuple not in self._mst_cache:
            self._mst_cache[food_tuple] = self._compute_mst(food_list)

        mst_cost = self._mst_cache[food_tuple]
        pos_dists = self._dist_map.get(pos, {})
        min_food_dist = min(
            pos_dists.get(f, float("inf")) for f in food_list
        )
        return int(min_food_dist + mst_cost)

    def astar(self, state: GameState) -> list[Directions]:
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
        walls = state.getWalls()
        if self._walls is not walls:
            self._init_maze(state)

        start_h = self.h(state)
        counter = 0
        fringe: list[
            tuple[int, int, int, GameState, list[Directions], int]
        ] = []
        heappush(fringe, (start_h, 0, counter, state, [], 0))
        closed: set[Key] = set()
        best_g: dict[Key, int] = {}
        best_g[key(state)] = 0

        while fringe:
            f, neg_g, _, current, path, g = heappop(fringe)

            if current.isWin():
                return path

            cur_key = key(current)
            if cur_key in closed:
                continue
            closed.add(cur_key)

            successors = current.generatePacmanSuccessors()
            if successors is None:
                continue

            for next_state, action in successors:
                next_key = key(next_state)
                if next_key in closed:
                    continue

                next_g = g + 1
                if next_key in best_g and best_g[next_key] <= next_g:
                    continue

                best_g[next_key] = next_g
                next_h = self.h(next_state)
                counter += 1
                heappush(
                    fringe,
                    (
                        next_g + next_h,
                        -next_g,
                        counter,
                        next_state,
                        path + [action],
                        next_g,
                    ),
                )

        return []


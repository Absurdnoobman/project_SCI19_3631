from pacman_module.game import Agent, Directions


def key(state):
   
    return (
        state.getPacmanPosition(),
        state.getGhostPosition(1),
        state.getGhostDirection(1),
        state.getFood(),
    )


class PacmanAgent(Agent):

    def __init__(self):
        super().__init__()

    def get_action(self, state):
     
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

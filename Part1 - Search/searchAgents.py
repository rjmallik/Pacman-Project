"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging
import random
from pacai.core.actions import Actions
from pacai.core.search import heuristic
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.directions import Directions
from pacai.core.distance import manhattan
from pacai.student.search import breadthFirstSearch

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.
    """

    def __init__(self, startingGameState):
        super().__init__()
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))

        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning(f"Warning: no food in corner {corner}")

        self.startState = (self.startingPosition, (False, False, False, False))
        self._expanded = 0

    def startingState(self):
        return self.startState

    def isGoal(self, state):
        position, visited_corners = state
        return all(visited_corners)

    def successorStates(self, state):
        successors = []
        position, visited_corners = state

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            if not self.walls[nextx][nexty]:
                next_position = (nextx, nexty)
                new_visited_corners = list(visited_corners)

                if next_position in self.corners:
                    corner_index = self.corners.index(next_position)
                    new_visited_corners[corner_index] = True

                successors.append(((next_position, tuple(new_visited_corners)), action, 1))

        self._expanded += 1
        return successors

    def actionsCost(self, actions):
        if actions is None:
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """
    corners = problem.corners
    position, visited_corners = state

    unvisited_corners = [corner for corner, visited in zip(corners, visited_corners) if not visited]
    
    if not unvisited_corners:
        return 0

    total_distance = 0
    current_pos = position

    while unvisited_corners:
        distances = [manhattan(current_pos, corner) for corner in unvisited_corners]
        min_distance = min(distances)
        nearest_corner_index = distances.index(min_distance)

        total_distance += min_distance

        current_pos = unvisited_corners[nearest_corner_index]
        unvisited_corners.pop(nearest_corner_index)

    return total_distance
    return heuristic.null(state, problem)

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    position, foodGrid = state

    foodList = foodGrid.asList()
    if not foodList:
        return 0
   
    min_distance = float('inf')
    for food in foodList:
        distance = manhattan(position, food)
        if distance < min_distance:
            min_distance = distance

    maxFoodDistance = 0
    for i in range(len(foodList) - 1):
        for j in range(i + 1, len(foodList)):
            foodDistance = manhattan(foodList[i], foodList[j])
            if foodDistance > maxFoodDistance:
                maxFoodDistance = foodDistance

    return maxFoodDistance + min_distance
    return heuristic.null(state, problem)

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        problem = AnyFoodSearchProblem(gameState)
        return breadthFirstSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        self.food = gameState.getFood()
    
    def isGoal(self, state):
        x, y = state
        return self.food[x][y]
    
    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions."""
        if actions is None:
            return float('inf')
        x, y = self.startingState()
        cost = 0
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return float('inf')
            cost += 1
        return cost
    
class MySearchAgent(BaseAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.startingPosition = None

    def registerInitialState(self, state):
        """
        Initialize any state tracking for the agent.
        """
        self.startingPosition = state.getPacmanPosition()

    def getAction(self, state):
        """
        Return a legal action for Pacman.
        """
        legal_actions = state.getLegalActions(self.index)

        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        return random.choice(legal_actions) if legal_actions else Directions.STOP
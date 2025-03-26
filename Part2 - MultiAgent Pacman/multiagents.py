import random
from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
import math

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        Choose among the best options according to the evaluation function.
        """
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        An improved evaluation function for ReflexAgent.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        foodList = successorGameState.getFood().asList()
        ghostStates = successorGameState.getGhostStates()
        
        foodDistance = [math.dist(newPosition, food) for food in foodList]
        if foodDistance:
            closestFoodDist = min(foodDistance)
        else:
            closestFoodDist = 1
        
        ghostDistances = [math.dist(newPosition, ghost.getPosition()) for ghost in ghostStates]
        scaredTimes = [ghost.getScaredTimer() for ghost in ghostStates]
        
        dangerScore = 0
        for i, dist in enumerate(ghostDistances):
            if scaredTimes[i] == 0 and dist < 2:
                dangerScore -= 100
            elif scaredTimes[i] > 0 and dist < 2:
                dangerScore += 50
        
        score = successorGameState.getScore()
        score += 10 / closestFoodDist
        score += dangerScore
        
        return score

class MinimaxAgent(MultiAgentSearchAgent):
    """ A minimax agent. """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
    
    def getAction(self, gameState):
        """ Returns the minimax action from the current gameState. """
        def minimax(state, depth, agentIndex):
            if depth == self.getTreeDepth() or state.isWin() or state.isLose():
                return self.getEvaluationFunction()(state)
            
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.getEvaluationFunction()(state)
            
            scores = [
                minimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                for action in legalActions
            ]
            
            return max(scores) if agentIndex == 0 else min(scores)
        
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return None
        
        bestAction = max(
            legalActions,
            key=lambda action: minimax(
                gameState.generateSuccessor(0, action),
                1, 1
            )
        )
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """ A minimax agent with alpha-beta pruning. """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
    
    def getAction(self, gameState):
        """ Returns the minimax action using alpha-beta pruning. """
        def alphabeta(state, depth, agentIndex, alpha, beta):
            if depth == self.getTreeDepth() or state.isWin() or state.isLose():
                return self.getEvaluationFunction()(state)
            
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.getEvaluationFunction()(state)
            
            if agentIndex == 0:
                value = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                value = float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value
        
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return None
        
        bestAction = max(
            legalActions,
            key=lambda action: alphabeta(
                gameState.generateSuccessor(0, action),
                1, 1, float('-inf'), float('inf')
            )
        )
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """ An expectimax agent. """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
    
    def getAction(self, gameState):
        """ Returns the expectimax action from the current gameState. """
        def expectimax(state, depth, agentIndex):
            if depth == self.getTreeDepth() or state.isWin() or state.isLose():
                return self.getEvaluationFunction()(state)
            
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.getEvaluationFunction()(state)
            
            if agentIndex == 0:
                return max(
                    expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                    for action in legalActions
                )
            else:
                return sum(
                    expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                    for action in legalActions
                ) / len(legalActions)
        
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return None
        
        return max(
            legalActions,
            key=lambda action: expectimax(
                gameState.generateSuccessor(0, action),
                1, 1
            )
        )
    
def betterEvaluationFunction(currentGameState):
    """
    A stronger evaluation function that evaluates game states.
    Features considered:
    - Game score (higher is better)
    - Distance to closest food (smaller is better)
    - Ghost distances (avoid dangerous ghosts, chase scared ghosts)
    - Number of remaining food pellets (fewer is better)
    - Capsules left (fewer capsules left is good if Pac-Man can still access them)
    """
    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()
    
    score = currentGameState.getScore()
    
    if foodList:
        closestFoodDist = min(math.dist(pacmanPosition, food) for food in foodList)
        score += 10 / closestFoodDist
    
    dangerScore = 0
    for ghost in ghostStates:
        ghostDist = math.dist(pacmanPosition, ghost.getPosition())
        if ghost.getScaredTimer() == 0 and ghostDist < 2:
            dangerScore -= 100
        elif ghost.getScaredTimer() > 0 and ghostDist < 5:
            dangerScore += 50
    score += dangerScore
    
    score -= 5 * len(foodList)
    
    score -= 20 * len(capsuleList)
    
    return score
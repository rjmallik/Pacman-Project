from pacai.agents.capture.capture import CaptureAgent

class SmartOffensiveAgent(CaptureAgent):
    """
    A smarter offensive agent that efficiently collects food while avoiding ghosts.
    """
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)

    def chooseAction(self, gameState):
        """
        Picks the best action by considering food, capsules, and enemy positions.
        """
        actions = gameState.getLegalActions(self.index)
        bestAction = None
        bestScore = float('-inf')

        for action in actions:
            successor = self.getSuccessor(gameState, action)
            score = self.evaluate(successor, action)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

    def evaluate(self, gameState, action):
        """
        Evaluates the game state based on:
        - Distance to food
        - Avoiding enemies
        - Collecting capsules when necessary
        """
        myPos = gameState.getAgentState(self.index).getPosition()
        foodList = self.getFood(gameState).asList()
        capsules = self.getCapsules(gameState)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]

        score = 0

        # Prioritize eating food
        if len(foodList) > 0:
            minFoodDist = min([self.getMazeDistance(myPos, food) for food in foodList])
            score -= minFoodDist  # Closer to food is better

        # Avoid ghosts
        for ghost in ghosts:
            ghostDist = self.getMazeDistance(myPos, ghost.getPosition())
            if ghostDist < 3:  # Ghost is too close
                score -= 100  # Heavy penalty to avoid it

        # Prioritize power capsules
        if len(capsules) > 0:
            minCapsuleDist = min([self.getMazeDistance(myPos, cap) for cap in capsules])
            score -= minCapsuleDist * 0.5  # Prefer capsules, but not over food

        return score

    def getSuccessor(self, gameState, action):
        return gameState.generateSuccessor(self.index, action)


class SmartDefensiveAgent(CaptureAgent):
    """
    A defensive agent that tracks enemy Pacman and prevents food loss.
    """
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)

    def chooseAction(self, gameState):
        """
        Picks the best action to defend food and chase enemy Pacman.
        """
        actions = gameState.getLegalActions(self.index)
        bestAction = None
        bestScore = float('-inf')

        for action in actions:
            successor = self.getSuccessor(gameState, action)
            score = self.evaluate(successor, action)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

    def evaluate(self, gameState, action):
        """
        Evaluates defense by:
        - Tracking enemy Pacman
        - Guarding important food
        """
        myPos = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]

        score = 0

        # Chase enemy Pacman
        if len(invaders) > 0:
            minInvaderDist = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])
            score -= minInvaderDist  # Prioritize being closer to invaders

        # Guard important food
        foodToDefend = self.getFoodYouAreDefending(gameState).asList()
        if len(foodToDefend) > 0:
            minFoodDist = min([self.getMazeDistance(myPos, food) for food in foodToDefend])
            score -= minFoodDist * 0.5  # Stay near food

        return score

    def getSuccessor(self, gameState, action):
        return gameState.generateSuccessor(self.index, action)


def createTeam(firstIndex, secondIndex, isRed,
               first='pacai.student.SmartOffensiveAgent',
               second='pacai.student.SmartDefensiveAgent'):
    """
    Create a well-balanced team with an offensive and defensive agent.
    """
    return [SmartOffensiveAgent(firstIndex), SmartDefensiveAgent(secondIndex)]

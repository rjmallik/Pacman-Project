import random
from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util.probability import flipCoin
from pacai.util import reflection


class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.qValues = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        return self.qValues.get((state, action), 0.0)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """

        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0  # Return 0 for terminal states

        return max(self.getQValue(state, action) for action in legalActions)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """

        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        bestActions = []
        maxQValue = float('-inf')

        for action in legalActions:
            qValue = self.getQValue(state, action)
            if qValue > maxQValue:
                maxQValue = qValue
                bestActions = [action]
            elif qValue == maxQValue:
                bestActions.append(action)

        return random.choice(bestActions)

    def update(self, state, action, nextState, reward):
        """
        Updates the Q-value for (state, action) using the Q-learning update rule:
        Q(s, a) ← (1 - α) * Q(s, a) + α * (reward + γ * max_a' Q(s', a'))
        """
        alpha = self.getAlpha()
        gamma = self.getDiscountRate()

        maxNextQValue = self.getValue(nextState)
        updatedQ = (reward + gamma * maxNextQValue)
        newQValue = (1 - alpha) * self.getQValue(state, action) + alpha * updatedQ

        self.qValues[(state, action)] = newQValue

    def getAction(self, state):
        """
        Implements epsilon-greedy action selection:
        - With probability ε, takes a random action.
        - Otherwise, takes the action with the highest Q-value.

        If no legal actions are available (terminal state), return None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        epsilon = self.getEpsilon()

        if flipCoin(epsilon):
            return random.choice(legalActions)
        else:
            return self.getPolicy(state)

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, extractor='pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)
        self.weights = {}

    def getQValue(self, state, action):
        """
        Computes the Q-value for a given (state, action) pair using feature-based approximation:
        Q(s, a) = Σ [feature_i(s, a) * weight_i]
        """
        features = self.featExtractor().getFeatures(state, action)

        return sum(self.weights.get(f, 0.0) * value for f, value in features.items())

    def update(self, state, action, nextState, reward):
        """
        Updates feature weights using Q-learning update rule:
        w_i ← w_i + α * (correction) * f_i(s, a)
        
        where correction = (reward + γ * V'(s)) - Q(s, a)
        """
        alpha = self.getAlpha()
        gamma = self.getDiscountRate()

        qValue = self.getQValue(state, action)
        futureValue = self.getValue(nextState)
        correction = (reward + gamma * futureValue) - qValue

        features = self.featExtractor().getFeatures(state, action)

        for feature, value in features.items():
            self.weights[feature] = self.weights.get(feature, 0.0) + alpha * correction * value

    def final(self, state):
        """
        Called at the end of each game. Prints weights if training is complete.
        """
        super().final(state)

        if self.episodesSoFar == self.numTraining:
            print("Final learned weights:", self.weights)

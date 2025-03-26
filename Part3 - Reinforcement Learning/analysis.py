"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    Lowered noise to 0.0 to ensure deterministic movement,
    helping the agent make reliable decisions in GridWorld.
    The discount remains at 0.9 to encourage long-term planning
    and crossing the bridge in the environment.
    """

    answerDiscount = 0.9
    answerNoise = 0.0

    return answerDiscount, answerNoise

def question3a():
    """
    The agent should prefer the close exit (+1) while risking
    falling off the cliff (-10). A high discount (0.9) encourages
    prioritizing future rewards. Low noise (0.1) ensures
    deterministic movement while taking the risk. No extra
    living reward is provided.
    """

    answerDiscount = 0.3
    answerNoise = 0.0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    The agent prefers the close exit (+1) but avoids the cliff (-10).
    A lower discount (0.3) makes the agent focus more on immediate
    rewards. A higher noise (0.3) ensures randomness, making it take
    a safer route away from the cliff.
    """

    answerDiscount = 0.3
    answerNoise = 0.3
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    The agent prefers the distant exit (+10), even though it risks
    falling off the cliff (-10). A high discount (0.9) ensures the
    agent prioritizes long-term rewards. Low noise (0.1) ensures
    it moves toward the distant exit reliably.
    """

    answerDiscount = 0.99
    answerNoise = 0.0
    answerLivingReward = -0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    The agent prefers the distant exit (+10) while avoiding the cliff (-10).
    A medium discount (0.7) balances short-term and long-term rewards.
    Higher noise (0.3) introduces randomness, helping the agent avoid
    risky paths near the cliff.
    """

    answerDiscount = 0.7
    answerNoise = 0.3
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    The agent avoids both exits and keeps moving around indefinitely.
    A low discount (0.1) discourages prioritizing future rewards.
    A high living reward (1.0) encourages continuous movement,
    ensuring the agent does not exit the grid.
    """

    answerDiscount = 0.0
    answerNoise = 0.0
    answerLivingReward = 0.2

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    Adjusts Q-learning parameters for faster learning.
    - Epsilon (0.3) controls exploration-exploitation tradeoff.
    - Learning rate (0.5) ensures moderate updates per step.
    """

    return NOT_POSSIBLE


if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
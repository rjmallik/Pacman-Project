"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.priorityQueue import PriorityQueue
from pacai.util.queue import Queue

def depthFirstSearch(problem):
    stack = Stack()
    visited = set()

    stack.push((problem.startingState(), []))

    while not stack.isEmpty():
        current_state, path = stack.pop()

        if current_state in visited:
            continue

        visited.add(current_state)

        if problem.isGoal(current_state):
            return path

        for successor, action, _ in problem.successorStates(current_state):
            if successor not in visited:
                stack.push((successor, path + [action]))

    return []


def breadthFirstSearch(problem):
    queue = Queue()
    visited = set()

    queue.push((problem.startingState(), []))

    while not queue.isEmpty():
        current_state, path = queue.pop()

        if current_state in visited:
            continue

        visited.add(current_state)

        if problem.isGoal(current_state):
            return path

        for successor, action, _ in problem.successorStates(current_state):
            if successor not in visited:
                queue.push((successor, path + [action]))

    return []


def uniformCostSearch(problem):
    pq = PriorityQueue()
    visited = set()

    pq.push((problem.startingState(), [], 0), 0)

    while not pq.isEmpty():
        current_state, path, current_cost = pq.pop()

        if current_state in visited:
            continue

        visited.add(current_state)

        if problem.isGoal(current_state):
            return path

        for successor, action, step_cost in problem.successorStates(current_state):
            if successor not in visited:
                new_cost = current_cost + step_cost
                pq.push((successor, path + [action], new_cost), new_cost)

    return []


def aStarSearch(problem, heuristic):
    pq = PriorityQueue()
    visited = set()

    start_state = problem.startingState()
    pq.push((start_state, [], 0), 0 + heuristic(start_state, problem))

    while not pq.isEmpty():
        current_state, path, current_cost = pq.pop()

        if current_state in visited:
            continue

        visited.add(current_state)

        if problem.isGoal(current_state):
            return path

        for successor, action, step_cost in problem.successorStates(current_state):
            if successor not in visited:
                new_cost = current_cost + step_cost
                heuristic_cost = heuristic(successor, problem)
                total_cost = new_cost + heuristic_cost

                pq.push((successor, path + [action], new_cost), total_cost)

    return []

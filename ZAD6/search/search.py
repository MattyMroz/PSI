# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # ZAD 1 LIFO

    stack = util.Stack()  # Stos
    visited = set()  # Zbiór odwiedzonych stanów

    # Dodajemy stan początkowy na stos
    startState = problem.getStartState()
    stack.push((startState, []))  # (stan, akcje)

    while not stack.isEmpty():
        state, actions = stack.pop()  # Pobieramy jaki stan

        if problem.isGoalState(state):  # Sprawdzamy czy to cel
            return actions

        if state in visited:  # Pomijamy jeśli już odwiedzony
            continue

        visited.add(state)  # Oznaczamy jako odwiedzony

        # Dodajemy następnych na stos
        for successor, action, stepCost in problem.getSuccessors(state):
            # Jeśli jeszcze nie odwiedzony
            if successor not in visited:
                # Nowa ścieżka = stara ścieżka + nowa akcja
                newActions = actions + [action]
                stack.push((successor, newActions))

    return []  # Brak rozwiązania


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # ZAD 2 FIFO

    queue = util.Queue()  # Kolejka
    visited = set()  # Zbiór odwiedzonych stanów

    # Dodajemy stan początkowy do kolejki
    startState = problem.getStartState()
    queue.push((startState, []))  # (stan, akcje)

    while not queue.isEmpty():
        state, actions = queue.pop()  # Pobieramy jaki stan

        if problem.isGoalState(state):  # Sprawdzamy czy to cel
            return actions

        if state in visited:  # Pomijamy jeśli już odwiedzony
            continue

        visited.add(state)  # Oznaczamy jako odwiedzony

        # Dodajemy następnych do kolejki
        for successor, action, stepCost in problem.getSuccessors(state):
            # Jeśli jeszcze nie odwiedzony
            if successor not in visited:
                newActions = actions + [action]
                queue.push((successor, newActions))

    return []  # Brak rozwiązania


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # ZAD 3

    pq = util.PriorityQueue()  # Kolejka priorytetowa
    visited = set()  # Zbiór odwiedzonych stanów (wystarczy set)

    # Dodajemy stan początkowy
    startState = problem.getStartState()
    pq.push((startState, [], 0), 0)  # ((stan, akcje, koszt), priorytet)

    while not pq.isEmpty():
        state, actions, cost = pq.pop()

        if problem.isGoalState(state):  # Sprawdzamy czy to cel
            return actions

        # Pomijamy jeśli już odwiedzony (pierwsze wyjęcie = najtańsze)
        if state in visited:
            continue

        visited.add(state)  # Oznaczamy jako odwiedzony

        # Dodajemy następnych do kolejki
        for successor, action, stepCost in problem.getSuccessors(state):
            newCost = cost + stepCost  # Obliczamy nowy koszt
            # Jeśli jeszcze nie odwiedzony
            if successor not in visited:
                newActions = actions + [action]
                pq.push((successor, newActions, newCost), newCost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # ZAD 4

    pq = util.PriorityQueue()  # Kolejka priorytetowa
    visited = set()  # Zbiór odwiedzonych stanów (wystarczy set)

    startState = problem.getStartState()  # Stan początkowy
    h = heuristic(startState, problem)
    # Priorytet to f(n) = g(n) + h(n), tutaj g=0
    pq.push((startState, [], 0), 0 + h)

    while not pq.isEmpty():
        state, actions, cost = pq.pop()  # cost = g(n)

        if problem.isGoalState(state):  # Sprawdzamy czy to cel
            return actions

        # Sprawdzamy czy już tu byliśmy (pierwsze wyjęcie = optymalne g)
        if state in visited:
            continue

        visited.add(state)

        # Dodajemy następnych do kolejki
        for successor, action, stepCost in problem.getSuccessors(state):
            # Jeśli jeszcze nie odwiedzony (nie zamknięty)
            if successor not in visited:
                newCost = cost + stepCost       # g(n)
                h = heuristic(successor, problem)  # h(n)
                newActions = actions + [action]

                priority = newCost + h  # f(n) = g(n) + h(n)
                pq.push((successor, newActions, newCost), priority)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

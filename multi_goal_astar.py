from astar import AStar
from functools import lru_cache
from random import random

MULTI_GOAL_NODE = None

class MultiGoalAStar(AStar):

    """Artificially adds a zero-cost edge node that connects to multiple end goals.  A* is then run in reverse starting from the zero-cost node."""

    def _heuristic_cost_estimate(self, n1, n2):
        pass

    def heuristic_cost_estimate(self, n1, n2):
        if n1 == MULTI_GOAL_NODE or n2 == MULTI_GOAL_NODE:
            return 0
        return self._heuristic_cost_estimate(n1, n2)

    def _distance_between(self, n1, n2):
        pass

    def distance_between(self, n1, n2):
        if n1 == MULTI_GOAL_NODE or n2 == MULTI_GOAL_NODE:
            return 0
        return self._distance_between(n1, n2)

    def _neighbors(self, node):
        pass

    def neighbors(self, node):
        if node == MULTI_GOAL_NODE:
            return self.goals
        neighbors = self._neighbors(node)
        if node in self.goals:
            neighbors.append(MULTI_GOAL_NODE)
        return neighbors

    def multi_astar(self, start, goals, reversePath=False):
        self.goals = set(goals)
        result_path = list(self.astar(start, MULTI_GOAL_NODE, not reversePath))
        self.goals = []
        if reversePath:
            return result_path[0:len(result_path)-1]
        return result_path[1:len(result_path)]

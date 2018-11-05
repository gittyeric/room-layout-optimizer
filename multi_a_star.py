from queue import PriorityQueue
from functools import lru_cache
from geo import cartesian_distance, is_reachable, obstacles_to_points, obstacle_to_contact_points

MULTI_GOAL_NODE = None

class MultiAStar():

    def __init__(self, all_nodes, obstacles):
        self.nodes = all_nodes
        self.obstacles = obstacles
        self.cur_goals = None

    def heuristic(self, n1, n2):
        if n1 == MULTI_GOAL_NODE or n2 == MULTI_GOAL_NODE:
            return 0
        return cartesian_distance(n1[0], n1[1], n2[0], n2[1])


    def cost(self, n1, n2):
        if n1 == MULTI_GOAL_NODE or n2 == MULTI_GOAL_NODE:
            return 0
        return cartesian_distance(n1[0], n1[1], n2[0], n2[1])


    @lru_cache(maxsize=512)
    def real_neighbors(self, node):
        return [n for n in self.nodes if not n == node and is_reachable(node, n, self.obstacles)]

    def neighbors(self, node):
        if node == MULTI_GOAL_NODE:
            return self.cur_goals
        return self.real_neighbors(node)


    # Returns the cost of the shortest path
    def a_star(self, start, goal):
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            current = frontier.get()

            if current == goal:
                break

            for next in self.neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current

        if frontier.empty():
            return None

        print("star search cost")
        print(cost_so_far[goal])

        return cost_so_far[goal]

    def multi_goal_a_star(self, start_point, goal_points):
        fake_end_goal = MULTI_GOAL_NODE
        self.cur_goals = goal_points
        cost = self.a_star(start_point, fake_end_goal)
        self.cur_goals = []
        return cost

    def obstacle_a_star(self, start_obstacle, end_obstacle):
        start_point = obstacle_to_contact_points(start_obstacle)[0]
        goal_points = obstacle_to_contact_points(end_obstacle)
        return self.multi_goal_a_star(start_point, goal_points)

def build_pathfinder(all_obstacles):
    return MultiAStar(obstacles_to_points(all_obstacles), all_obstacles)
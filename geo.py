from scipy.spatial.distance import euclidean
from shapely.geometry import Point, Polygon, LineString, box
from shapely import affinity
from shapely.ops import transform
import math
from astar import AStar

from multi_goal_astar import MultiGoalAStar
from functools import lru_cache


@lru_cache(maxsize=4096)
def cartesian_distance(x1, y1, x2, y2):
    return euclidean((x1, y1), (x2, y2))


# A lazy, inaccurate approximation of reachability
@lru_cache(maxsize=2048)
def lazy_is_reachable(start_point, end_point, room_width, room_height):
    third_room_diameter = math.sqrt(room_width*room_width + room_height*room_height) / 3
    return cartesian_distance(start_point[0], start_point[1], end_point[0], end_point[1]) < third_room_diameter

@lru_cache(maxsize=4096)
def create_line(start_point, end_point):
    return LineString([start_point, end_point])

@lru_cache(maxsize=2048)
def get_bounding_box(bounds):
     return box(*bounds)

def is_reachable(start_point, end_point, all_obstacles):
    path_line = create_line(start_point, end_point)
    line_box = path_line.bounds

    box_collisions = []
    for obstacle in all_obstacles:
        if rect_area(line_box, obstacle.shape.bounds) > 0:
            box_collisions.append(obstacle)

    for obstacle in box_collisions:
        if obstacle.shape.crosses(path_line):
            return False
    return True


def is_reachable2(start_point, end_point, room_minus_obstacles):
    path_line = create_line(start_point, end_point)
    return room_minus_obstacles.contains(path_line)


@lru_cache(maxsize=1024)
def obstacle_to_points(obstacle):
    x, y = obstacle.shape.exterior.coords.xy
    return [(x[i], y[i]) for i in range(len(x))]


# Also returns a reverse mapping of points to indicies in the points array
def obstacles_to_points(obstacles, include_point_map=False):
    nodes = []
    point_to_index_map = {}
    for obstacle in obstacles:
        points = obstacle_to_points(obstacle)
        if include_point_map:
            cur_index = len(nodes)
            for i in range(len(points)):
                point_to_index_map[points[i]] = cur_index + i
        nodes += points
    return nodes, point_to_index_map


# Also returns a reverse mapping of points to indicies in the points array
def obstacles_to_contact_points(obstacles, include_point_map=False):
    nodes = []
    point_to_index_map = {}
    for obstacle in obstacles:
        points = obstacle_to_contact_points(obstacle)
        if include_point_map:
            cur_index = len(nodes)
            for i in range(len(points)):
                point_to_index_map[points[i]] = cur_index + i
        nodes += points
    return nodes, point_to_index_map


@lru_cache(maxsize=1024)
def obstacle_to_contact_points(obstacle):
    x, y = obstacle.shape.exterior.coords.xy
    return [(x[i], y[i]) for i in range(len(x)) if i in obstacle.contact_point_indices]


class ObstaclePathfinder(MultiGoalAStar):

    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, all_nodes, all_obstacles, room_width, room_height):
        self.nodes = all_nodes
        self.obstacles = all_obstacles
        self.room_width = room_width
        self.room_height = room_height

    # Artifically inflate or deflate the cartesian distance between points based
    # on how much you're moving toward an end goal as the crow flies. This means A* will be greedy
    # and not 100% accurate, both for performance and to be more human-like.
    # Use cartesian distance alone for perfect but slow results.
    def _heuristic_cost_estimate(self, n1, n2):
        distance = cartesian_distance(n1[0], n1[1], n2[0], n2[1])
        min_cost_dists = [math.inf, math.inf]
        for goal in self.goals:
            min_cost_dists[0] = min(min_cost_dists[0], cartesian_distance(n1[0], n1[1], goal[0], goal[1]))
            min_cost_dists[1] = min(min_cost_dists[1], cartesian_distance(n2[0], n2[1], goal[0], goal[1]))

        return distance * (min_cost_dists[1] / min_cost_dists[0])

    def _distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        return cartesian_distance(n1[0], n1[1], n2[0], n2[1])

    @lru_cache(maxsize=1024)
    def _neighbors(self, node):
        # Return all points reachable from this node
        return [n for n in self.nodes if not n == node and is_reachable(node, n, self.obstacles)] #lazy_is_reachable(node, n, self.room_width, self.room_height)]

    def minimal_path(self, start_obstacle, end_obstacle):
        return self.multi_astar(obstacle_to_points(start_obstacle)[0], obstacle_to_points(end_obstacle))

    @lru_cache(maxsize=1024)
    def minimal_cost(self, start_obstacle, end_obstacle):
        path_solution = self.minimal_path(start_obstacle, end_obstacle)
        if path_solution is None:
            return None
        cost = 0
        for i in range(1, len(path_solution)):
            cost += cartesian_distance(path_solution[i - 1][0], path_solution[i - 1][1], path_solution[i][0], path_solution[i][1])
        return cost

    # Shortest path if you could walk through obstacles
    @lru_cache(maxsize=1024)
    def minimal_unreal_cost(self, start_obstacle, end_obstacle):
        start_point = obstacle_to_contact_points(start_obstacle)[0]
        end_points = obstacle_to_contact_points(end_obstacle)
        return min([cartesian_distance(start_point[0], start_point[1], p[0], p[1]) for p in end_points])


def build_pathfinder(all_obstacles, room_width, room_height):
    return ObstaclePathfinder(obstacles_to_points(all_obstacles), all_obstacles, room_width, room_height)


@lru_cache(maxsize=1024)
def line_intersection_len(p1a, p1b, p2a, p2b):
    left_a = p1a
    left_b = p1b
    right_a = p2a
    right_b = p2b

    if p1a > p2a:
        left_a = p2a
        left_b = p2b
        right_a = p1a
        right_b = p1b

    # Left line is completely before right
    if left_b <= right_a:
        return 0
    # Left line partly intersects right
    if left_a <= right_a and left_b < right_b:
        return left_b - right_a
    # Right line is subset of left line
    elif left_a <= right_a and left_b >= right_b:
        return right_b - right_a


@lru_cache(maxsize=1024)
def rect_area(bounds1, bounds2):
    return abs(line_intersection_len(bounds1[0], bounds1[2], bounds2[0], bounds2[2]) *
               line_intersection_len(bounds1[1], bounds1[3], bounds2[1], bounds2[3]))


# Returns a bounding-box approximation of intersection area, not exact for non-zero values!
def intersection_area(shape1, shape2):
    if shape1.intersects(shape2) or shape1.overlaps(shape2):
        bounds1 = shape1.bounds
        bounds2 = shape2.bounds
        area = rect_area(bounds1, bounds2)
        return area
    else:
        return 0


def _transform_polygon(poly, x, y, r):
    return affinity.rotate(transform(lambda a, b, z=None: (a+x, b+y), poly), r, 'centroid', use_radians=True)


def transform_polygons(flat_xyr_tuples, polygons):
    return [
        _transform_polygon(poly, flat_xyr_tuples[i * 3], flat_xyr_tuples[i * 3 + 1], flat_xyr_tuples[i * 3 + 2]) for i, poly in enumerate(polygons)
    ]

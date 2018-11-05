import random
import geo
from scipy.optimize import basinhopping, Bounds
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import box
import math
from display import SolutionCanvas
from functools import lru_cache
from random import randrange
import threading
import multiprocessing
import numpy as np
import time



ALL_POINTS_OK_FOR_CONTACT = None

# Container for the Shapely shape and all the points that can be touched during a trip to this obstacle
class Obstacle():

    def __init__(self, shape, contact_point_indices=ALL_POINTS_OK_FOR_CONTACT):
        self.shape = shape
        self.contact_point_indices = contact_point_indices

        if contact_point_indices == ALL_POINTS_OK_FOR_CONTACT:
            self.contact_point_indices = [i for i in range(len(shape.exterior.coords.xy))]


# Randomly jiggle half of the inputs by up to +- stepsize/2
class Stepper(object):

    def __init__(self, stepsize):
        self.stepsize = stepsize

    def __call__(self, x):
        for i in range(0, len(x), 2):
            index = randrange(0, len(x))
            x[index] += randrange(-int(max(1, self.stepsize/2)), int(max(1, self.stepsize/2)))
        return x


def minimize(optimizer, cost_fnc, improvement, best_solution, lower_bounds, upper_bounds, options):
    return optimizer._minimize(cost_fnc, improvement, best_solution, lower_bounds, upper_bounds, options)


class FuncThread(threading.Thread):
    def __init__(self, target, args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)


def obstacles_to_distance_map(obstacles, points, room_width, room_height, impossible_path_len = math.inf):
    point_len = len(points)
    distance_map = np.zeros((point_len, point_len), dtype='float32')

    room_minus_obstacles = box(0, 0, room_width, room_height)
    for obstacle in obstacles:
        room_minus_obstacles = room_minus_obstacles.difference(obstacle.shape)

    now2 = int(round(time.time() * 1000))

    for i in range(point_len):
        for j in range(i+1, point_len):
            pi = points[i]
            pj = points[j]
            distance = impossible_path_len
            if geo.is_reachable2(pi, pj, room_minus_obstacles):
                distance = geo.cartesian_distance(pi[0], pi[1], pj[0], pj[1])
            distance_map[i, j] = distance
            distance_map[j, i] = distance

    now3 = int(round(time.time() * 1000))
    print("With reach check: " + str(now3 - now2))

    return distance_map

class RoomOptimizer():

    def __init__(self,
                 room_width,
                 room_height,
                 max_fatness_width,
                 obstacles,
                 trips,
                 fixed_obstacles=[],
                 preview_dimensions=None):

        # Room dimensions
        self.room_width = room_width
        self.room_height = room_height

        # The max fatness of the walkers traversing paths
        # Increase to give more walking room between objects
        self.max_fatness_width = 2

        # Obstacles
        # Shapely polygon instances that are to be arranged for optimization (unless is_fixed is True)
        # Fixed obstacles cannot be moved and are only used in trips
        self.obstacles = obstacles
        self.shapes = [obstacle.shape for obstacle in obstacles]

        self.fixed_obstacles = fixed_obstacles
        self.fixed_shapes = [fixed.shape for fixed in fixed_obstacles]

        self.bloated_fixed_obstacles = [Obstacle(obstacle.shape.buffer(max_fatness_width / 2.0), obstacle.contact_point_indices) for obstacle in fixed_obstacles]
        self.bloated_fixed_shapes = [obstacle.shape for obstacle in self.bloated_fixed_obstacles]

        self.all_shapes = self.shapes + self.fixed_shapes

        # Trips:
        # The trips that need to be traversed between objects to accomplish a task
        # This is an array of index arrays, where each index maps to the index of the rectangles in the room
        # Kind of lazy to make you set this here but oh well
        self.trips = trips

        # Keep the best
        self.best_cost = math.inf
        self.best_solution = None

        # Precompute some stuff
        self.total_visits = 0
        for trip in self.trips:
            self.total_visits += len(trip)
        self.avg_visits_per_trip = self.total_visits / len(self.trips)

        # This should force all possible solutions to be "cheaper" than impossible ones
        self.impossibility_scalar = self.room_width * self.room_height * len(self.obstacles) * self.total_visits

        self.canvas = None
        if preview_dimensions is not None:
            self.canvas = SolutionCanvas(preview_dimensions[0], preview_dimensions[1], room_width, room_height, self.fixed_shapes)


    def _real_cost(self, args):
        return self._cost(args, False)

    def _quick_cost(self, args):
        return self._cost(args, True)

    # To be SciPi friendly, takes a flattened array of 5 entries per object: (X, Y, Width, Height, Radian rotation)
    def _cost(self, args, quick_estimate): # args: Flat array of values, implicitly grouped in 3's (x, y, z)

        # Apply the input state to all the polygons
        input_shapes = geo.transform_polygons(args, self.shapes)
        all_updated_shapes = input_shapes + self.fixed_shapes

        # Hard Constraints
        # Here hard constraints are implemented as really expensive soft constraints,
        # since we're using an NP solver that tries to optimize locally by greedily following
        # gradients; making really impossible states cost more than almost possible states
        # will help the solver converge to a local optimum faster.
        cost = self._hard_cost(input_shapes, all_updated_shapes)

        # Don't bother calculating expensive soft constraints if hard constraints aren't even met
        if cost == 0:
            # Soft Constraints
            # Sum up general costs that are tolerable but should be minimized
            cost = self._path_cost(input_shapes, quick_estimate)

        # print("cost")
        # print(cost)

        return cost

    def _hard_cost(self, input_shapes, all_shapes):
        cost = 0
        # Add an unreasonable extra cost for physically overlapping obstacle placements
        for i in range(len(input_shapes)):
            for j in range(i + 1, len(all_shapes)):
                shape_i = all_shapes[i]
                shape_j = all_shapes[j]
                intersection_area = geo.intersection_area(shape_i, shape_j)
                cost += self.impossibility_scalar * intersection_area

        # Add an unreasonable cost for being placed outside the room
        for shape in input_shapes:
            bounds = shape.bounds
            t = pow(self.impossibility_scalar, 2) * (
                max(0, -bounds[0]) +
                max(0, -bounds[1]) +
                max(0, bounds[2] - self.room_width) +
                max(0, bounds[3] - self.room_height)
            )
            cost += t
        return cost

    def _path_cost(self, input_shapes, quick_estimate):
        # 'bloated' obstacles are grown to half the max fatness of the walker to ensure
        # that any solution allows ample walking room, and allows the walker to be represented as a
        # point traveling through space (creating a line) rather than reason about a circle
        bloated_shapes = [shape.buffer(self.max_fatness_width / 2.0) for shape in input_shapes]
        all_bloated_shapes = bloated_shapes + self.bloated_fixed_shapes
        all_obstacles = self.obstacles + self.fixed_obstacles
        all_bloated_obstacles = [Obstacle(shape, all_obstacles[i].contact_point_indices) for i, shape in
                                 enumerate(all_bloated_shapes)]

        all_points, point_to_index = geo.obstacles_to_points(all_bloated_obstacles, include_point_map = not quick_estimate)
        if quick_estimate:
            return self._quick_path_cost(all_bloated_obstacles, all_points)
        return self._real_cost(all_bloated_obstacles, all_points, point_to_index)

    def _real_path_cost(self, all_bloated_obstacles, all_points, point_to_index):
        cost = 0
        print("===========DIKSTRA=============")
        crow_flies_distances = obstacles_to_distance_map(all_bloated_obstacles, all_points, self.room_width,
                                                         self.room_height, self.impossibility_scalar)
        print("here we go: " + str(len(all_points)))
        real_distances = shortest_path(crow_flies_distances, method='D', directed=False, return_predecessors=False)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        for trip in self.trips:
            trip_len = len(trip)
            last_point = None
            for t in range(trip_len):
                obstacle = all_bloated_obstacles[trip[t]]
                if t == 0:
                    last_point = geo.obstacle_to_contact_points(obstacle)[0]
                if t > 0:
                    last_obstacle = all_bloated_obstacles[trip[t - 1]]
                    path_cost = math.inf
                    cur_point = None
                    # Greedily jump to the closest contact point of the next obstacle
                    for contact_point in geo.obstacle_to_contact_points(obstacle):
                        distance = real_distances[point_to_index[last_point], point_to_index[contact_point]]
                        if distance < path_cost:
                            cur_point = contact_point
                            path_cost = distance
                    last_point = cur_point

                    if path_cost == None or path_cost == math.inf:
                        print("gooooooooooooooooooooooooooooooooooooooooooooooooooooooddddddddddddddd")
                        cost += self.impossibility_scalar * (trip_len - t)
                    else:
                        cost += path_cost
        return cost

    def _quick_path_cost(self, all_bloated_obstacles, all_points):
        cost = 0
        # Minimize time to take all trips by summing up total distance as the cost
        a_star_solver = geo.build_pathfinder(all_bloated_obstacles, self.room_width, self.room_height)

        for trip in self.trips:
            trip_len = len(trip)
            for t in range(1, trip_len):
                obstacle = all_bloated_obstacles[trip[t]]
                last_obstacle = all_bloated_obstacles[trip[t - 1]]
                path_cost = a_star_solver.minimal_unreal_cost(last_obstacle, obstacle)
                cost += path_cost
        return cost


    def _greedy_distance(self, start_point, destination_obstacle, all_bloated_obstacles):
        pass


    # Find unreachable island obstacles and penalize
    def _island_costs(self, all_bloated_obstacles, all_points):
        cost = 0



        for trip in self.trips:
            last_point = geo.obstacle_to_contact_points(all_bloated_obstacles[trip[0]])[0]
            for t in range(1, len(trip)):
                obstacle = all_bloated_obstacles[trip[t]]
                path_cost, destination_point = self._greedy_distance(last_point, obstacle, all_bloated_obstacles)
                cost += path_cost
                last_point = destination_point

        return cost


    def _cost_with_fixed_rotation(self, obstacle_xys):
        fixed_rotations = [None]*math.floor(len(obstacle_xys) / 2) * 3
        for i in range(0, len(obstacle_xys), 2):
            fixed_rotations[math.floor(i/2)*3] = obstacle_xys[i]
            fixed_rotations[math.floor(i/2)*3+1] = obstacle_xys[i + 1]
            fixed_rotations[math.floor(i/2)*3+2] = 0

        return self._quick_cost(fixed_rotations)


    def _get_random_placement(self, obstacle):
        rectangle = obstacle.shape.bounds
        width = rectangle[2] - rectangle[0]
        height = rectangle[3] - rectangle[1]

        return (random.randrange(0, self.room_width - width, 1),
                random.randrange(0, self.room_height - height, 1))

    def _xys_to_fixed_rotation_xyrs(self, flattened_xys):
        flattened = [None]*int(3 * len(flattened_xys) / 2)
        for i in range(0, len(flattened_xys), 2):
            index3 = math.floor(i/2*3)
            flattened[index3] = flattened_xys[i]
            flattened[index3 + 1] = flattened_xys[i+1]
            flattened[index3 + 2] = 0
        return flattened

    def _bound_to_constraints(self, lower_bounds, upper_bounds):
        cons = []
        for bi in range(len(lower_bounds)):
            l = {'type': 'ineq',
                 'fun': lambda x, i=bi: x[i] - lower_bounds[i]}
            u = {'type': 'ineq',
                 'fun': lambda x, i=bi: upper_bounds[i] - x[i]}
            cons.append(l)
            cons.append(u)
        return cons

    def _improvement(self, xyrs, f, accept):
        if f < self.best_cost:
            self.best_solution = xyrs
            self.best_cost = f
            print('Got new minimum cost: ' + str(f))

    def _visual_improvement(self, xyrs, f, accept):
        if not (self.canvas is None):
            self.canvas.paint_partial(geo.transform_polygons(xyrs, self.shapes))
        if f < self.best_cost:
            self._improvement(xyrs, f, accept)
            if not (self.canvas is None):
                self.canvas.paint_solution(geo.transform_polygons(xyrs, self.shapes))

    def _decent_improvement(self, xys, f, accept):
        xyrs = self._xys_to_fixed_rotation_xyrs(xys)
        self._improvement(xyrs, f, accept)

    def _visual_decent_improvement(self, xys, f, accept):
        xyrs = self._xys_to_fixed_rotation_xyrs(xys)
        self._visual_improvement(xyrs, f, accept)

    def _minimize(self, cost_fnc, improve_fnc, init_obstacle_params, lower_bounds, upper_bounds, options):
        print("mininnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
        longest_line_len = math.sqrt(pow(self.room_height, 2) + pow(self.room_width, 2))
        step_size = 0.1  # On average, barely shake up the current inputs (this converges to near zero anyway!)
        take_step = Stepper(step_size)
        all_options = {
            'niter': self.room_height * self.room_width * self.total_visits,
            # Tempurature should be proportion to 'differences between local optimas',
            # in this case, local optimas should not differ proportionally to more than roughly the average trip distance
            'T': longest_line_len * self.avg_visits_per_trip,
            'callback': improve_fnc,
            'take_step': take_step,
            'disp': False,
            'stepsize': step_size,
            'niter_success': pow(self.room_height * self.room_width * len(self.trips), 2),
        }
        if options:
            all_options.update(options)

        print('init params')
        print(init_obstacle_params)

        constraints = self._bound_to_constraints(lower_bounds, upper_bounds)
        solution = basinhopping(cost_fnc, init_obstacle_params, \
                                niter=all_options["niter"], \
                                stepsize=all_options["stepsize"], \
                                take_step=all_options["take_step"], \
                                T=all_options["T"], \
                                callback=all_options["callback"], \
                                disp=all_options["disp"], \
                                minimizer_kwargs= {"method":"COBYLA", "constraints": constraints},# "bounds":Bounds(lower_bounds, upper_bounds)}, \
                                niter_success=all_options["niter_success"])

        return {
            'cost': solution.fun,
            'obstacle_params': solution.x
        }

    # Given an array of (width, height) tuples, approximate the best configuration and cost
    # Returns: {
    #   cost: $totalDistance,
    #   obstacles: <Array of Shapely Polygons>
    #   obstacle_params: The best raw input parameters found (a flattened array of xyr tuples)
    # }
    def minimize(self, options=None):
        thread_count = 1 #max(1, multiprocessing.cpu_count() - 1)
        threads = [0] * thread_count
        random_solution = map(self._get_random_placement, self.obstacles)
        random_solution_flattened = [el for xy_tuple in random_solution for el in xy_tuple]

        # Start with the best approximation using fixed rotation to much more quickly approximate a decent solution
        # (Throwing out the least important dimension will make the NP-solver converge a lot quicker)
        decent_options = dict(options)
        decent_lower_bounds = [0]*(len(self.obstacles)*2)
        decent_upper_bounds = [self.room_width]*(len(self.obstacles)*2)
        for i in range(len(self.obstacles)):
            decent_upper_bounds[(i*2) + 1] = self.room_height
        decent_options['niter'] = int(min(self.room_height, self.room_width))

        for i in range(thread_count):
            visualize = (self.canvas is not None) and i == 0
            improve_func = self._decent_improvement
            if visualize:
                improve_func = self._visual_decent_improvement
            threads[i] = FuncThread(minimize, [self, self._cost_with_fixed_rotation, improve_func,
                                               random_solution_flattened,
                                               decent_lower_bounds, decent_upper_bounds, decent_options])
            threads[i].start()
        for i in range(thread_count):
            threads[i].join()

        print("Decent solution finding stopped")
        print("Solving for full-state approximate solutions")

        # Now for 'real world' approximation with all input state
        lower_bounds = [0] * (len(self.obstacles) * 3)
        upper_bounds = [self.room_width] * (len(self.obstacles) * 3)
        for i in range(len(self.obstacles)):
            upper_bounds[(i * 3) + 1] = self.room_height
            upper_bounds[(i * 3) + 2] = math.pi

        better_options = dict(options)
        better_options['niter'] = int(self.room_height * self.room_width)
        best_solution = self.best_solution
        for i in range(thread_count):
            visualize = (self.canvas is not None) and i == 0
            improve_func = self._improvement
            if visualize:
                improve_func = self._visual_improvement
            threads[i] = FuncThread(minimize, [self, self._quick_cost, improve_func, best_solution,
                              lower_bounds, upper_bounds, better_options])
            threads[i].start()
        for i in range(thread_count):
            threads[i].join()

        print("Full-state approximate finding stopped")
        print("Solving for full-state, perfect-cost solutions")

        best_solution = self.best_solution
        # Reset solutions since we're changing cost functions now
        self.best_solution = None
        self.best_cost = math.inf

        for i in range(thread_count):
            visualize = (self.canvas is not None) and i == 0
            improve_func = self._improvement
            if visualize:
                improve_func = self._visual_improvement
            threads[i] = FuncThread(minimize, [self, self._real_cost, improve_func, best_solution,
                                               lower_bounds, upper_bounds, options])
            threads[i].start()
        for i in range(thread_count):
            threads[i].join()
import optimizer
import biz_objects as biz

room_width = 30
room_height = 30
entrance = biz.entrance()
the_loo_obstacle = biz.the_loo()
obstacles = [biz.desk()] * 4
obstacle_len = len(obstacles)


# Everyone enters, then visits 2 other desks, the restroom, 2 more desks, then exits
def create_trip(trip_index, trip_len):
    return [trip_len + 1] + \
           [(trip_index + i) % trip_len for i in range(2)] + \
           [trip_len] + \
           [(trip_index + i) % trip_len for i in range(2)] + \
           [trip_len + 1]

def __init__():
    solver = optimizer.RoomOptimizer(
        room_width=room_width,
        room_height=room_height,
        max_fatness_width=2,
        obstacles=obstacles,
        # Create a trip for every desk
        trips=[create_trip(o, obstacle_len) for o in range(obstacle_len)],
        fixed_obstacles=[the_loo_obstacle, entrance],
        preview_dimensions=(800, 600)
    )

    print("Optimizing...")
    solution = solver.minimize({})
    print("kewl")

__init__()
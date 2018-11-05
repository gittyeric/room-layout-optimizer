from shapely.geometry import Polygon, box
from optimizer import Obstacle

def rect(width, height):
    return box(0, 0, width, height)

# Biz Obstacles
def desk():
    return Obstacle(Polygon([(0, 0), (5, 0), (5, 3), (2.5, 3 + 4), (0, 3)]), [3])

def the_loo():
    return Obstacle(box(3, 0, 6, 1))

def entrance():
    return Obstacle(box(0, 1, 1, 4))

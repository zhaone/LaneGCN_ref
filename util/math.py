import numpy as np


def point_distance_line(point, line_point1, line_point2):
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


if __name__ == '__main__':
    print(point_distance_line(np.array([0, 4]), np.array([3, 4]), np.array([0, 0])))
    print(point_distance_line(np.array([0, 4]), np.array([3, 0]), np.array([0, 0])))
    print(point_distance_line(np.array([-3, -4]), np.array([3, 4]), np.array([0, 0])))

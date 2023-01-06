import math

import numpy as np

from panda3d.core import TransformState, Point3, Point2, LMatrix3, BitMask32
from metadrive.constants import BodyName
from metadrive.constants import CollisionGroup

EPS = 1e-6


def straight_lane_dtc(
        dist_to_left_side: float, dist_to_right_side: float,
        heading_diff_cos: float, width: float, length: float,
):
    """
    Calculate the time to collision for a straight lane.

    Left side
    >>> straight_lane_dtc(10.0, 8.0, 2.0,  -0.2, 2.0, 4.5)
    3.2851020514433644

    >>> straight_lane_dtc(10.0, 8.0, 2.0,  -1.0, 2.0, 4.5)
    0.575

    Right side
    >>> straight_lane_dtc(10.0, 8.0, 2.0, 0.2, 2.0, 4.5)
    0.2851020514433644

    >>> straight_lane_dtc(10.0, 6.0, 4.0, 1.0, 2.0, 4.5)
    0.175
    """

    if heading_diff_cos == 0:
        # Or a small threshold to avoid division by zero
        return math.inf

    dist_to_edge = dist_to_right_side if heading_diff_cos > 0 else dist_to_left_side
    abs_cos = abs(heading_diff_cos)
    abs_tan = math.sqrt(1 - abs_cos ** 2) / abs_cos
    ttc_path = dist_to_edge / abs_cos - length / 2 - width / 2 * abs_tan
    ttc_path = max(ttc_path, 0)

    return ttc_path


def circular_lane_dtc(
        dist_to_left_side: float, dist_to_right_side: float,
        heading_diff_cos: float, width: float, length: float,
        direction: int, radius_inner: float, radius_outer: float,
):
    """
    Calculate the time to collision for a circular lane.

    Clockwise
    >>> circular_lane_dtc(10.0, 5.0, 5.0,  0.00, 2.0, 4.5,  1, 10, 20)
    0.975

    >>> circular_lane_dtc(10.0, 5.0, 5.0,  0.65, 2.0, 4.5,  1, 10, 20)
    2.3192815992346434

    >>> circular_lane_dtc(10.0, 5.0, 5.0,  0.70, 2.0, 4.5,  1, 10, 20)
    0.5867924164593301

    >>> circular_lane_dtc(10.0, 5.0, 5.0,  1.00, 2.0, 4.5,  1, 10, 20)
    0.275

    >>> circular_lane_dtc(10.0, 5.0, 5.0, -1.00, 2.0, 4.5,  1, 10, 20)
    0.275

    >>> circular_lane_dtc(10.0, 5.0, 5.0,  0.99, 2.0, 4.5,  1, 10, 20)
    0.2662469079600318

    Counterclockwise
    >>> circular_lane_dtc(10.0, 5.0, 5.0,  0.00, 2.0, 4.5, -1, 10, 20)
    0.975

    >>> circular_lane_dtc(10.0, 5.0, 5.0, -0.65, 2.0, 4.5, -1, 10, 20)
    2.3192815992346434

    >>> circular_lane_dtc(10.0, 5.0, 5.0, -0.70, 2.0, 4.5, -1, 10, 20)
    0.5867924164593301

    >>> circular_lane_dtc(10.0, 5.0, 5.0,  1.00, 2.0, 4.5,  1, 10, 20)
    0.275

    >>> circular_lane_dtc(10.0, 5.0, 5.0, -1.00, 2.0, 4.5,  1, 10, 20)
    0.275

    """
    heading_diff_sin = math.sqrt(1 - heading_diff_cos ** 2)  # Safe as heading_diff is in [0, pi]
    heading_diff_sin = max(heading_diff_sin, EPS)
    heading_diff_cot = heading_diff_cos / heading_diff_sin

    vehicle_half_width_on_lane = width / heading_diff_sin / 2

    # TTC to outer circle
    if direction == 1:
        # Clockwise
        d = radius_outer - (dist_to_left_side - vehicle_half_width_on_lane)
        path_to_outer = law_of_cosines(d, radius_outer, heading_diff_cos, True)  # Always True due to d < R
        ttc_path_outer = path_to_outer_front_corner = path_to_outer - (length / 2 + width / 2 * heading_diff_cot)
    else:
        # Counter-clockwise
        assert direction == -1
        d = radius_outer - (dist_to_right_side - vehicle_half_width_on_lane)
        path_to_outer = law_of_cosines(d, radius_outer, -heading_diff_cos, True)
        ttc_path_outer = path_to_outer_front_corner = path_to_outer - (length / 2 - width / 2 * heading_diff_cot)
    ttc_path_outer = max(ttc_path_outer, 0)

    if heading_diff_cos * direction <= 0:
        # Not possible to collide with the inner circle
        return ttc_path_outer

    # TTC to inner circle (potentially)
    if direction == 1:
        # Clockwise
        d = radius_inner + (dist_to_right_side - vehicle_half_width_on_lane)
        path_to_inner = law_of_cosines(d, radius_inner, heading_diff_cos, False)  # Take shorter of the two
        ttc_path_inner = path_to_inner_front_corner = path_to_inner - (length / 2 - width / 2 * heading_diff_cot)
    else:
        # Counter-clockwise
        d = radius_inner + (dist_to_left_side - vehicle_half_width_on_lane)
        path_to_inner = law_of_cosines(d, radius_inner, -heading_diff_cos, False)
        ttc_path_inner = path_to_inner_front_corner = path_to_inner - (length / 2 + width / 2 * heading_diff_cot)
    ttc_path_inner = max(ttc_path_inner, 0)

    # Take the shorter of the two
    ttc_path = min(ttc_path_outer, ttc_path_inner)
    return ttc_path


def law_of_cosines(a: float, c: float, theta_cos: float, positive: bool):
    """
    Law of cosines.

    c^2 = a^2 + b^2 - 2ab * cos(theta)

    >>> law_of_cosines(1, 3 ** 0.5, 0.5, True)
    1.9999999999999998
    """
    B = -2 * a * theta_cos
    C = a ** 2 - c ** 2
    # Solve b^2 + B * b + C = 0
    Delta = B ** 2 - 4 * C
    if Delta < 0:
        # No solution
        return math.inf
    if positive:
        b = (-B + math.sqrt(Delta)) / 2
    else:
        b = (-B - math.sqrt(Delta)) / 2
    return b


def circle_ttc(position1, position2, velocity1, velocity2, radius1, radius2):
    position21 = position2 - position1
    dist = np.linalg.norm(position21)
    if dist <= radius1 + radius2:
        return 0.0

    velocity12 = velocity1 - velocity2
    speed = max(np.linalg.norm(velocity12), EPS)
    cos = np.dot(position21, velocity12) / (dist * speed)
    if cos <= 0:
        return float('inf')

    coll_to_inter2 = dist * np.sqrt(1 - cos ** 2)
    if coll_to_inter2 > radius1 + radius2:
        return float('inf')

    coll_to_inter1 = np.sqrt((radius1 + radius2) ** 2 - coll_to_inter2 ** 2)
    dtc1 = dist * cos - coll_to_inter1
    ttc = dtc1 / (speed / 3.6)
    return ttc


def vehicle_circles(position, heading, length, width):
    d = length / 2 - width / 2
    circle1 = position + d * heading
    circle2 = position - d * heading
    return [circle1, circle2]


def vehicle_ttc(position1, position2, heading1, heading2, velocity1, velocity2, length1, length2, width1, width2):
    radius1 = np.sqrt(2) / 2 * width1
    radius2 = np.sqrt(2) / 2 * width2
    circles1 = vehicle_circles(position1, heading1, length1, width1)
    circles2 = vehicle_circles(position2, heading2, length2, width2)

    min_ttc = float('inf')
    for c1 in circles1:
        for c2 in circles2:
            ttc = circle_ttc(c1, c2, velocity1, velocity2, radius1, radius2)
            min_ttc = min(ttc, min_ttc)

    return min_ttc


def traffic_cone_ttc(vehicle_position, cone_position, heading, velocity, length, width, radius):
    veh_radius = np.sqrt(2) / 2 * width
    veh_circles = vehicle_circles(vehicle_position, heading, length, width)
    min_ttc = float('inf')
    for c1 in veh_circles:
        ttc = circle_ttc(c1, cone_position, velocity, 0, veh_radius, radius)
        min_ttc = min(ttc, min_ttc)
    return min_ttc


def traffic_barrier_circles(position, heading, length):
    d = length / 7
    circles = []
    for i in range(6):
        circles.append(position + (length / 2 - (i + 1) * d) * heading)
    return circles


def traffic_barrier_ttc(vehicle_position, barrier_position, vehicle_heading, barrier_heading,
                        velocity, vehicle_length, vehicle_width, barrier_length, barrier_width):
    veh_radius = np.sqrt(2) / 2 * vehicle_width
    veh_circles = vehicle_circles(vehicle_position, vehicle_heading, vehicle_length, vehicle_width)
    bar_radius = 2 * barrier_width
    bar_circles = traffic_barrier_circles(barrier_position, barrier_heading, barrier_length)
    min_ttc = float('inf')
    for c1 in veh_circles:
        for c2 in bar_circles:
            ttc = circle_ttc(c1, c2, velocity, 0, veh_radius, bar_radius)
            min_ttc = min(ttc, min_ttc)
    return min_ttc


def sweep_test_dtc(
        world, chassis,
        distance: float = 100.0,
        lift: float = 0.65,
        filter: BitMask32 = CollisionGroup.ContinuousLaneLine
) -> float:
    # world = self.engine.physics_world.static_world
    # chassis = self.vehicle.system.chassis

    heading_deg = chassis.transform.hpr.x + 90.0
    mat_rot = LMatrix3.rotate_mat(heading_deg)
    mat_rot_inv = LMatrix3.rotate_mat(-heading_deg)

    sweep_vec = Point2(distance, 0.0)
    sweep_vec = mat_rot.xform_vec(sweep_vec)
    lift_vec = Point3(0.0, 0.0, lift)

    lift_ts = TransformState.make_pos(lift_vec)
    shift_ts = TransformState.make_pos2d(sweep_vec)
    start_ts = lift_ts.compose(chassis.transform)
    end_ts = shift_ts.compose(start_ts)

    chassis_shape = chassis.shapes[0]
    hit_result = world.sweepTestClosest(chassis_shape, start_ts, end_ts, filter)

    if hit_result.node is not None:
        node = hit_result.node
        node_name = node.getName()
        if node_name == BodyName.Lane:
            raise ValueError
        else:
            if filter == CollisionGroup.ContinuousLaneLine:
                assert node_name in (BodyName.White_continuous_line, BodyName.Yellow_continuous_line)
            # print(node)
            from_pos = hit_result.from_pos
            hit_pos = hit_result.hit_pos
            hit_vec = hit_pos - from_pos
            hit_vec = hit_vec.xy
            hit_vec = mat_rot_inv.xform_vec(hit_vec)
            dtc = hit_vec.x - chassis_shape.half_extents_with_margin.y
            if dtc < 0.0:
                # NOTE: The rare occurance of negative dtc may be due to incorrect collision detection.
                # And in non-fatal collision, dtc may be negative after the collision.
                # import sys
                # sys.stderr.write(f"dtc <= 0, {dtc}, {from_pos}, {hit_pos}, {chassis.transform}, {node}")
                dtc = 0.0
            return dtc
    else:
        return math.inf


def cos_heading_range(dist_to_left, dist_to_right, width, min_gap, max_heading):
    dist_left = dist_to_left - width / 2 - min_gap
    dist_right = dist_to_right - width / 2 - min_gap
    road_width = dist_left + dist_right
    if dist_left <= dist_right:
        lower_heading = max(dist_left / (road_width / 2) * max_heading, 0)
        upper_heading = max_heading
    else:
        upper_heading = max(dist_right / (road_width / 2) * max_heading, 0)
        lower_heading = max_heading
    lower_cos_heading = np.cos((90 + lower_heading) * np.pi / 180)
    upper_cos_heading = np.cos((90 - upper_heading) * np.pi / 180)
    return lower_cos_heading, upper_cos_heading


if __name__ == "__main__":
    import doctest

    doctest.testmod()

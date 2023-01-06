"""
MetaDrive feasibility rules.

All functions (except from_info) should be in form of:
    def from_foo(env: 'MySafeMetaDriveEnv', *, param1: float, param2: float, ...): ...

The return value should be a tuple of bool values (feasible, infeasible).
For feasible (with AND relation)
    - True means the rule is satisfied, thus continue to check other rules.
    - False means the rule is not satisfied, thus current state cannot be proven feasible.
For infeasible (with OR relation)
    - True means the rule is satisfied, thus current state is infeasible (short circuit).
    - False means the rule is not satisfied, thus continue to check other rules.
"""

from typing import TYPE_CHECKING

from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.static_object.traffic_object import TrafficCone, TrafficBarrier
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import TerminationState, CollisionGroup

from srlnbc.env.metadrive_utils import EPS, circular_lane_dtc, cos_heading_range, straight_lane_dtc, sweep_test_dtc, traffic_barrier_ttc, traffic_cone_ttc, vehicle_ttc

if TYPE_CHECKING:
    from srlnbc.env.my_metadrive import MySafeMetaDriveEnv


def from_info(info):
    """Special case for definitive feasibility check."""
    if info[TerminationState.OUT_OF_ROAD] or info[TerminationState.CRASH]:
        return False, True
    elif info[TerminationState.SUCCESS]:
        return True, False
    else:
        return False, False


def from_speed_limit(env: 'MySafeMetaDriveEnv', *, lower_speed: float, upper_speed: float):
    speed = env.vehicle.speed  # [km/h]
    if speed <= lower_speed:
        return True, False
    elif speed > upper_speed:
        return False, True
    else:
        return False, False


def from_vehicle_ttc(env: 'MySafeMetaDriveEnv', *, lower_vehicle_ttc: float, upper_vehicle_ttc: float):
    objs = env.vehicle.lidar.get_surrounding_objects(env.vehicle)
    min_vehicle_ttc = float('inf')
    for obj in objs:
        if isinstance(obj, BaseVehicle):
            road_ttc = vehicle_ttc(
                position1=env.vehicle.position,
                position2=obj.position,
                heading1=env.vehicle.heading,
                heading2=obj.heading,
                velocity1=env.vehicle.velocity,
                velocity2=obj.velocity,
                length1=env.vehicle.LENGTH,
                length2=obj.LENGTH,
                width1=env.vehicle.WIDTH,
                width2=obj.WIDTH
            )
            min_vehicle_ttc = min(road_ttc, min_vehicle_ttc)

    if 0 <= min_vehicle_ttc <= lower_vehicle_ttc:
        return False, True
    elif min_vehicle_ttc > upper_vehicle_ttc:
        return True, False
    else:
        return False, False


def from_surrounding_ttc(env: 'MySafeMetaDriveEnv', *, infeasible_ttc: float, feasible_ttc: float):
    objs = env.vehicle.lidar.get_surrounding_objects(env.vehicle)
    min_ttc = float('inf')
    for obj in objs:
        if isinstance(obj, BaseVehicle):
            ttc = vehicle_ttc(
                position1=env.vehicle.position,
                position2=obj.position,
                heading1=env.vehicle.heading,
                heading2=obj.heading,
                velocity1=env.vehicle.velocity,
                velocity2=obj.velocity,
                length1=env.vehicle.LENGTH,
                length2=obj.LENGTH,
                width1=env.vehicle.WIDTH,
                width2=obj.WIDTH
            )
        elif isinstance(obj, TrafficCone):
            ttc = traffic_cone_ttc(
                vehicle_position=env.vehicle.position,
                cone_position=obj.position,
                heading=env.vehicle.heading,
                velocity=env.vehicle.velocity,
                length=env.vehicle.LENGTH,
                width=env.vehicle.WIDTH,
                radius=obj.RADIUS
            )
        elif isinstance(obj, TrafficBarrier):
            ttc = traffic_barrier_ttc(
                vehicle_position=env.vehicle.position,
                barrier_position=obj.position,
                vehicle_heading=env.vehicle.heading,
                barrier_heading=obj.heading,
                velocity=env.vehicle.velocity,
                vehicle_length=env.vehicle.LENGTH,
                vehicle_width=env.vehicle.WIDTH,
                barrier_length=obj.LENGTH,
                barrier_width=obj.WIDTH
            )
            # print('barrier ttc:', ttc)
        else:
            ttc = float('inf')
        min_ttc = min(ttc, min_ttc)

    if 0 <= min_ttc <= infeasible_ttc:
        return False, True
    elif min_ttc > feasible_ttc:
        return True, False
    else:
        return False, False


def from_traffic_object_ttc_sweep_test(env: 'MySafeMetaDriveEnv', *, infeasible_ttc: float, feasible_ttc: float):
    dtc = sweep_test_dtc(
        env.engine.physics_world.dynamic_world,
        env.vehicle.system.chassis,
        distance=100.0,
        lift=0.0,
        filter=CollisionGroup.TrafficObject,
    )
    if dtc == 0.0:
        # Workaround for cases after non-fatal collision
        return False, False

    ttc = dtc / max(env.vehicle.speed / 3.6, EPS)

    if 0 < ttc <= infeasible_ttc:
        return False, True
    elif ttc > feasible_ttc:
        return True, False
    else:
        return False, False


def from_road_ttc_geometric(env: 'MySafeMetaDriveEnv', *, lower_road_ttc: float, upper_road_ttc: float, min_road_edge_gap: float):
    vehicle_length, vehicle_width = env.vehicle.LENGTH, env.vehicle.WIDTH
    dist_to_left, dist_to_right = env.vehicle.dist_to_left_side, env.vehicle.dist_to_right_side
    rightmost_lane = env.vehicle.navigation.current_ref_lanes[-1]
    normalized_cos_heading = env.vehicle.heading_diff(rightmost_lane)
    cos_heading = (normalized_cos_heading - 0.5) * 2  # diff with lateral direction

    if isinstance(rightmost_lane, StraightLane):
        dtc = straight_lane_dtc(
            dist_to_left_side=dist_to_left,
            dist_to_right_side=dist_to_right,
            heading_diff_cos=cos_heading,
            width=vehicle_width,
            length=vehicle_length
        )
    elif isinstance(rightmost_lane, CircularLane):
        lane_direction = rightmost_lane.direction
        leftmost_lane = env.vehicle.navigation.current_ref_lanes[0]
        if lane_direction == -1:
            radius_inner = leftmost_lane.radius - leftmost_lane.width / 2
            radius_outer = rightmost_lane.radius + rightmost_lane.width / 2
        else:
            radius_inner = rightmost_lane.radius - rightmost_lane.width / 2
            radius_outer = leftmost_lane.radius + leftmost_lane.width / 2
        dtc = circular_lane_dtc(
            dist_to_left_side=dist_to_left,
            dist_to_right_side=dist_to_right,
            heading_diff_cos=cos_heading,
            width=vehicle_width,
            length=vehicle_length,
            direction=lane_direction,
            radius_inner=radius_inner,
            radius_outer=radius_outer
        )
    else:
        dtc = float('inf')
    if dtc < float('inf') and dtc * abs(cos_heading) <= min_road_edge_gap:
        return False, True
    return _from_road_ttc_common(dtc, env.vehicle.speed, lower_road_ttc, upper_road_ttc)


def from_road_ttc_sweep_test(env: 'MySafeMetaDriveEnv', *, lower_road_ttc: float, upper_road_ttc: float):
    if env.vehicle.dist_to_left_side <= 0 or env.vehicle.dist_to_right_side <= 0:
        # Short circuit for out of reference path
        return False, True

    dtc = sweep_test_dtc(
        env.engine.physics_world.static_world,
        env.vehicle.system.chassis,
        distance=100.0,
    )
    return _from_road_ttc_common(dtc, env.vehicle.speed, lower_road_ttc, upper_road_ttc)


def from_road_heading(env: 'MySafeMetaDriveEnv', *, infeasible_heading: float, feasible_heading: float, min_road_edge_gap: float):
    dist_to_left, dist_to_right = env.vehicle.dist_to_left_side, env.vehicle.dist_to_right_side
    rightmost_lane = env.vehicle.navigation.current_ref_lanes[-1]
    normalized_cos_heading = env.vehicle.heading_diff(rightmost_lane)
    cos_heading = (normalized_cos_heading - 0.5) * 2  # diff with lateral direction
    infeasible_cos_heading_range = cos_heading_range(
        dist_to_left=dist_to_left,
        dist_to_right=dist_to_right,
        width=env.vehicle.WIDTH,
        min_gap=min_road_edge_gap,
        max_heading=infeasible_heading
    )
    if cos_heading < infeasible_cos_heading_range[0] or cos_heading > infeasible_cos_heading_range[1]:
        return False, True
    feasible_cos_heading_range = cos_heading_range(
        dist_to_left=dist_to_left,
        dist_to_right=dist_to_right,
        width=env.vehicle.WIDTH,
        min_gap=min_road_edge_gap,
        max_heading=feasible_heading
    )
    if feasible_cos_heading_range[0] < cos_heading < feasible_cos_heading_range[1]:
        return True, False
    else:
        return False, False


def _from_road_ttc_common(dtc: float, speed: float, lower_road_ttc: float, upper_road_ttc: float):
    road_ttc = dtc / max(speed / 3.6, EPS)
    if 0 <= road_ttc <= lower_road_ttc:
        return False, True
    elif road_ttc > upper_road_ttc:
        return True, False
    else:
        return False, False

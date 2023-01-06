import functools

import numpy as np
from srlnbc.env.simple_safety_gym import SimpleEngine


def apply_rot_mat(xy: np.ndarray, rot_mat: np.ndarray, batch_size: int):
    """
    pos: (B, 2)
    rot_mat: (B, 2, 2)
    returns: (B, 2)
    """
    xy = np.broadcast_to(xy, (batch_size, 2))
    xy = xy[:, np.newaxis, :]
    xy = xy @ rot_mat
    return xy[:, 0, :]


def relative_xy(pos: np.ndarray, center: np.ndarray, rot_mat: np.ndarray):
    """
    pos: (N, 2)
    center: (B, 2)
    rot_mat: (B, 2, 2)
    returns: (B, N, 2)
    """
    rel = pos - center[:, np.newaxis, :]    # (B, N, 2)
    return rel @ rot_mat                    # (B, N, 2) 


def relative_dist_angle(pos: np.ndarray, center: np.ndarray, rot_mat: np.ndarray, angle_normalize: bool = False):
    """
    pos: (N, 2)
    center: (B, 2)
    rot_mat: (B, 2, 2)
    returns: (B, N, 3)
    """
    vec = relative_xy(pos, center, rot_mat) # (B, N, 2)
    x, y = vec[..., 0], vec[..., 1]         # (B, N)
    z = x + 1j * y                          # (B, N)
    dist = np.abs(z)                        # (B, N)
    angle = np.angle(z)                     # (B, N)
    if angle_normalize:
        angle %= (2 * np.pi)
    return dist, angle


def relative_obs(pos: np.ndarray, center: np.ndarray, rot_mat: np.ndarray, exp_gain: float):
    """
    pos: (N, 2)
    center: (B, 2)
    rot_mat: (B, 2, 2)
    returns: (B, N, 3)
    """
    dist, angle = relative_dist_angle(pos, center, rot_mat)
    return np.stack((
        np.exp(-exp_gain * dist),
        np.cos(angle),
        np.sin(angle)
    ), axis=-1)


def lidar_max(sensor: np.ndarray, bin: np.ndarray, lidar_num_bins: int):
    """
    sensor: (B, N)
    bin: (B, N)
    returns: (B, lidar_num_bins)
    Not sure if there is a better way to do this.
    """
    mask = bin[:, :, np.newaxis] == np.arange(lidar_num_bins)  # (B, N, lidar_num_bins)
    sensor = np.broadcast_to(sensor[:, :, np.newaxis], mask.shape)
    return np.max(sensor, axis=1, where=mask, initial=0.0)


def obs_lidar_pseudo(
    pos: np.ndarray, center: np.ndarray, rot_mat: np.ndarray,
    lidar_num_bins: int, lidar_max_dist: float, lidar_exp_gain: float, lidar_alias: bool,
):
    """
    pos: (N, 2)
    center: (B, 2)
    rot_mat: (B, 2, 2)
    returns: (B, lidar_num_bins)
    """
    dist, angle = relative_dist_angle(pos, center, rot_mat, True)

    bin_size = (np.pi * 2) / lidar_num_bins
    bin = (angle / bin_size).astype(np.int64)  # (B, N), truncated towards 0
    bin_angle = bin * bin_size
    if lidar_max_dist is None:
        sensor = np.exp(-lidar_exp_gain * dist)
    else:
        sensor = np.maximum(lidar_max_dist - dist, 0.0) / lidar_max_dist
    lidar = lidar_max(sensor, bin, lidar_num_bins)
    if lidar_alias:
        alias = (angle - bin_angle) / bin_size
        assert np.all((alias >= 0) & (alias <= 1)), f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
        bin_plus = (bin + 1) % lidar_num_bins
        bin_minus = (bin - 1) % lidar_num_bins
        sensor_plus = alias * sensor
        sensor_minus = (1 - alias) * sensor
        lidar = np.maximum(lidar, lidar_max(sensor_plus, bin_plus, lidar_num_bins))
        lidar = np.maximum(lidar, lidar_max(sensor_minus, bin_minus, lidar_num_bins))
    return lidar


def relative_dist(pos: np.ndarray, center: np.ndarray):
    """
    pos: (N, 2)
    center: (B, 2)
    returns: (B, N)
    """
    rel = pos - center[:, np.newaxis, :]    # (B, N, 2)
    return np.sqrt(np.sum(np.square(rel), axis=-1))


def collect_obs(config: dict, robot_override: dict):
    """
    config MUST contain a fully defined layout.
    
    robot_override: {
        'batch_size': ..., # N
        'robot_loc': ...,  # (X, Y)
        'robot_vel': ...,  # (vx, vy)
        'robot_rot': ...,  # (w)
        'robot_acc': ...,  # (ax, ay)
        'robot_rot_vel': ...,  # (omega)
    }
    Either a single element or a batch of elements with same length is accepted.
    """
    # Some constants
    mag_XY = (0.0, -0.5)
    exp_gain = 1.0

    # Apply default values
    config = { **SimpleEngine.DEFAULT, **config }
    config_ns = SimpleEngine(config)
    SimpleEngine.build_observation_space(config_ns)

    known_keys = ['batch_size', 'robot_loc', 'robot_vel', 'robot_rot', 'robot_acc', 'robot_rot_vel']
    assert all(key in known_keys for key in robot_override.keys())
    batch_size = robot_override['batch_size']
    robot_loc = robot_override.get('robot_loc', config_ns.robot_locations[0])
    robot_vel = robot_override.get('robot_vel', (0., 0.))
    robot_rot = robot_override.get('robot_rot', config_ns.robot_rot)
    robot_acc = robot_override.get('robot_acc', (0., 0.))
    robot_rot_vel = robot_override.get('robot_rot_vel', 0.)

    robot_loc = np.broadcast_to(robot_loc, (batch_size, 2))
    robot_vel = np.broadcast_to(robot_vel, (batch_size, 2))
    robot_rot = np.broadcast_to(robot_rot, (batch_size, 1))
    robot_acc = np.broadcast_to(robot_acc, (batch_size, 2))
    robot_rot_vel = np.broadcast_to(robot_rot_vel, (batch_size, 1))

    # robot_rot around z-axis starts at x-axis
    robot_rot_cos, robot_rot_sin = np.cos(robot_rot), np.sin(robot_rot)
    robot_rot_mat = np.concatenate(
        (robot_rot_cos, -robot_rot_sin, robot_rot_sin, robot_rot_cos),
        axis=-1).reshape((batch_size, 2, 2))

    apply_robot_rot_mat = functools.partial(apply_rot_mat, rot_mat=robot_rot_mat, batch_size=batch_size)

    goal_locations = np.array(config['goal_locations'])
    hazards_locations = np.array(config['hazards_locations'])

    # Compute obs
    obs_dim = config_ns.observation_space.shape[0]
    obs_buf = np.empty((batch_size, obs_dim), dtype=np.float32)
    for k, s in config_ns.observation_layout.items():
        # Simplest cases: Provide as is
        if k == 'accelerometer':
            obs_buf[:, s] = robot_acc
        elif k == 'gyro':
            obs_buf[:, s] = robot_rot_vel
        elif k == 'velocimeter':
            obs_buf[:, s] = robot_vel
        # More complex cases: Need computation
        elif k == 'magnetometer':
            # Mujoco magnetic defaults to 0 -0.5 0. DO NOT CHANGE!
            # See https://mujoco.readthedocs.io/en/latest/XMLreference.html
            mag_xy = apply_robot_rot_mat(mag_XY)
            obs_buf[:, s] = mag_xy
        elif k == 'goal_pos':
            obs_buf[:, s] = relative_obs(goal_locations, robot_loc, robot_rot_mat, exp_gain)[:, 0]
        elif k == 'hazards_pos':
            obs_buf[:, s] = relative_obs(hazards_locations, robot_loc, robot_rot_mat, exp_gain).reshape((batch_size, -1))
        elif k == 'hazards_lidar':
            obs_buf[:, s] = obs_lidar_pseudo(
                hazards_locations, robot_loc, robot_rot_mat,
                config_ns.lidar_num_bins, config_ns.lidar_max_dist, config_ns.lidar_exp_gain, config_ns.lidar_alias
            )
        else:
            obs_buf[:, s] = 0

    # Compute cost
    cost_buf = np.empty((batch_size,), dtype=np.float32)
    cost = {
        'cost_hazards': 0,
    }
    hazards_dist = relative_dist(hazards_locations, robot_loc)
    hazards_min_dist = np.min(hazards_dist, axis=-1)
    cost['cost_hazards'] = config_ns.hazards_cost * (config_ns.hazards_size - hazards_min_dist)
    cost_buf = cost['cost_hazards']

    robot_override_broadcasted = {
        'robot_loc': robot_loc,
        'robot_vel': robot_vel,
        'robot_rot': robot_rot,
        'robot_acc': robot_acc,
        'robot_rot_vel': robot_rot_vel,
    }
    return obs_buf, cost_buf, robot_override_broadcasted

if __name__ == "__main__":
    import os

    from srlnbc.env.config import point_goal_config, car_goal_config, doggo_goal_config
    from srlnbc.utils.path import PROJECT_ROOT
    
    config = {
        **doggo_goal_config,
        'robot_rot': 0,
        '_seed': 0,
    }
    env = SimpleEngine(config)
    env.reset()
    hazards_pos = np.stack(env.hazards_pos, axis=0)[:, :2]
    goal_pos = env.goal_pos[:2]
    config = {
        **config,
        'hazards_locations': hazards_pos,
        'goal_locations': [goal_pos],
        'robot_locations': [(0, 0)],
    }

    n = 101
    x_lim = (-2, 2)
    y_lim = (-2, 2)
    xs = np.linspace(x_lim[0], x_lim[1], n)
    ys = np.linspace(y_lim[0], y_lim[1], n)
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    obs_buf, _, _ = collect_obs(config, {
        'batch_size': n ** 2,
        'robot_loc': np.stack((xs, ys), axis=1),
        'robot_rot': 0,
        'robot_vel': (1, 0),
    })

    path = os.path.join(PROJECT_ROOT, 'data')
    if not os.path.exists(path):
        os.makedirs(path)
    np.savez(
        os.path.join(path, 'doggo_goal_left.npz'),
        obs=obs_buf.reshape((n, n, -1)),
        x=xs.reshape((n, n)),
        y=ys.reshape((n, n)),
        hazards_pos=hazards_pos,
        goal_pos=goal_pos
    )

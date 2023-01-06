import gym.spaces
import numpy as np

from safety_gym.envs.engine import Engine

def normalize_obs(pos, robot_pos, robot_mat):
    pos = np.concatenate((pos, np.zeros((pos.shape[0], 1))), axis=-1)
    vec = (pos - robot_pos) @ robot_mat
    x, y = vec[:, 0], vec[:, 1]
    z = x + 1j * y
    dist = np.abs(z)
    dist = np.exp(-dist)
    # dist = np.minimum(dist, 4.0) / 4.0
    angle = np.angle(z) # % (2 * np.pi)
    # order = np.argsort(angle)
    # dist = dist[order]
    # angle = angle[order]
    return np.stack((dist, np.cos(angle), np.sin(angle)), axis=-1)

class SimpleEngine(Engine):
    DEFAULT = {
        **Engine.DEFAULT,
        'observe_hazards_pos': False,
        'action_scale': 1.0,  # True value scale of action space (for each dimension)

        'ttc_upper_threshold': 2.0,
        'ttc_lower_threshold': 1.0,
        'lower_distance': 0.0,
        'upper_distance': float('inf'),
    }

    def obs(self):
        self.sim.forward()
        obs = {}

        obs['accelerometer'] = self.world.get_sensor('accelerometer')[:2]
        obs['velocimeter'] = self.world.get_sensor('velocimeter')[:2]
        obs['gyro'] = self.world.get_sensor('gyro')[-1:]
        obs['magnetometer'] = self.world.get_sensor('magnetometer')[:2]
        if 'doggo.xml' in self.robot_base: self.extra_sensor_obs(obs)  # Must call after simplified sensors

        robot_pos = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        obs['goal_pos'] = normalize_obs(self.goal_pos[np.newaxis, :2], robot_pos, robot_mat)
        if self.observe_hazards_pos:
            obs['hazards_pos'] = normalize_obs(np.stack(self.hazards_pos)[:, :2], robot_pos, robot_mat)
        else:
            obs['hazards_lidar'] = self.obs_lidar(self.hazards_pos, None)
        # obs['robot_pos'] = robot_pos[:2]

        flat_obs = np.zeros(self.obs_flat_size)
        for k, v in self.observation_layout.items():
            flat_obs[v] = obs[k].flat

        return flat_obs

    def build_observation_space(self):
        obs_space_dict = {}

        obs_space_dict['accelerometer'] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        obs_space_dict['velocimeter'] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        obs_space_dict['gyro'] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        obs_space_dict['magnetometer'] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        if 'doggo.xml' in self.robot_base: self.build_extra_sensor_observation_space(obs_space_dict)  # Must call after simplified sensors

        obs_space_dict['goal_pos'] = gym.spaces.Box(-np.inf, np.inf, (1, 3), dtype=np.float32)
        if self.observe_hazards_pos:
            obs_space_dict['hazards_pos'] = gym.spaces.Box(-np.inf, np.inf, (self.hazards_num, 3), dtype=np.float32)
        else:
            obs_space_dict['hazards_lidar'] = gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        # obs_space_dict['robot_pos'] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)

        self.obs_space_dict = obs_space_dict
        self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        self.observation_layout = {}

        offset = 0
        for k in sorted(self.obs_space_dict.keys()):
            space = self.obs_space_dict[k]
            size = np.prod(space.shape)
            start, end = offset, offset + size
            self.observation_layout[k] = slice(start, end)
            offset += size
        assert offset == self.obs_flat_size

    def extra_sensor_obs(self, obs):
        from safety_gym.envs.engine import quat2mat
        for sensor in self.sensors_obs:
            if sensor in obs: continue
            obs[sensor] = self.world.get_sensor(sensor)
        for sensor in self.robot.hinge_vel_names:
            obs[sensor] = self.world.get_sensor(sensor)
        for sensor in self.robot.ballangvel_names:
            obs[sensor] = self.world.get_sensor(sensor)
        if self.sensors_angle_components:
            for sensor in self.robot.hinge_pos_names:
                theta = float(self.world.get_sensor(sensor))  # Ensure not 1D, 1-element array
                obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
            for sensor in self.robot.ballquat_names:
                quat = self.world.get_sensor(sensor)
                obs[sensor] = quat2mat(quat)
        else:  # Otherwise read sensors directly
            for sensor in self.robot.hinge_pos_names:
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.ballquat_names:
                obs[sensor] = self.world.get_sensor(sensor)

    def build_extra_sensor_observation_space(self, obs_space_dict):
        for sensor in self.sensors_obs:
            if sensor in obs_space_dict: continue
            dim = self.robot.sensor_dim[sensor]
            obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
        for sensor in self.robot.hinge_vel_names:
            obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        for sensor in self.robot.ballangvel_names:
            obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        if self.sensors_angle_components:
            for sensor in self.robot.hinge_pos_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
            for sensor in self.robot.ballquat_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (3, 3), dtype=np.float32)
        else:
            for sensor in self.robot.hinge_pos_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
            for sensor in self.robot.ballquat_names:
                obs_space_dict[sensor] = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)

    def get_time_to_collision(self):
        velocity_vec = self.world.robot_vel()[:2]
        velocity = np.linalg.norm(velocity_vec)
        velocity = np.clip(velocity, 1e-6, None)
        min_ttc = -1.0
        for hazard_pos in self.hazards_pos:
            hazard_vec = hazard_pos[:2] - self.world.robot_pos()[:2]
            dist = np.linalg.norm(hazard_vec)
            if dist <= self.hazards_size:
                min_ttc = 0.0
                break
            cos_theta = np.dot(velocity_vec, hazard_vec) / (velocity * dist)
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            delta = self.hazards_size ** 2 - (dist * sin_theta) ** 2
            if cos_theta > 0 and delta >= 0:
                ttc = (dist * cos_theta - np.sqrt(delta)) / velocity
                if min_ttc == -1.0:
                    min_ttc = ttc
                else:
                    min_ttc = min(ttc, min_ttc)
        return min_ttc

    def get_distance_to_hazard(self):
        min_dist = float('inf')
        for hazard_pos in self.hazards_pos:
            hazard_vec = hazard_pos[:2] - self.world.robot_pos()[:2]
            dist = np.linalg.norm(hazard_vec) - self.hazards_size
            min_dist = min(dist, min_dist)
        return min_dist

    def get_feasibility_info(self):
        ttc = self.get_time_to_collision()
        ttc_feasible = ttc < 0 or ttc > self.config['ttc_upper_threshold']
        ttc_infeasible = 0 <= ttc <= self.config['ttc_lower_threshold']

        dist = self.get_distance_to_hazard()
        dist_feasible = dist > self.config['upper_distance']
        dist_infeasible = dist <= self.config['lower_distance']

        if dist_feasible:
            feasible, infeasible = True, False
        elif dist_infeasible:
            feasible, infeasible = False, True
        else:
            feasible, infeasible = ttc_feasible, ttc_infeasible

        return {
            'feasible': feasible,
            'infeasible': infeasible,
        }

    def step(self, action):
        action = np.array(action, copy=False)   # Cast to ndarray
        action = action * self.action_scale     # Convert normalized action to real action
        next_obs, reward, done, info = super(SimpleEngine, self).step(action)
        if 'cost_exception' in info:
            # Simulation exception
            # Example: MujocoException Got MuJoCo Warning: Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable. Time = 2.3700.
            assert 'cost' not in info and done
            assert not np.isnan(action).any()
            info['cost'] = info['cost_exception']
        info.update(self.get_feasibility_info())
        return next_obs, reward, done, info

    def plot_map(self, func, data_path, save_path=None, name=''):
        data = np.load(data_path)
        obs = data['obs']
        hazards = data['hazards_pos']
        goal = data['goal_pos']
        xs = data['x']
        ys = data['y']

        map = func(obs)
        max_value = np.max(map)
        min_value = np.min(map)
        assert max_value > 0 and min_value < 0
        map[map > 0] /= np.max(map)
        map[map < 0] /= -np.min(map)

        import os
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(4, 3.2))
        cf = ax.contourf(xs, ys, map, levels=100, cmap='coolwarm')
        fig.colorbar(cf, shrink=0.9)
        for c in cf.collections:
            c.set_edgecolor('face')
        ax.contour(xs, ys, map, levels=[0], colors='r')

        for pos in hazards:
            circle = Circle(pos, self.hazards_size, fill=False, linestyle='--', color='k')
            ax.add_patch(circle)
        circle = Circle(goal, self.goal_size, fill=False, color='k')
        ax.add_patch(circle)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, name))
        plt.show()


if __name__ == "__main__":
    from srlnbc.env.config import point_goal_config_simple
    env = SimpleEngine(point_goal_config_simple)
    obs = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(info)

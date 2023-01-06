import os
from typing import Tuple

import gym
import numpy as np
from gym.spaces import Box


class CarBrake(gym.Env):
    def __init__(self):
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(2,), dtype=np.float32)
        self.action_space = Box(low=-2, high=1, shape=(1,), dtype=np.float32)
        self.dt = 0.1

    def reset(self) -> np.ndarray:
        s = np.random.uniform(0, 25)
        v = np.random.uniform(0, 10)
        self.state = np.array([s, v], dtype=np.float32)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        s, v = self.state
        a = np.clip(action, self.action_space.low, self.action_space.high)[0]
        new_s = s - v * self.dt
        new_v = max(v + a * self.dt, 0)
        reward = v * self.dt
        self.state[0] = new_s
        self.state[1] = new_v
        done = False
        cost = float(s < 0)

        if s <= 0:
            ttc = 0.0
        elif v == 0:
            ttc = -1.0
        else:
            ttc = s / v

        feasible = ttc < 0 or ttc > 3.0
        infeasible = 0 <= ttc <= 0.5

        info = {'cost': cost, 'feasible': feasible, 'infeasible': infeasible}
        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        return np.copy(self.state)

    def plot_map(self, func, save_path=None):
        n = 101
        x_lim = (0, 25)
        y_lim = (0, 10)
        xs = np.linspace(*x_lim, n)
        ys = np.linspace(*y_lim, n)
        x_grid, y_grid = np.meshgrid(xs, ys)
        obs = np.stack((x_grid, y_grid), axis=2)

        map = func(obs)
        max_value = np.max(map)
        min_value = np.min(map)
        assert max_value > 0 and min_value < 0
        map[map > 0] /= np.max(map)
        map[map < 0] /= -np.min(map)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 3.2))
        cf = ax.contourf(x_grid, y_grid, map, levels=100, cmap='coolwarm')
        fig.colorbar(cf)
        for c in cf.collections:
            c.set_edgecolor('face')
        ax.contour(x_grid, y_grid, map, levels=[0], colors='r')

        v_max = np.sqrt(2 * abs(self.action_space.low) * xs)
        plt.plot(xs, v_max, color='k')

        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xlabel('distance [m]')
        plt.ylabel('velocity [m/s]')
        plt.tight_layout()
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, 'car_brake_map.pdf'))
        plt.show()

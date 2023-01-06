import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from numpy import random
import os


class UnicycleEnv(gym.Env):
    def __init__(self):
        self.dt = 0.1
        self.render_initialized = False
        self.step_cnt = 0
        self.obstacle_radius = 0.5
        self.fps = 10
        self.goal = [0, 5, 0, 0.5 * np.pi]
        self.phi = None
        self.sis_info = dict()
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(5,))
        self.action_space = gym.spaces.Box(low=np.array([-4.0, -np.pi]), high=np.array([4.0, np.pi]))
        self.sis_paras = None

    def rk4(self, s, u, dt):
        dot_s1 = self.dynamics(s, u)
        dot_s2 = self.dynamics(s + 0.5 * dt * dot_s1, u)
        dot_s3 = self.dynamics(s + 0.5 * dt * dot_s2, u)
        dot_s4 = self.dynamics(s + dt * dot_s3, u)
        dot_s = (dot_s1 + 2 * dot_s2 + 2 * dot_s3 + dot_s4) / 6.0
        return dot_s

    def dynamics(self, s, u):
        v = s[2]
        theta = s[3]

        dot_x = v * np.cos(theta)
        dot_y = v * np.sin(theta)
        dot_v = u[0]
        dot_theta = u[1]

        dot_s = np.array([dot_x, dot_y, dot_v, dot_theta])
        return dot_s

    def set_sis_paras(self, sis_paras):
        self.sis_paras = sis_paras

    def step(self, u):
        u = np.multiply(u, self.action_space.high)
        u = np.clip(u, self.action_space.low, self.action_space.high)

        last_state = np.copy(self.state)
        dot_state = self.rk4(self.state, u, self.dt)
        self.state = self.state + dot_state * self.dt

        rew = self.compute_reward(last_state)

        self.step_cnt += 1
        done = False
        if self.step_cnt >= 100:
            done = True

        # state, reward, done, info
        info = {}
        info.update(dict(cost=self.compute_cost()))
        old_phi = self.phi
        self.phi = self.adaptive_safety_index()  # the self.phi only works with evaluator with fixed sis_paras
        if old_phi <= 0:
            delta_phi = max(self.phi, 0)
        else:
            delta_phi = self.phi - old_phi

        # update info dict
        info.update({'delta_phi': delta_phi})
        info.update(self.sis_info)

        ttc = self.get_time_to_collision()
        feasible = ttc < 0 or ttc > 0.1
        infeasible = ttc == 0
        info.update({'feasible': feasible, 'infeasible': infeasible})

        return self.get_obs(), rew, done, info

    def reset(self):
        self.state = np.array([
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(-1.5, 1.5),
            1. + random.random(),
            random.random() * np.pi / 2 + np.pi / 4
        ])
        self.phi = self.adaptive_safety_index()
        self.ref = np.zeros_like(self.state)
        self.step_cnt = 0
        return self.get_obs()

    def compute_reward(self, last_state):
        last_dist = np.linalg.norm([self.goal[0] - last_state[0], self.goal[1] - last_state[1]])
        dist = np.linalg.norm([self.goal[0] - self.state[0], self.goal[1] - self.state[1]])
        return last_dist - dist

    def compute_cost(self):
        dist = np.linalg.norm(self.state[:2])
        if dist <= self.obstacle_radius:
            return 1.0
        else:
            return 0.0

    def get_obs(self):
        obs = np.zeros(5, dtype=np.float32)
        obs[:3] = self.state[:3]
        theta = self.state[3]
        obs[3] = np.cos(theta)
        obs[4] = np.sin(theta)
        return obs

    def adaptive_safety_index(self, sigma=0.04, k=2, n=2):
        '''
        synthesis the safety index that ensures the valid solution
        '''
        # initialize safety index

        '''
        function phi(index::CollisionIndex, x, obs)
            o = [obs.center; [0,0]]
            d = sqrt((x[1]-o[1])^2 + (x[2]-o[2])^2)
            dM = [x[1]-o[1], x[2]-o[2], x[3]*cos(x[4])-o[3], x[3]*sin(x[4])-o[4]]
            dim = 2
            dp = dM[[1,dim]]
            dv = dM[[dim+1,dim*2]]
            dot_d = dp'dv / d
            return (index.margin + obs.radius)^index.phi_power - d^index.phi_power - index.dot_phi_coe*dot_d
        end
        '''
        if self.sis_paras is not None:
            sigma, k, n = self.sis_paras
        phi = -1e8
        sis_info_t = self.sis_info.get('sis_data', [])
        sis_info_tp1 = []

        rela_pos = self.state[:2]
        d = np.linalg.norm(rela_pos)
        robot_to_hazard_angle = np.arctan((-rela_pos[1]) / (-rela_pos[0] + 1e-8))
        vel_rela_angle = self.state[-1] - robot_to_hazard_angle
        dotd = self.state[2] * np.cos(vel_rela_angle)

        # if dotd <0, then we are getting closer to hazard
        sis_info_tp1.append((d, dotd))

        # compute the safety index
        phi_tmp = (sigma + self.obstacle_radius) ** n - d ** n - k * dotd
        # select the largest safety index
        if phi_tmp > phi:
            phi = phi_tmp

        # sis_info is a list consisting of tuples, len is num of obstacles
        self.sis_info.update(dict(sis_data=sis_info_tp1, sis_trans=(sis_info_t, sis_info_tp1)))
        return phi

    def get_unicycle_plot(self):
        theta = self.state[3]
        ang = (-self.state[3] + np.pi / 2) / np.pi * 180
        s = self.state[[0, 1]]
        t = self.state[[0, 1]] + np.hstack([np.cos(theta), np.sin(theta)])
        c = s
        s = s - (t - s)
        return np.hstack([s[0], t[0]]), np.hstack([s[1], t[1]])

    def get_time_to_collision(self):
        x, y, v, theta = self.state

        velocity_vec = np.array([v * np.cos(theta), v * np.sin(theta)])
        velocity = np.linalg.norm(velocity_vec)
        velocity = np.clip(velocity, 1e-6, None)

        hazard_vec = np.array([-x, -y])
        dist = np.linalg.norm(hazard_vec)
        if dist <= self.obstacle_radius + 0.05:
            return 0.0

        cos_theta = np.dot(velocity_vec, hazard_vec) / (velocity * dist)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        delta = self.obstacle_radius ** 2 - (dist * sin_theta) ** 2
        if cos_theta > 0 and delta >= 0:
            return (dist * cos_theta - np.sqrt(delta)) / velocity
        else:
            return -1.0

    def get_avoidable(self, state, obstacle_y):
        x, y, v, theta = state

        hazard_vec = np.array([-x, obstacle_y - y])
        dist = np.linalg.norm(hazard_vec)
        if dist <= self.obstacle_radius:
            return 1.0

        velocity_vec = np.array([v * np.cos(theta), v * np.sin(theta)])
        velocity = np.linalg.norm(velocity_vec)
        velocity = np.clip(velocity, 1e-6, None)
        cos_theta = np.dot(velocity_vec, hazard_vec) / (velocity * dist)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        delta = self.obstacle_radius ** 2 - (dist * sin_theta) ** 2
        if cos_theta <= 0 or delta < 0:
            return -1.0

        acc = self.action_space.low[0]
        if np.cross(velocity_vec, hazard_vec) >= 0:
            omega = self.action_space.low[1]
        else:
            omega = self.action_space.high[1]
        action = np.array([acc, omega])
        s = np.copy(state)
        while s[2] > 0:
            dot_s = self.rk4(s, action, self.dt)
            s = s + dot_s * self.dt
            dist = np.linalg.norm([-s[0], obstacle_y - s[1]])
            if dist <= self.obstacle_radius:
                return 1.0
        return -1.0

    def plot_map(self, func, v=1.5, theta=np.pi / 2, save_path=None):
        n = 101
        x_lim = (-1.5, 1.5)
        y_lim = (-1.5, 1.5)

        xs = np.linspace(x_lim[0], x_lim[1], n)
        ys = np.linspace(y_lim[0], y_lim[1], n)
        xs, ys = np.meshgrid(xs, ys)
        vs = v * np.ones_like(xs)
        thetas = theta * np.ones_like(xs)
        obs = np.stack((xs, ys, vs, np.cos(thetas), np.sin(thetas)), axis=-1)

        map = func(obs)
        max_value = np.max(map)
        min_value = np.min(map)
        assert max_value > 0 and min_value < 0
        map[map > 0] /= np.max(map)
        map[map < 0] /= -np.min(map)

        avoidable = np.zeros_like(xs)
        for i in range(n):
            for j in range(n):
                avoidable[i, j] = self.get_avoidable([xs[i, j], ys[i, j], v, theta], 0.0)

        fig, ax = plt.subplots(figsize=(4, 3.2))
        cf = ax.contourf(xs, ys, map, levels=100, cmap='coolwarm')
        fig.colorbar(cf, shrink=0.9)
        for c in cf.collections:
            c.set_edgecolor('face')
        ax.contour(xs, ys, map, levels=[0], colors='r')
        ax.contour(xs, ys, avoidable, levels=[0], colors='k')
        circle = Circle((0.0, 0.0), self.obstacle_radius, fill=False, linestyle='--', color='k')
        ax.add_patch(circle)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, 'unicycle_map.pdf'))
        plt.show()

    def render(self, mode='human'):
        plt.clf()
        ax = plt.axes()
        ax.add_patch(plt.Rectangle((-3.5, -2), 7.0, 8.0, edgecolor='black',
                                   facecolor='none', linewidth=2))
        ax.add_patch(plt.Rectangle((-1.5, -1.5), 3, 3, edgecolor='r', linestyle='--',
                                   facecolor='none', linewidth=2))
        plt.axis("equal")
        plt.axis('off')
        x, y, v, theta = 1.0, -1.0, 0.5, 0.55 * np.pi
        plt.arrow(x, y, v * np.cos(theta), v * np.sin(theta),
                  color='b', head_width=0.2, zorder=5)
        # plt.plot([0, 0], [-2, 6], color='grey', linestyle='--', zorder=0)
        # plt.plot([-3.5, 3.5], [0, 0], color='grey', linestyle='--', zorder=0)
        ax.add_patch(plt.Circle((0, 0), 0.5,
                                edgecolor='none', facecolor='blue', alpha=0.5, zorder=10))
        ax.add_patch(plt.Circle(self.goal[:2], 0.2,
                                edgecolor='none', facecolor='red', zorder=10))
        # self.fig.canvas.flush_events()
        # text_x, text_y_start = -6, 4
        # ge = iter(range(0, 1000, 4))
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'ego_x: {:.2f}m'.format(self.state[0]))
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'ego_y: {:.2f}m'.format(self.state[1]))
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'ego_v: {:.2f}m/s'.format(self.state[2]))
        # plt.text(text_x, text_y_start - 0.1 * next(ge),
        #          r'ego_angle: ${:.2f}\degree$'.format(self.state[3] / np.pi * 180.))
        #
        # next(ge)
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'ref_x: {:.2f}m'.format(self.ref[0]))
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'ref_y: {:.2f}m'.format(self.ref[1]))
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'ref_v: {:.2f}m/s'.format(self.ref[2]))
        # plt.text(text_x, text_y_start - 0.1 * next(ge),
        #          r'ref_angle: ${:.2f}\degree$'.format(self.ref[3] / np.pi * 180.))
        # next(ge)
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'action_v: {:.2f}m'.format(self.action[0]))
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'action_a: {:.2f}m'.format(self.action[1]))
        # next(ge)
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'reward: {:.2f}m'.format(self.compute_reward()))
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'cost: {:.2f}m'.format(self.compute_cost()))
        # next(ge)
        # d = self.sis_info.get('sis_data')[0][0]
        # dotd = self.sis_info.get('sis_data')[0][1]
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'd: {:.2f}m'.format(d))
        # plt.text(text_x, text_y_start - 0.1 * next(ge), 'dotd: {:.2f}m/s'.format(dotd))
        # time.sleep(1)
        # plt.show()
        # plt.pause(10)


if __name__ == '__main__':
    path = '../results/'
    if not os.path.exists(path):
        os.makedirs(path)
    env = UnicycleEnv()
    env.reset()
    env.render()
    plt.savefig(os.path.join(path, 'unicycle.pdf'))
    plt.show()

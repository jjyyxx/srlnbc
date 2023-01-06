from typing import Dict, Optional, Tuple

import numpy as np
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.utils import Config
from metadrive.utils.math_utils import clip, norm

from srlnbc.env.metadrive_rules import (
    from_info,
    from_speed_limit,
    from_road_ttc_geometric,
    from_road_ttc_sweep_test,
    from_vehicle_ttc,
    from_surrounding_ttc,
    from_road_heading,
    from_traffic_object_ttc_sweep_test,
)


RULE_MAP = {
    'speed_limit': from_speed_limit,
    'road_ttc_geometric': from_road_ttc_geometric,
    'road_ttc_sweep_test': from_road_ttc_sweep_test,
    'vehicle_ttc': from_vehicle_ttc,
    'surrounding_ttc': from_surrounding_ttc,
    'road_heading': from_road_heading,
    'traffic_object_ttc_sweep_test': from_traffic_object_ttc_sweep_test,
}


def seeding(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    seed_seq = np.random.SeedSequence(seed)
    seed = seed_seq.entropy
    bit_generator = np.random.PCG64(seed)
    return np.random.Generator(bit_generator), seed


class MySafeMetaDriveEnv(SafeMetaDriveEnv):
    def __init__(self, config: dict):
        feasibility = config.pop('feasibility', {})
        super().__init__(config)
        self.feasibility_config: Dict[str, dict] = feasibility

        # NOTE: Code to make MetaDriveEnv reproducible.
        # Hack 1
        #   The MetaDriveEnv would call seed() in each call to reset(),
        #   which introduces a non-deterministic behavior.
        #   But one could pass `force_seed` to reset() to make it reproducible.
        # Hack 2
        #   RLlib will seed the env by calling `env.seed(seed)`,
        #   which contradicts MetaDriveEnv's seed() behavior.
        #   We override the seed() method with two branches:
        #     1. Route to original MetaDriveEnv if called during reset().
        #     2. Create a persistent RNG and use it to generate `force_seed`
        #        if called externally.
        self.myenv_seeded = False
        self.myenv_in_reset = False

    def default_config(self) -> Config:
        config = super(MySafeMetaDriveEnv, self).default_config()
        config.update(
            {
                'crash_vehicle_penalty': 0,
                'crash_object_penalty': 0,
                'yaw_rate_penalty': 0.4,
                'steering_penalty': 0.4,
                'max_delta_steering': 0.05,
                'max_delta_acc': 0.2,
            },
            allow_add_new_key=True
        )
        return config

    def seed(self, seed=None):
        """For reproducibility, see notes in __init__."""
        if self.myenv_in_reset:
            return super().seed(seed)
        else:
            self.myenv_seeded = True
            self.myenv_rng, seed = seeding(seed)
            print(f"Note: Env seeded with {seed}")
            return [seed]

    def reset(self):
        """For reproducibility, see notes in __init__."""
        if self.myenv_seeded:
            force_seed = int(self.myenv_rng.integers(self.start_seed, self.start_seed + self.env_num))
        else:
            # NOTE: Leave this branch as a workaround for RLlib's env pre-check,
            # which is before the env is seeded.
            # NOTE: Have to be deterministic here, otherwise MetaDrive will
            # mess up the global NumPy RNG (due to different map has different
            # number of random sources).
            print("Warning: Env not seeded yet and will be deterministic!")
            force_seed = self.start_seed

        self.myenv_in_reset = True
        obs = super().reset(force_seed=force_seed)
        self.myenv_in_reset = False

        return obs

    def step(self, action):
        last_action = self.vehicle.last_current_action[-1]
        clipped_action = np.array((
            np.clip(action[0],
                    last_action[0] - self.config['max_delta_steering'],
                    last_action[0] + self.config['max_delta_steering']),
            np.clip(action[1],
                    last_action[1] - self.config['max_delta_acc'],
                    last_action[1] + self.config['max_delta_acc'])
        ))

        next_obs, reward, done, info = super(MySafeMetaDriveEnv, self).step(clipped_action)

        self.update_feasibility_info(info)

        return next_obs, reward, done, info

    def update_feasibility_info(self, info):
        # Check 1: From info
        feasible, infeasible = from_info(info)
        if feasible or infeasible:
            # Definitive feasibility info, short circuit.
            assert feasible ^ infeasible
            info['feasible'] = feasible
            info['infeasible'] = infeasible
            return

        # Check 2: From other rules
        feasible, infeasible = True, False
        for rule, params in self.feasibility_config.items():
            rule_fn = RULE_MAP[rule]
            rule_feasible, rule_infeasible = rule_fn(self, **params)
            assert not (rule_feasible and rule_infeasible)
            feasible &= rule_feasible
            infeasible |= rule_infeasible
            if infeasible:
                break
        info['feasible'] = feasible
        info['infeasible'] = infeasible

    def reward_function(self, vehicle_id: str):
        reward, step_info = super(MySafeMetaDriveEnv, self).reward_function(vehicle_id)

        vehicle = self.vehicles[vehicle_id]

        # Current yaw rate
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last) / (norm(*heading_dir_now) * norm(*heading_dir_last))
        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))
        yaw_rate = clip(beta_diff / 0.1, 0.0, 1.0)

        # Current steering
        steering = vehicle.steering  # [-1, 1]

        lane = vehicle.navigation.current_lane
        if isinstance(lane, CircularLane):
            ref_yaw_rate = vehicle.speed / 3.6 / lane.radius
            theta = np.arctan((vehicle.FRONT_WHEELBASE + vehicle.REAR_WHEELBASE) / lane.radius)
            ref_steering = -lane.direction * theta * 180 / np.pi / vehicle.max_steering
        else:
            ref_yaw_rate = 0.0
            ref_steering = 0.0

        reward -= self.config['yaw_rate_penalty'] * (yaw_rate - ref_yaw_rate) ** 2
        reward -= self.config['steering_penalty'] * (steering - ref_steering) ** 2

        return reward, step_info

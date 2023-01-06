import argparse
import os
import pickle

from srlnbc.agents.safety.ppo_barrier.ppo_barrier import PPOBarrierTrainer
from srlnbc.env.config import point_goal_config
from srlnbc.env.register import register_safety_gym
from srlnbc.env.simple_safety_gym import SimpleEngine
from srlnbc.utils.path import PROJECT_ROOT


register_safety_gym()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()

    def value(obs):
        with open(os.path.dirname(os.path.dirname(args.path)) + '/params.pkl', 'rb') as f:
            config = pickle.load(f)
        config.update({
            'num_workers': 0,
            'num_envs_per_worker': 1
        })
        trainer = PPOBarrierTrainer(config=config)
        trainer.restore(args.path)
        import tensorflow as tf
        obs_tensor = tf.convert_to_tensor(obs.reshape((-1, obs.shape[-1])))
        _, _, extra_out = trainer.get_policy().model({'obs': obs_tensor})

        from srlnbc.agents.safety.mixin import Postprocessing

        value = extra_out[Postprocessing.CERTIFICATE].numpy().reshape(obs.shape[:2])

        return value

    env = SimpleEngine(point_goal_config)
    env.plot_map(
        func=value,
        data_path=os.path.join(PROJECT_ROOT, 'data/point_goal_right.npz'),
        save_path=os.path.join(PROJECT_ROOT, 'results/figures'),
        name='point_goal_right.pdf',
    )

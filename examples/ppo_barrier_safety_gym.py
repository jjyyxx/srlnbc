import ray.tune
import ray.tune.utils.log
from ray.rllib.agents.callbacks import MultiCallbacks

from srlnbc.agents.callbacks import SafetyCallbacks
from srlnbc.agents.safety.ppo_barrier.ppo_barrier import PPOBarrierTrainer
from srlnbc.env.register import register_safety_gym
from srlnbc.utils.tune import create_progress, create_callbacks

register_safety_gym()

DEBUG = False

stop_criteria = {
    'timesteps_total': int(1e7),
}

agent_config = {
    # Env config
    'env': 'point_goal',
    'env_config': {
        'ttc_upper_threshold': 2.0,  # ablation: (2, 1), (5, 0.5), (inf, 0)
        'ttc_lower_threshold': 1.0,
    },

    # Worker config
    'framework': 'tf2',
    'eager_tracing': False if DEBUG else True,
    'num_workers': 0 if DEBUG else 5,
    'num_envs_per_worker': 1 if DEBUG else 4,
    'num_gpus': 0,

    # PPO config
    'rollout_fragment_length': 1000,
    'train_batch_size': 2000 if DEBUG else 20000,
    'sgd_minibatch_size': 0,
    'num_sgd_iter': 30,
    'lr': 3e-4,
    'lambda': 0.97,
    'gamma': 0.99,
    'no_done_at_end': True,

    # Safety config
    'penalty_lr': 1e-2,
    'penalty_init': 1.0,
    'max_penalty': 2.0,
    'penalty_margin_clip': 0.0,
    'barrier_lambda': 0.1,
    'epsilon': 0.01,
    'num_barrier_step': 20,  # ablation: 20, 10, 5, 1

    # Model config
    'model': {
        'custom_model_config': {
            'fcnet_hiddens': [256, 256],
            'fcnet_activation': 'tanh',
            'vf_share_layers': False,
            'free_log_std': True,
            'init_log_std': -0.5,
        }
    },
    'callbacks': MultiCallbacks([
        SafetyCallbacks,
    ]),
    "seed": 1,
}

if DEBUG:
    trainer = PPOBarrierTrainer(config=agent_config)
    while True:
        trainer.train()
else:
    ray.tune.run(
        PPOBarrierTrainer,
        checkpoint_freq=50,
        keep_checkpoints_num=3,
        checkpoint_at_end=True,
        stop=stop_criteria,
        config=agent_config,
        verbose=ray.tune.utils.log.Verbosity.V1_EXPERIMENT,
        progress_reporter=create_progress(),
        callbacks=create_callbacks(),
        log_to_file=True,
        max_concurrent_trials=3
    )

import ray.tune
import ray.tune.utils.log
from ray.rllib.agents.callbacks import MultiCallbacks

from srlnbc.agents.callbacks.metadrive import DrivingCallbacks
from srlnbc.agents.safety.ppo_lagrangian.ppo_lagrangian import PPOLagrangianTrainer
from srlnbc.env.register import register_metadrive
from srlnbc.models.tf.safety_statewise_fcnet import Keras_SafetyStatewiseFullyConnectedNetwork
from srlnbc.utils.tune import create_callbacks, create_progress

register_metadrive()

DEBUG = False

stop_criteria = {
    'timesteps_total': int(1e7),
}

agent_config = {
    # Env config
    'env': 'metadrive',
    'env_config': {
        'crash_vehicle_penalty': 0,
        'crash_object_penalty': 0,
    },

    # Worker config
    'framework': 'tf2',
    'eager_tracing': False if DEBUG else True,
    'num_workers': 0 if DEBUG else 10,
    'num_envs_per_worker': 1,
    'num_gpus': 0,

    # PPO config
    'horizon': 1000,
    'rollout_fragment_length': 1000,
    'train_batch_size': 2000 if DEBUG else 20000,
    'sgd_minibatch_size': 0,
    'num_sgd_iter': 30,
    'lr': 3e-4,
    'lambda': 0.95,
    'gamma': 0.99,
    'no_done_at_end': False,

    # Safety config
    'cost_threshold': 0.0,
    'penalty_lr': 1e-2,
    'penalty_statewise': False,
    'penalty_init': 1.0,

    # Model config
    'model': {
        'custom_model': Keras_SafetyStatewiseFullyConnectedNetwork,
        'custom_model_config': {
            'fcnet_hiddens': [256, 256],
            'fcnet_activation': 'tanh',
            'vf_share_layers': False,
            'free_log_std': True,
            'init_log_std': -0.0,
        }
    },
    'callbacks': MultiCallbacks([
        DrivingCallbacks
    ]),
    "seed": 1,
}

if DEBUG:
    trainer = PPOLagrangianTrainer(config=agent_config)
    while True:
        trainer.train()
else:
    ray.tune.run(
        PPOLagrangianTrainer,
        checkpoint_freq=50,
        keep_checkpoints_num=3,
        checkpoint_at_end=True,
        stop=stop_criteria,
        config=agent_config,
        verbose=ray.tune.utils.log.Verbosity.V1_EXPERIMENT,
        progress_reporter=create_progress(),
        callbacks=create_callbacks(),
        log_to_file=True,
        max_concurrent_trials=2
    )

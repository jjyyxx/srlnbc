from typing import Type

from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_DEFAULT_CONFIG
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict

from srlnbc.agents.safety.ppo_lagrangian.ppo_lagrangian import PPOLagrangianTrainer
from srlnbc.models.tf.safety_certificate_fcnet import Keras_SafetyCertificateFullyConnectedNetwork

PPO_BARRIER_SPECIFIC_CONFIG = {
    "model": {
        "custom_model": Keras_SafetyCertificateFullyConnectedNetwork,
    },
    "penalty_lr": 5e-2,
    "penalty_init": 1.0,
    "max_penalty": 2.0,
    "penalty_margin_clip": 0.0,
    "use_gae_returns": True,
    "simple_optimizer": True,
    "barrier_lambda": 0.1,
    "epsilon": 0.01,
    "num_barrier_step": 10,
    "_disable_execution_plan_api": False
}


class PPOBarrierTrainer(PPOLagrangianTrainer):
    """
    This is a custom PPOTrainer that uses the Lagrangian method to solve the
    safety problem.
    """

    @classmethod
    @override(PPOTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return cls.merge_trainer_configs(PPO_DEFAULT_CONFIG, PPO_BARRIER_SPECIFIC_CONFIG,
                                         _allow_unknown_configs=True)

    @override(PPOTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        from srlnbc.agents.safety.ppo_barrier.ppo_barrier_tf_policy import PPOBarrierTFPolicy
        return PPOBarrierTFPolicy

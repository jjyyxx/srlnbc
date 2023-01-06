from typing import Type

from ray.rllib.evaluation.postprocessing import Postprocessing as ReturnPostprocessing
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, \
    StandardizeFields, SelectExperiences
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator
from ray.rllib.agents.ppo.ppo import PPOTrainer, UpdateKL, warn_about_bad_reward_scales, DEFAULT_CONFIG as PPO_DEFAULT_CONFIG

from srlnbc.agents.safety.execution_ops import CentralizeFields, UpdatePenalty
from srlnbc.agents.safety.mixin import Postprocessing as CostPostprocessing


PPO_LAGRANGIAN_SPECIFIC_CONFIG = {
    "cost_threshold": 0.0,
    "penalty_lr": 5e-2,
    "penalty_init": 1.0,
    "penalty_statewise": False,
    "use_gae_returns": True,
    "simple_optimizer": True,
    "_disable_execution_plan_api": False
}


class PPOLagrangianTrainer(PPOTrainer):
    """
    This is a custom PPOTrainer that uses the Lagrangian method to solve the
    safety problem.
    """
    @classmethod
    @override(PPOTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return cls.merge_trainer_configs(PPO_DEFAULT_CONFIG, PPO_LAGRANGIAN_SPECIFIC_CONFIG, _allow_unknown_configs=True)

    @override(PPOTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        from srlnbc.agents.safety.ppo_lagrangian.ppo_lagrangian_tf_policy import PPOLagrangianTFPolicy
        return PPOLagrangianTFPolicy

    @staticmethod
    @override(PPOTrainer)
    def execution_plan(workers: WorkerSet, config: TrainerConfigDict,
                       **kwargs) -> LocalIterator[dict]:
        assert len(kwargs) == 0, (
            "PPO execution_plan does NOT take any additional parameters")

        rollouts = (
            ParallelRollouts(workers, mode="bulk_sync")
                # Collect batches for the trainable policies.
                .for_each(SelectExperiences(workers.trainable_policies()))
                # Concatenate the SampleBatches into one.
                .combine(ConcatBatches(
                    min_batch_size=config["train_batch_size"],
                    count_steps_by=config["multiagent"]["count_steps_by"],
                ))
                # Standardize advantages.
                .for_each(StandardizeFields([
                    ReturnPostprocessing.ADVANTAGES,
                ]))
                # Centralize cost advantages.
                .for_each(CentralizeFields([
                    CostPostprocessing.ADVANTAGES,
                ]))
        )

        train_op = (
            rollouts
                .for_each(UpdatePenalty(workers, config.get("penalty_statewise", False)))
                .for_each(TrainOneStep(
                    workers,
                    num_sgd_iter=config["num_sgd_iter"],
                    sgd_minibatch_size=config["sgd_minibatch_size"]
                ))
                .for_each(lambda t: t[1])
                # Update KL after each round of training.
                .for_each(UpdateKL(workers))
        )

        # Warn about bad reward scales and return training metrics.
        return StandardMetricsReporting(train_op, workers, config) \
            .for_each(lambda result: warn_about_bad_reward_scales(
                config, result))

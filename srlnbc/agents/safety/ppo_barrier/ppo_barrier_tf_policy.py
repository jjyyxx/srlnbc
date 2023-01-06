import functools
from typing import Dict, List, Type, Union

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, kl_and_loss_stats, LearningRateSchedule, \
    EntropyCoeffSchedule, KLCoeffMixin
from ray.rllib.evaluation.postprocessing import Postprocessing as ReturnPostprocessing
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_variable
from ray.rllib.utils.typing import TensorType

from srlnbc.agents.safety.mixin import compute_gae_for_sample_batch, RETURN_GAE_PACK, \
    Postprocessing as CostPostprocessing, gather_from_info


class PenaltyMixin:
    def __init__(self, config):
        self.penalty_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["penalty_lr"],
            name="penalty_optimizer")
        param_init = PenaltyMixin._compute_param_init(config["penalty_init"])
        self.penalty_param: tf.Variable = get_variable(
            param_init,
            tf_name="penalty_param",
            trainable=True,
            framework=config["framework"])

        @tf.function
        def update_penalty(penalty_param, optimizer, train_batch):
            if self.penalty <= config["max_penalty"]:
                penalty_margin = train_batch[CostPostprocessing.PENALTY_MARGIN]
                with tf.GradientTape() as tape:
                    penalty_loss = -penalty_param * tf.reduce_mean(penalty_margin)
                gradients = tape.gradient(penalty_loss, [penalty_param])
                optimizer.apply_gradients(zip(gradients, [penalty_param]))

        self.update_penalty = functools.partial(update_penalty, self.penalty_param, self.penalty_optimizer)

    @staticmethod
    def _compute_param_init(penalty_init: float):
        penalty_init = float(penalty_init)
        return float(np.log(max(np.exp(penalty_init) - 1, 1e-8)))

    @property
    def penalty(self):
        return tf.math.softplus(self.penalty_param)

    @override(Policy)
    def get_state(self) -> Union[Dict[str, TensorType], List[TensorType]]:
        state = super().get_state()
        state["penalty"] = {
            "param": float(self.penalty_param),
            "optimizer": self.penalty_optimizer.get_weights()
        }
        return state

    @override(Policy)
    def set_state(self, state: dict) -> None:
        if "penalty" in state:
            penalty_state = state.pop("penalty")
            # Initialize optimizer
            self.update_penalty({
                CostPostprocessing.PENALTY_MARGIN: tf.zeros(1)
            })
            self.penalty_param.assign(penalty_state["param"])
            self.penalty_optimizer.set_weights(penalty_state["optimizer"])
        else:
            print("No penalty state to restore!")
        super().set_state(state)

def validate_keras(policy: "PPOBarrierTFPolicy", obs_space, action_space, config):
    assert config["framework"] == "tf2"
    assert isinstance(policy.model, tf.keras.Model)
    policy.view_requirements[SampleBatch.INFOS].used_for_training = False


def stats(policy: "PPOBarrierTFPolicy", train_batch: SampleBatch):
    return {
        **kl_and_loss_stats(policy, train_batch),
        "certificate_loss": policy._mean_certificate_loss,
        "penalty": policy._mean_penalty,
    }


def postprocess(policy: "PPOBarrierTFPolicy", sample_batch: SampleBatch, other_agent_batches=None, episode=None):
    gather_from_info(
        sample_batch,
        [CostPostprocessing.COST,
         CostPostprocessing.FEASIBLE,
         CostPostprocessing.INFEASIBLE],
        no_tracing=policy._no_tracing
    )

    sample_batch = compute_gae_for_sample_batch(policy, sample_batch, policy._value, RETURN_GAE_PACK)
    sample_batch[CostPostprocessing.ADVANTAGES] = np.zeros_like(sample_batch[CostPostprocessing.COST])
    sample_batch[CostPostprocessing.VALUE_TARGETS] = np.zeros_like(sample_batch[CostPostprocessing.COST])

    certificate = sample_batch[CostPostprocessing.CERTIFICATE]
    penalty_margin = np.zeros_like(certificate)
    for i in range(policy.config["num_barrier_step"]):
        pm = policy.config["epsilon"] * (1 - (1 - policy.config["barrier_lambda"]) ** (i + 1)) / \
             policy.config["barrier_lambda"] + certificate[i + 1:] - \
             (1 - policy.config["barrier_lambda"]) ** (i + 1) * certificate[:-i - 1]
        pm = np.clip(pm, policy.config["penalty_margin_clip"], None)
        penalty_margin[:-i - 1] = penalty_margin[:-i - 1] + policy.config["gamma"] ** i * pm
    penalty_margin = penalty_margin * (1 - policy.config["gamma"]) / \
                     (1 - policy.config["gamma"] ** policy.config["num_barrier_step"])
    sample_batch[CostPostprocessing.PENALTY_MARGIN] = penalty_margin.astype(np.float32)

    # if episode is not None:
    #     episode.user_data["penalty_margin"] = penalty_margin

    costs = sample_batch[CostPostprocessing.COST]
    episode_cost = costs.sum()
    episode_cost_array = np.zeros_like(costs)
    episode_cost_array[-1] = episode_cost
    episode_mask = np.zeros_like(costs)
    episode_mask[-1] = 1.0

    sample_batch[CostPostprocessing.EPISODE_COSTS] = episode_cost_array
    sample_batch[CostPostprocessing.EPISODE_MASK] = episode_mask
    return sample_batch


def ppo_lagrangian_surrogate_loss(
        policy: "PPOBarrierTFPolicy", model: tf.keras.Model,
        dist_class: Type[TFActionDistribution],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    logits, state, extra_outs = model(train_batch)

    curr_action_dist = dist_class(logits, model)
    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

    logp_ratio = tf.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
        train_batch[SampleBatch.ACTION_LOGP])

    # Only calculate kl loss if necessary (kl-coeff > 0.0).
    if policy.config["kl_coeff"] > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)
        mean_kl_loss = tf.reduce_mean(action_kl)
    else:
        mean_kl_loss = 0.0

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = tf.reduce_mean(curr_entropy)

    advantage = train_batch[ReturnPostprocessing.ADVANTAGES]
    penalty_margin = train_batch[CostPostprocessing.PENALTY_MARGIN]
    penalty_margin = (penalty_margin - tf.reduce_mean(penalty_margin)) / \
                     tf.maximum(1e-4, tf.math.reduce_std(penalty_margin))

    surrogate_loss = tf.minimum(
        advantage * logp_ratio,
        advantage * tf.clip_by_value(
            logp_ratio, 1 - policy.config["clip_param"],
                        1 + policy.config["clip_param"]))
    mean_surrogate_loss = tf.reduce_mean(-surrogate_loss)

    mean_surrogate_cost_loss = tf.reduce_mean(-logp_ratio * penalty_margin)
    mean_penalty = penalty = tf.stop_gradient(policy.penalty)
    mean_policy_loss = (mean_surrogate_loss - penalty * mean_surrogate_cost_loss) / (1 + penalty)

    prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
    value_fn_out = extra_outs[SampleBatch.VF_PREDS]
    vf_loss1 = tf.math.square(value_fn_out -
                              train_batch[ReturnPostprocessing.VALUE_TARGETS])
    vf_clipped = prev_value_fn_out + tf.clip_by_value(
        value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
        policy.config["vf_clip_param"])
    vf_loss2 = tf.math.square(vf_clipped -
                              train_batch[ReturnPostprocessing.VALUE_TARGETS])
    vf_loss = tf.maximum(vf_loss1, vf_loss2)
    mean_vf_loss = tf.reduce_mean(vf_loss)

    certificate = extra_outs[CostPostprocessing.CERTIFICATE]
    feasible = train_batch[CostPostprocessing.FEASIBLE]
    infeasible = train_batch[CostPostprocessing.INFEASIBLE]

    feasible_loss = feasible * tf.maximum(policy.config["epsilon"] + certificate, 0)
    feasible_loss = tf.reduce_sum(feasible_loss) / tf.maximum(tf.reduce_sum(feasible), 1)

    infeasible_loss = infeasible * tf.maximum(policy.config["epsilon"] - certificate, 0)
    infeasible_loss = tf.reduce_sum(infeasible_loss) / tf.maximum(tf.reduce_sum(infeasible), 1)

    invariance_loss = 0
    mask = tf.cast(train_batch[CostPostprocessing.EPISODE_MASK][:-1], tf.bool)
    for i in range(policy.config["num_barrier_step"]):
        inv_loss = tf.maximum(
            policy.config["epsilon"] * (1 - (1 - policy.config["barrier_lambda"]) ** (i + 1)) /
            policy.config["barrier_lambda"] + certificate[i + 1:] -
            (1 - policy.config["barrier_lambda"]) ** (i + 1) * certificate[:-i - 1], 0)
        inv_loss = (1 - tf.cast(mask, tf.float32)) * inv_loss
        mask = mask[:-1] | mask[1:]
        inv_loss = tf.reduce_mean(inv_loss)
        invariance_loss = invariance_loss + policy.config["gamma"] ** i * inv_loss
    invariance_loss = invariance_loss * (1 - policy.config["gamma"]) / \
                      (1 - policy.config["gamma"] ** policy.config["num_barrier_step"])

    mean_certificate_loss = feasible_loss + infeasible_loss + invariance_loss

    # NOTE: Will sum of mean lead to numerical issues?
    total_loss = mean_policy_loss + \
                 policy.config["vf_loss_coeff"] * mean_vf_loss + \
                 policy.config["vf_loss_coeff"] * mean_certificate_loss - \
                 policy.entropy_coeff * mean_entropy

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store stats in policy for stats_fn.
    policy._total_loss = total_loss
    policy._mean_policy_loss = mean_policy_loss
    policy._mean_vf_loss = mean_vf_loss
    policy._mean_entropy = mean_entropy
    policy._mean_kl_loss = mean_kl_loss
    policy._mean_certificate_loss = mean_certificate_loss
    policy._value_fn_out = value_fn_out
    policy._mean_penalty = mean_penalty

    return total_loss


def extra_scalar_inference(model: tf.keras.Model, key: str, **input_dict):
    _, _, extra_outs = model(input_dict)
    return extra_outs[key][0]


def setup_mixins(policy: "PPOBarrierTFPolicy", obs_space, action_space, config) -> None:
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    PenaltyMixin.__init__(policy, config)

    policy._value = tf.function(functools.partial(extra_scalar_inference, policy.model, SampleBatch.VF_PREDS))


PPOBarrierTFPolicy: Type[DynamicTFPolicy] = PPOTFPolicy.with_updates(
    name="PPOBarrierTFPolicy",
    loss_fn=ppo_lagrangian_surrogate_loss,
    get_default_config=None,
    postprocess_fn=postprocess,
    stats_fn=stats,
    extra_action_out_fn=None,
    after_init=validate_keras,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        PenaltyMixin
    ]
)

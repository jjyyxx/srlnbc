import functools
from typing import List, Type, Union
import numpy as np

import tensorflow as tf

from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, kl_and_loss_stats, LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin
from ray.rllib.evaluation.postprocessing import Postprocessing as ReturnPostprocessing
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.utils.framework import get_variable
from ray.rllib.utils.typing import TensorType

from srlnbc.agents.safety.mixin import compute_gae_for_sample_batch, gather_cost_from_info, RETURN_GAE_PACK, \
    COST_GAE_PACK, Postprocessing as CostPostprocessing

class PenaltyMixin:
    """
    NOTE: Current implementatino of statewise penalty is quite inconsistent with the simple penalty method.
    Seeking a more unified implementation.
    """
    def __init__(self, config):
        self.cost_threshold = float(config["cost_threshold"])
        self.penalty_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["penalty_lr"],
            name="penalty_optimizer")
        self.penalty_statewise = config["penalty_statewise"]
        if self.penalty_statewise:
            @tf.function
            def update_penalty(penalty_params, cost_threshold, optimizer, train_batch):                
                penalty_margin = train_batch[CostPostprocessing.VALUE_TARGETS] - cost_threshold
                with tf.GradientTape() as tape:
                    _, _, extra_outs = self.model(train_batch)
                    penalty_loss = -tf.reduce_mean(extra_outs[CostPostprocessing.MULTIPLIER] * penalty_margin)
                gradients = tape.gradient(penalty_loss, penalty_params)
                optimizer.apply_gradients(zip(gradients, penalty_params))
            penalty_params = [var for var in self.model.trainable_variables if 'multiplier' in var.name]
            self.update_penalty = functools.partial(update_penalty, penalty_params, self.cost_threshold, self.penalty_optimizer)
        else:
            penalty_init = float(config["penalty_init"])
            param_init = float(np.log(max(np.exp(penalty_init) - 1, 1e-8)))
            self.penalty_param = get_variable(
                param_init,
                tf_name="penalty_param",
                trainable=True,
                framework=config["framework"])
            @tf.function
            def update_penalty(penalty_param, cost_threshold, optimizer, train_batch):
                mean_episode_cost = tf.reduce_sum(train_batch[CostPostprocessing.EPISODE_COSTS]) /\
                                    tf.reduce_sum(train_batch[CostPostprocessing.EPISODE_MASK])
                penalty_margin = mean_episode_cost - cost_threshold
                with tf.GradientTape() as tape:
                    penalty_loss = -penalty_param * penalty_margin
                gradients = tape.gradient(penalty_loss, [penalty_param])
                optimizer.apply_gradients(zip(gradients, [penalty_param]))
            self.update_penalty = functools.partial(update_penalty, self.penalty_param, self.cost_threshold, self.penalty_optimizer)

    @property
    def penalty(self):
        assert not self.penalty_statewise
        return tf.math.softplus(self.penalty_param)

def validate_keras(policy: "PPOLagrangianTFPolicy", obs_space, action_space, config):
    assert config["framework"] == "tf2"
    assert isinstance(policy.model, tf.keras.Model)
    policy.view_requirements[SampleBatch.INFOS].used_for_training = False

def stats(policy: "PPOLagrangianTFPolicy", train_batch: SampleBatch):
    return {
        **kl_and_loss_stats(policy, train_batch),
        "cost_vf_loss": policy._mean_cost_vf_loss,
        "penalty": policy._mean_penalty,
    }

def postprocess(policy: "PPOLagrangianTFPolicy", sample_batch: SampleBatch, other_agent_batches = None, episode = None):
    gather_cost_from_info(sample_batch, no_tracing=policy._no_tracing)
    
    sample_batch = compute_gae_for_sample_batch(policy, sample_batch, policy._value, RETURN_GAE_PACK)
    sample_batch = compute_gae_for_sample_batch(policy, sample_batch, policy._cost_value, COST_GAE_PACK)
    
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
    policy: "PPOLagrangianTFPolicy", model: tf.keras.Model,
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
    cost_advantage = train_batch[CostPostprocessing.ADVANTAGES]

    surrogate_loss = tf.minimum(
        advantage * logp_ratio,
        advantage * tf.clip_by_value(
            logp_ratio, 1 - policy.config["clip_param"],
            1 + policy.config["clip_param"]))
    mean_surrogate_loss = tf.reduce_mean(-surrogate_loss)

    if policy.config["penalty_statewise"]:
        multiplier = train_batch[CostPostprocessing.MULTIPLIER]
        mean_penalty = mean_multiplier = tf.reduce_mean(multiplier)
        mean_surrogate_cost_loss = tf.reduce_mean(-multiplier * logp_ratio * cost_advantage)
        mean_policy_loss = (mean_surrogate_loss - mean_surrogate_cost_loss) / (1 + mean_multiplier)
    else:
        mean_surrogate_cost_loss = tf.reduce_mean(-logp_ratio * cost_advantage)
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

    cost_fn_out = extra_outs[CostPostprocessing.CF_PREDS]
    cost_vf_loss = tf.pow(cost_fn_out - train_batch[CostPostprocessing.VALUE_TARGETS], 2.0)
    mean_cost_vf_loss = tf.reduce_mean(cost_vf_loss)

    # NOTE: Will sum of mean lead to numerical issues?
    total_loss = mean_policy_loss + \
                 policy.config["vf_loss_coeff"] * mean_vf_loss + \
                 policy.config["vf_loss_coeff"] * mean_cost_vf_loss - \
                 policy.entropy_coeff * mean_entropy

    if policy.config["kl_coeff"] > 0.0:
        total_loss += policy.kl_coeff * mean_kl_loss

    # Store stats in policy for stats_fn.
    policy._total_loss = total_loss
    policy._mean_policy_loss = mean_policy_loss
    policy._mean_vf_loss = mean_vf_loss
    policy._mean_entropy = mean_entropy
    policy._mean_kl_loss = mean_kl_loss
    policy._mean_cost_vf_loss = mean_cost_vf_loss
    policy._value_fn_out = value_fn_out
    policy._mean_penalty = mean_penalty

    return total_loss

@tf.function
def extra_scalar_inference(model: tf.keras.Model, key: str, **input_dict):
    _, _, extra_outs = model(input_dict)
    return extra_outs[key][0]

def setup_mixins(policy: "PPOLagrangianTFPolicy", obs_space, action_space, config) -> None:
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    PenaltyMixin.__init__(policy, config)

    policy._value = functools.partial(extra_scalar_inference, policy.model, SampleBatch.VF_PREDS)
    policy._cost_value = functools.partial(extra_scalar_inference, policy.model, CostPostprocessing.CF_PREDS)

PPOLagrangianTFPolicy: Type[DynamicTFPolicy] = PPOTFPolicy.with_updates(
    name="PPOLagrangianTFPolicy",
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

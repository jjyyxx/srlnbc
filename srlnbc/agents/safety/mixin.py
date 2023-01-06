import operator
from typing import Callable, Dict, List, NamedTuple

import numpy as np
from ray.rllib.evaluation.postprocessing import discount_cumsum, Postprocessing as ReturnPostprocessing
from ray.rllib.models import ActionDistribution
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import TensorType

from srlnbc.models.safety_model import SafetyModel


class GAEKeyPack(NamedTuple):
    # Inputs
    rewards: str
    vf_preds: str
    # Outputs
    advantages: str
    value_targets: str


class Postprocessing:
    """Constant definitions for safety RL."""

    COST = "cost"
    CF_PREDS = "cf_preds"
    ADVANTAGES = "cost_advantages"
    VALUE_TARGETS = "cost_targets"
    EPISODE_COSTS = "episode_cost"
    EPISODE_MASK = "episode_mask"
    MULTIPLIER = "multiplier"
    CERTIFICATE = "certificate"
    PENALTY_MARGIN = "penalty_margin"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"


RETURN_GAE_PACK = GAEKeyPack(SampleBatch.REWARDS, SampleBatch.VF_PREDS,
                             ReturnPostprocessing.ADVANTAGES, ReturnPostprocessing.VALUE_TARGETS)

COST_GAE_PACK = GAEKeyPack(Postprocessing.COST, Postprocessing.CF_PREDS,
                           Postprocessing.ADVANTAGES, Postprocessing.VALUE_TARGETS)


def extra_action_out(policy: Policy, input_dict: Dict[str, TensorType], state_batches: List[TensorType], model: SafetyModel, action_dist: ActionDistribution):
    return {
        Postprocessing.CF_PREDS: model.cost_function(),
    }

def gather_cost_from_info(input_dict: Dict[str, TensorType], *, no_tracing: bool = False, cost_key: str = "cost") -> np.ndarray:
    if no_tracing:
        # Workaround for dummy batch, where info is a float.
        costs = input_dict[SampleBatch.INFOS]
    else:
        infos = input_dict[SampleBatch.INFOS].tolist()
        getter = operator.itemgetter(cost_key)
        costs = np.fromiter(map(getter, infos), dtype=np.float32, count=len(infos))
    input_dict[Postprocessing.COST] = costs
    return costs

def gather_from_info(input_dict: Dict[str, TensorType], keys: List[str], no_tracing: bool = False):
    if no_tracing:
        value = input_dict[SampleBatch.INFOS]
        for key in keys:
            input_dict[key] = value
    else:
        infos = input_dict[SampleBatch.INFOS].tolist()
        for key in keys:
            getter = operator.itemgetter(key)
            value = np.fromiter(map(getter, infos), dtype=np.float32, count=len(infos))
            input_dict[key] = value

@DeveloperAPI
def compute_advantages(
        rollout: SampleBatch,
        last_r: float,
        gamma: float = 0.9,
        lambda_: float = 1.0,
        use_gae: bool = True,
        use_gae_returns: bool = True,
        use_critic: bool = True,
        key_pack: GAEKeyPack = RETURN_GAE_PACK,
    ):
    K_REW, K_VFP, K_ADV, K_VTG = key_pack
    assert K_VFP in rollout or not use_critic, "use_critic=True but values not found"
    assert use_critic or not use_gae,  "Can't use gae without using a value function"
    if use_gae:
        vpred_t = np.concatenate([rollout[K_VFP], np.array([last_r])])
        delta_t = rollout[K_REW] + gamma * vpred_t[1:] - vpred_t[:-1]
        rollout[K_ADV] = discount_cumsum(delta_t, gamma * lambda_)
        if use_gae_returns:
            rollout[K_VTG] = (rollout[K_ADV] + rollout[K_VFP]).astype(np.float32)
        else:
            rew_t = np.concatenate([rollout[K_REW], np.array([last_r])])
            rollout[K_VTG] = discount_cumsum(rew_t, gamma)[:-1].astype(np.float32)
    else:
        rewards_plus_v = np.concatenate([rollout[K_REW], np.array([last_r])])
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(np.float32)
        if use_critic:
            rollout[K_ADV] = discounted_returns - rollout[K_VFP]
            rollout[K_VTG] = discounted_returns
        else:
            rollout[K_ADV] = discounted_returns
            rollout[K_VTG] = np.zeros_like(rollout[K_ADV])
    rollout[K_ADV] = rollout[K_ADV].astype(np.float32)
    return rollout

def compute_gae_for_sample_batch(
        policy: Policy,
        sample_batch: SampleBatch,
        value_func: Callable[[SampleBatch], float],
        key_pack: GAEKeyPack,
        ) -> SampleBatch:
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    else:
        input_dict = sample_batch.get_single_step_input_dict(policy.model.view_requirements, index="last")
        last_r = value_func(**input_dict)
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_gae_returns=policy.config["use_gae_returns"],
        use_critic=policy.config.get("use_critic", True),
        key_pack=key_pack)
    return batch

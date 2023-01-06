from typing import List
import numpy as np
from ray.rllib.execution.common import _check_sample_batch_type
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.typing import SampleBatchType

from srlnbc.agents.safety.mixin import Postprocessing as CostPostprocessing


def centralize(array: np.ndarray):
    return array - array.mean()


class CentralizeFields:
    """Callable used to centralize fields of batches but NOT standardize (i.e. not dividing std)."""

    def __init__(self, fields: List[str]):
        self.fields = fields

    def __call__(self, samples: SampleBatchType) -> SampleBatchType:
        _check_sample_batch_type(samples)
        wrapped = False

        if isinstance(samples, SampleBatch):
            samples = samples.as_multi_agent()
            wrapped = True

        for policy_id in samples.policy_batches:
            batch = samples.policy_batches[policy_id]
            for field in self.fields:
                batch[field] = centralize(batch[field])

        if wrapped:
            samples = samples.policy_batches[DEFAULT_POLICY_ID]

        return samples


class UpdatePenalty:
    def __init__(self, workers: WorkerSet, statewise: bool):
        self.workers = workers
        self.statewise = statewise

    def __call__(self, samples: SampleBatchType) -> SampleBatchType:
        def select_and_pop_penalty(policy, batch: SampleBatch) -> SampleBatch:
            # costs = batch.pop(CostPostprocessing.EPISODE_COSTS)
            # mask = batch.pop(CostPostprocessing.EPISODE_MASK)
            # # NOTE: Is a separate batch necessary?
            # if self.statewise:
            #     penalty_batch = SampleBatch({
            #         SampleBatch.OBS: batch[SampleBatch.OBS],
            #         CostPostprocessing.VALUE_TARGETS: batch[CostPostprocessing.VALUE_TARGETS],
            #     })
            # else:
            #     penalty_batch = SampleBatch({
            #         CostPostprocessing.EPISODE_COSTS: costs,
            #         CostPostprocessing.EPISODE_MASK: mask,
            #     })
            penalty_batch = batch
            policy._lazy_tensor_dict(penalty_batch)
            penalty_batch.set_training()
            return penalty_batch
        if isinstance(samples, MultiAgentBatch):
            self.workers.local_worker().foreach_trainable_policy(
                lambda p, pid: p.update_penalty(select_and_pop_penalty(p, samples.policy_batches[pid]))
            )
        else:
            self.workers.local_worker().foreach_trainable_policy(
                lambda p, pid: p.update_penalty(select_and_pop_penalty(p, samples))
            )
        return samples

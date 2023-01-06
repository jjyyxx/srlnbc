from typing import Dict, List, Optional, Sequence, Tuple

import gym.spaces
import numpy as np
import tensorflow as tf

from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType

from srlnbc.agents.safety.mixin import Postprocessing as CostPostprocessing
from srlnbc.models.tf.misc import create_fc_policy_branch, create_fc_value_branch


class Keras_SafetyFullyConnectedNetwork(tf.keras.Model):
    """Generic fully connected network implemented in tf Keras."""

    def __init__(
            self,
            input_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: Optional[int] = None,
            *,
            name: str = "",
            fcnet_hiddens: Optional[Sequence[int]] = (),
            fcnet_activation: Optional[str] = None,
            post_fcnet_hiddens: Optional[Sequence[int]] = (),
            post_fcnet_activation: Optional[str] = None,
            no_final_linear: bool = False,
            vf_share_layers: bool = False,
            free_log_std: bool = False,
            init_log_std: Optional[float] = None,
            **kwargs,
    ):
        super().__init__(name=name)

        hiddens = list(fcnet_hiddens or ()) + \
            list(post_fcnet_hiddens or ())
        activation = fcnet_activation
        if not fcnet_hiddens:
            activation = post_fcnet_activation
        activation = get_activation_fn(activation)

        if free_log_std:
            assert num_outputs % 2 == 0 and init_log_std is not None
            num_outputs = num_outputs // 2

        # We are using obs_flat, so take the flattened shape as input.
        inputs = tf.keras.layers.Input(
            shape=(int(np.product(input_space.shape)), ), name="observations")

        logits_out, last_layer, self.num_outputs = create_fc_policy_branch(
            hiddens, activation, inputs, no_final_linear, num_outputs, free_log_std, init_log_std)
        policy_out = logits_out if logits_out is not None else last_layer
        value_out = create_fc_value_branch(hiddens, activation, inputs, last_layer, vf_share_layers, "value")
        cost_out = create_fc_value_branch(hiddens, activation, inputs, last_layer, vf_share_layers, "cost")

        self.base_model = tf.keras.Model(
            inputs, [policy_out, value_out, cost_out])

    def call(self, input_dict: SampleBatch) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        model_out, value_out, cost_out = self.base_model(input_dict[SampleBatch.OBS])
        extra_outs = {
            SampleBatch.VF_PREDS: tf.reshape(value_out, [-1]),
            CostPostprocessing.CF_PREDS: tf.reshape(cost_out, [-1]),
        }
        return model_out, [], extra_outs

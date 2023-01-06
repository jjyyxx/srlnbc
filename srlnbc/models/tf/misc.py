from typing import List

import tensorflow as tf

from ray.rllib.models.tf.misc import normc_initializer

class AppendLogStd(tf.keras.layers.Layer):
    def __init__(self, num_outputs: int, init_log_std: float = -0.5):
        super(AppendLogStd, self).__init__()
        self.log_std = tf.Variable(
            initial_value=tf.constant(init_log_std, dtype=tf.float32, shape=(num_outputs,)),
            trainable=True, name="log_std"
        )

    def call(self, x):
        log_std = tf.broadcast_to(self.log_std, tf.shape(x))
        return tf.concat([x, log_std], axis=-1)

def create_fc_policy_branch(
    hiddens: List[int], activation,
    input_layer,
    no_final_linear: bool, num_outputs: int,
    free_log_std: bool, init_log_std: float):

    # Last hidden layer output (before logits outputs).
    last_layer = input_layer
    input_size = input_layer.shape[1]
    logits_out = None
    i = 1
    for size in hiddens[:-1]:
        last_layer = tf.keras.layers.Dense(
            size,
            name=f"fc_{i}",
            activation=activation,
            kernel_initializer=normc_initializer(1.0))(last_layer)
        i += 1

    if no_final_linear and num_outputs:
        logits_out = tf.keras.layers.Dense(
            num_outputs,
            name="fc_out",
            activation=activation,
            kernel_initializer=normc_initializer(1.0))(last_layer)
    else:
        if len(hiddens) > 0:
            last_layer = tf.keras.layers.Dense(
                hiddens[-1],
                name=f"fc_{i}",
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
        if num_outputs:
            logits_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(last_layer)
        else:
            num_outputs = ([input_size] + hiddens[-1:])[-1]
    if free_log_std and logits_out is not None:
        logits_out = AppendLogStd(num_outputs, init_log_std)(logits_out)
    return logits_out, last_layer, num_outputs

def create_fc_value_branch(
    hiddens: List[int], activation: str,
    input_layer, last_layer,
    vf_share_layers: bool = False, name: str = "value",
    output_activation: str = None):
    last_vf_layer = None
    if not vf_share_layers:
        last_vf_layer = input_layer
        i = 1
        for size in hiddens:
            last_vf_layer = tf.keras.layers.Dense(
                size,
                name=f"fc_{name}_{i}",
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_vf_layer)
            i += 1
    value_out = tf.keras.layers.Dense(
        1,
        name=f"{name}_out",
        activation=output_activation,
        kernel_initializer=normc_initializer(0.01))(
            last_vf_layer if last_vf_layer is not None else last_layer)
    return value_out

def extract_config_from_default(cls: type):
    import inspect
    from ray.rllib.models.catalog import MODEL_DEFAULTS
    sig = inspect.signature(cls.__init__)
    d = {
        k: MODEL_DEFAULTS[k]
        for k, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY and k in MODEL_DEFAULTS
    }
    return d

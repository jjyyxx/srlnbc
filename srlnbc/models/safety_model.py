from abc import ABCMeta, abstractmethod

from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import TensorType

class SafetyModel(metaclass=ABCMeta):
    """Defines an abstract neural network model for use with RLlib.

    Data flow:
        obs -> forward() -> model_out
            \-> value_function() -> V(s)
            \->  cost_function() -> C(s)
    """

    @PublicAPI
    @abstractmethod
    def cost_function(self) -> TensorType:
        """Returns the cost value function output for the most recent forward pass.

        Note that a `forward` call has to be performed first, before this
        methods can return anything and thus that calling this method does not
        cause an extra forward pass through the network.

        Returns:
            Cost value estimate tensor of shape [BATCH].
        """

    @PublicAPI
    @property
    @abstractmethod
    def penalty(self) -> TensorType:
        """
        Lagrangian multiplier.
        """

    @PublicAPI
    @abstractmethod
    def get_penalty_param(self) -> TensorType:
        """
        Lagrangian multiplier parameters.
        """

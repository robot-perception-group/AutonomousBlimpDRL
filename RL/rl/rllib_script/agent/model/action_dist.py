import numpy as np
import gym

from rl.rllib_script.agent.model.misc import ModelV2
from rl.rllib_script.agent.model.tx2_model import TorchModelV2
import torch


from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
TensorType = Any
ModelConfigDict = dict


class ActionDistribution:
    """The policy action distribution of an agent.

    Attributes:
        inputs (Tensors): input vector to compute samples from.
        model (ModelV2): reference to model producing the inputs.
    """

    def __init__(self, inputs: List[TensorType], model: ModelV2):
        """Initializes an ActionDist object.

        Args:
            inputs (Tensors): input vector to compute samples from.
            model (ModelV2): reference to model producing the inputs. This
                is mainly useful if you want to use model variables to compute
                action outputs (i.e., for auto-regressive action distributions,
                see examples/autoregressive_action_dist.py).
        """
        self.inputs = inputs
        self.model = model

    def sample(self) -> TensorType:
        """Draw a sample from the action distribution."""
        raise NotImplementedError

    def deterministic_sample(self) -> TensorType:
        """
        Get the deterministic "sampling" output from the distribution.
        This is usually the max likelihood output, i.e. mean for Normal, argmax
        for Categorical, etc..
        """
        raise NotImplementedError

    def sampled_action_logp(self) -> TensorType:
        """Returns the log probability of the last sampled action."""
        raise NotImplementedError

    def logp(self, x: TensorType) -> TensorType:
        """The log-likelihood of the action distribution."""
        raise NotImplementedError

    def kl(self, other: "ActionDistribution") -> TensorType:
        """The KL-divergence between two action distributions."""
        raise NotImplementedError

    def entropy(self) -> TensorType:
        """The entropy of the action distribution."""
        raise NotImplementedError

    def multi_kl(self, other: "ActionDistribution") -> TensorType:
        """The KL-divergence between two action distributions.

        This differs from kl() in that it can return an array for
        MultiDiscrete. TODO(ekl) consider removing this.
        """
        return self.kl(other)

    def multi_entropy(self) -> TensorType:
        """The entropy of the action distribution.

        This differs from entropy() in that it can return an array for
        MultiDiscrete. TODO(ekl) consider removing this.
        """
        return self.entropy()

    @staticmethod
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        """Returns the required shape of an input parameter tensor for a
        particular action space and an optional dict of distribution-specific
        options.

        Args:
            action_space (gym.Space): The action space this distribution will
                be used for, whose shape attributes will be used to determine
                the required shape of the input parameter tensor.
            model_config (dict): Model's config dict (as defined in catalog.py)

        Returns:
            model_output_shape (int or np.ndarray of ints): size of the
                required input vector (minus leading batch dimension).
        """
        raise NotImplementedError


class TorchDistributionWrapper(ActionDistribution):
    """Wrapper class for torch.distributions."""

    def __init__(self, inputs: List[TensorType], model: TorchModelV2):
        # If inputs are not a torch Tensor, make them one and make sure they
        # are on the correct device.
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)
            if isinstance(model, TorchModelV2):
                inputs = inputs.to(next(model.parameters()).device)
        super().__init__(inputs, model)
        # Store the last sample here.
        self.last_sample = None

    def logp(self, actions: TensorType) -> TensorType:
        return self.dist.log_prob(actions)

    def entropy(self) -> TensorType:
        return self.dist.entropy()

    def kl(self, other: ActionDistribution) -> TensorType:
        return torch.distributions.kl.kl_divergence(self.dist, other.dist)

    def sample(self) -> TensorType:
        self.last_sample = self.dist.sample()
        return self.last_sample

    def sampled_action_logp(self) -> TensorType:
        assert self.last_sample is not None
        return self.logp(self.last_sample)

class TorchDiagGaussian(TorchDistributionWrapper):
    """Wrapper class for PyTorch Normal distribution."""

    def __init__(self, inputs: List[TensorType], model: TorchModelV2):
        super().__init__(inputs, model)
        mean, log_std = torch.chunk(self.inputs, 2, dim=1)
        self.dist = torch.distributions.normal.Normal(mean, torch.exp(log_std))

    def deterministic_sample(self) -> TensorType:
        self.last_sample = self.dist.mean
        return self.last_sample

    def logp(self, actions: TensorType) -> TensorType:
        return super().logp(actions).sum(-1)

    def entropy(self) -> TensorType:
        return super().entropy().sum(-1)

    def kl(self, other: ActionDistribution) -> TensorType:
        return super().kl(other).sum(-1)

    @staticmethod
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape) * 2


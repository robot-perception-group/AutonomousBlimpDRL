import numpy as np
import copy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.torch.misc import (
    SlimFC,
    normc_initializer as torch_normc_initializer,
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class TorchBatchNormModel(TorchModelV2, nn.Module):
    """Example of a TorchModelV2 using batch normalization.
    https://github.com/ray-project/ray/blob/90fd38c64ac282df63c2a7fbccf66a46217991a4/rllib/examples/models/batch_norm_model.py#L155
    """

    capture_index = 0

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in [64, 64]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=torch_normc_initializer(1.0),
                    activation_fn=nn.Tanh,
                )
            )
            prev_layer_size = size
            # Add a batch norm layer.
            layers.append(nn.LayerNorm(prev_layer_size))

        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=self.num_outputs,
            initializer=torch_normc_initializer(0.01),
            activation_fn=None,
        )

        self._hidden_layers = nn.Sequential(*layers)
        self._hidden_out = None

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=torch_normc_initializer(0.01),
            activation_fn=None,
        )

        layers_v = copy.deepcopy(layers)
        self._hidden_layers_v = nn.Sequential(*layers_v)
        self._hidden_out_v = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self._hidden_layers.train(mode=bool(input_dict.get("is_training", False)))

        self._hidden_out = self._hidden_layers(input_dict["obs"])
        self._hidden_out_v = self._hidden_layers_v(input_dict["obs"])

        logits = self._logits(self._hidden_out)
        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out_v is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._hidden_out_v), [-1])

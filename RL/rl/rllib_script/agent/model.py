import numpy as np
from ray.rllib.models.modelv2 import ModelV2
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
        input_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        (self._hidden_layers, self._hidden_out, self._logits) = self._create_bn_layers(
            input_layer_size=input_layer_size,
            out_size=self.num_outputs,
            output_init_weights=1e-12,
        )

        (
            self._hidden_layers_v,
            self._hidden_out_v,
            self._value_branch,
        ) = self._create_bn_layers(
            input_layer_size=input_layer_size,
            out_size=1,
            sizes=[128, 128],
        )

    def _create_bn_layers(
        self,
        input_layer_size,
        out_size,
        sizes=[64, 64],
        output_init_weights=1e-2,
        activation_fn=nn.Tanh,
    ):
        layers = []
        prev_layer_size = input_layer_size
        for size in sizes:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=torch_normc_initializer(1.0),
                    activation_fn=activation_fn,
                )
            )
            prev_layer_size = size
            layers.append(nn.LayerNorm(prev_layer_size))

        _hidden_layers = nn.Sequential(*layers)
        _hidden_out = None
        _branch = SlimFC(
            in_size=prev_layer_size,
            out_size=out_size,
            initializer=torch_normc_initializer(output_init_weights),
            activation_fn=None,
        )
        return _hidden_layers, _hidden_out, _branch

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self._hidden_layers.train(mode=bool(input_dict.get("is_training", False)))
        self._hidden_layers_v.train(mode=bool(input_dict.get("is_training", False)))

        self._hidden_out = self._hidden_layers(input_dict["obs"])
        self._hidden_out_v = self._hidden_layers_v(input_dict["obs"])

        logits = self._logits(self._hidden_out)
        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out_v is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._hidden_out_v), [-1])

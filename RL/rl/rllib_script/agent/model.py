import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as torch_normc_initializer
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
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
            sizes=model_config["custom_model_config"]["actor_sizes"],
        )

        (
            self._hidden_layers_v,
            self._hidden_out_v,
            self._value_branch,
        ) = self._create_bn_layers(
            input_layer_size=input_layer_size,
            out_size=1,
            sizes=model_config["custom_model_config"]["critic_sizes"],
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


class TorchRNNModel(TorchRNN, nn.Module):
    """https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py"""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_size=64,
        lstm_state_size=128,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

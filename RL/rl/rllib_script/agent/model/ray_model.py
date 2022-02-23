from typing import Dict, List, Union

import gym
import numpy as np
import tree  # pip install dm_tree
from gym.spaces import Discrete, MultiDiscrete
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.misc import normc_initializer as torch_normc_initializer
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models import ModelCatalog


torch, nn = try_import_torch()


def _create_bn_layers(
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


class TorchBatchNormModel(TorchModelV2, nn.Module):
    """Example of a TorchModelV2 using batch normalization.

    modified from
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

        actor_sizes = model_config["custom_model_config"].get("actor_sizes", [64, 64])
        critic_sizes = model_config["custom_model_config"].get(
            "critic_sizes", [128, 128]
        )

        (self._hidden_layers, self._hidden_out, self._logits) = _create_bn_layers(
            input_layer_size=input_layer_size,
            out_size=self.num_outputs,
            output_init_weights=1e-12,
            sizes=actor_sizes,
        )
        (
            self._hidden_layers_v,
            self._hidden_out_v,
            self._value_branch,
        ) = _create_bn_layers(
            input_layer_size=input_layer_size,
            out_size=1,
            sizes=critic_sizes,
        )

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


class TorchBatchNormRNNModel(TorchRNN, nn.Module):
    """modified from
    https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
    https://github.com/ray-project/ray/blob/master/rllib/models/torch/recurrent_net.py
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        input_layer_size = int(np.product(obs_space.shape))
        hidden_sizes = model_config["custom_model_config"].get("hidden_sizes", [64, 64])
        self.cell_size = model_config["custom_model_config"].get("lstm_cell_size", 64)

        self.time_major = model_config.get("_time_major", False)
        self.use_prev_action = model_config["custom_model_config"].get(
            "lstm_use_prev_action", True
        )

        self.use_prev_reward = model_config["custom_model_config"].get(
            "lstm_use_prev_reward", True
        )

        self.action_space_struct = get_base_struct_from_space(action_space)
        self.action_dim = 0

        for space in tree.flatten(self.action_space_struct):
            if isinstance(space, Discrete):
                self.action_dim += space.n
            elif isinstance(space, MultiDiscrete):
                self.action_dim += np.sum(space.nvec)
            elif space.shape is not None:
                self.action_dim += int(np.product(space.shape))
            else:
                self.action_dim += int(len(space))

        # Add prev-action/reward nodes to input to LSTM.
        lstm_input_size = hidden_sizes[-1]
        if self.use_prev_action:
            lstm_input_size += self.action_dim
        if self.use_prev_reward:
            lstm_input_size += 1

        self.hidden, _, _ = _create_bn_layers(
            input_layer_size=input_layer_size,
            out_size=hidden_sizes[-1],
            output_init_weights=1e-12,
            sizes=hidden_sizes,
        )
        self.lstm = nn.LSTM(
            lstm_input_size, self.cell_size, batch_first=not self.time_major
        )
        self._logits_branch = SlimFC(
            in_size=self.cell_size,
            out_size=self.num_outputs,
            activation_fn=None,
            initializer=torch_normc_initializer(1e-12),
        )
        self._value_branch = SlimFC(
            in_size=self.cell_size,
            out_size=1,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )

        # __sphinx_doc_begin__
        # Add prev-a/r to this model's view, if required.
        if model_config["custom_model_config"]["lstm_use_prev_action"]:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
                SampleBatch.ACTIONS, space=self.action_space, shift=-1
            )
        if model_config["custom_model_config"]["lstm_use_prev_reward"]:
            self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(
                SampleBatch.REWARDS, shift=-1
            )
        # __sphinx_doc_end__

        self._features = None

    @override(TorchRNN)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        assert seq_lens is not None
        wrapped_out, _ = self.forward_hidden(input_dict, [], None)

        prev_a_r = []
        # Prev actions.
        if self.model_config["custom_model_config"]["lstm_use_prev_action"]:
            try:
                prev_a = input_dict[SampleBatch.PREV_ACTIONS]
            except KeyError:
                print(
                    "[ Warning ] lstm detect keyerror, prev_actions not in the batch key"
                )
                prev_a = torch.zeros(self.action_dim)

            if isinstance(self.action_space, (Discrete, MultiDiscrete)):
                prev_a = one_hot(prev_a.float(), self.action_space)
            else:
                prev_a = prev_a.float()
            prev_a_r.append(torch.reshape(prev_a, [-1, self.action_dim]))

        # Prev rewards.
        if self.model_config["custom_model_config"]["lstm_use_prev_reward"]:
            try:
                prev_r = input_dict[SampleBatch.PREV_REWARDS].float()
            except KeyError:
                print(
                    "[ Warning ] lstm detect keyerror, prev_rewards not in the batch key"
                )
                prev_r = torch.zeros(1)

            prev_a_r.append(torch.reshape(prev_r, [-1, 1]))

        # Concat prev. actions + rewards to the "main" input.
        if prev_a_r:
            wrapped_out = torch.cat([wrapped_out] + prev_a_r, dim=1)

        # Push everything through our LSTM.
        input_dict["obs_flat"] = wrapped_out
        return super().forward(input_dict, state, seq_lens)

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        linear = next(self._logits_branch._model.children())
        h = [
            linear.weight.new(1, self.cell_size).zero_().squeeze(0),
            linear.weight.new(1, self.cell_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self._value_branch(self._features), [-1])

    def forward_hidden(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self.hidden.train(mode=bool(input_dict.get("is_training", False)))

        hidden_out = self.hidden(input_dict["obs"])

        return hidden_out, []

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
        self._features, [h, c] = self.lstm(
            inputs, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        logits = self._logits_branch(self._features)
        return logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


ModelCatalog.register_custom_model("bn_model", TorchBatchNormModel)
ModelCatalog.register_custom_model("bnrnn_model", TorchBatchNormRNNModel)

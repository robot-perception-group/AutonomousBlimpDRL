"""
Because RLlib does not support TX2, this code is basically ray model object with some custom model code.
Most functionalities are not supported. 
"""

from typing import Any, Dict, List, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import tree  # pip install dm_tree
from gym.spaces import Discrete, MultiDiscrete
from rl.rllib_script.agent.model.misc import (ModelV2, SlimFC, ViewRequirement,
                                              get_base_struct_from_space)
from rl.rllib_script.agent.model.misc import \
    normc_initializer as torch_normc_initializer
from rl.rllib_script.agent.model.misc import one_hot
from rl.rllib_script.agent.model.sample_batch import SampleBatch

TensorType = Any
ModelConfigDict = dict



class TorchModelV2(ModelV2):
    """Torch version of ModelV2.

    Note that this class by itself is not a valid model unless you
    inherit from nn.Module and implement forward() in a subclass."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        """Initialize a TorchModelV2.

        Here is an example implementation for a subclass
        ``MyModelClass(TorchModelV2, nn.Module)``::

            def __init__(self, *args, **kwargs):
                TorchModelV2.__init__(self, *args, **kwargs)
                nn.Module.__init__(self)
                self._hidden_layers = nn.Sequential(...)
                self._logits = ...
                self._value_branch = ...
        """

        if not isinstance(self, nn.Module):
            raise ValueError(
                "Subclasses of TorchModelV2 must also inherit from "
                "nn.Module, e.g., MyModel(TorchModelV2, nn.Module)")

        ModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            framework="torch")

    def variables(self, as_dict: bool = False) -> \
            Union[List[TensorType], Dict[str, TensorType]]:
        p = list(self.parameters())
        if as_dict:
            return {k: p[i] for i, k in enumerate(self.state_dict().keys())}
        return p

    def trainable_variables(self, as_dict: bool = False) -> \
            Union[List[TensorType], Dict[str, TensorType]]:
        if as_dict:
            return {
                k: v
                for k, v in self.variables(as_dict=True).items()
                if v.requires_grad
            }
        return [v for v in self.variables() if v.requires_grad]

def add_time_dimension(padded_inputs: TensorType,
                       *,
                       max_seq_len: int,
                       framework: str = "tf",
                       time_major: bool = False):
    """Adds a time dimension to padded inputs.

    Args:
        padded_inputs (TensorType): a padded batch of sequences. That is,
            for seq_lens=[1, 2, 2], then inputs=[A, *, B, B, C, C], where
            A, B, C are sequence elements and * denotes padding.
        max_seq_len (int): The max. sequence length in padded_inputs.
        framework (str): The framework string ("tf2", "tf", "tfe", "torch").
        time_major (bool): Whether data should be returned in time-major (TxB)
            format or not (BxT).

    Returns:
        TensorType: Reshaped tensor of shape [B, T, ...] or [T, B, ...].
    """

    # Sequence lengths have to be specified for LSTM batch inputs. The
    # input batch must be padded to the max seq length given here. That is,
    # batch_size == len(seq_lens) * max(seq_lens)
    if framework in ["tf2", "tf", "tfe"]:
        assert time_major is False, "time-major not supported yet for tf!"
        padded_batch_size = tf.shape(padded_inputs)[0]
        # Dynamically reshape the padded batch to introduce a time dimension.
        new_batch_size = padded_batch_size // max_seq_len
        new_shape = (
            [new_batch_size, max_seq_len] + list(padded_inputs.shape[1:]))
        return tf.reshape(padded_inputs, new_shape)
    else:
        assert framework == "torch", "`framework` must be either tf or torch!"
        padded_batch_size = padded_inputs.shape[0]

        # Dynamically reshape the padded batch to introduce a time dimension.
        new_batch_size = padded_batch_size // max_seq_len
        if time_major:
            new_shape = (max_seq_len, new_batch_size) + padded_inputs.shape[1:]
        else:
            new_shape = (new_batch_size, max_seq_len) + padded_inputs.shape[1:]
        return torch.reshape(padded_inputs, new_shape)

class TorchRNN(TorchModelV2):
    """Helper class to simplify implementing RNN models with TorchModelV2.

    Instead of implementing forward(), you can implement forward_rnn() which
    takes batches with the time dimension added already.

    Here is an example implementation for a subclass
    ``MyRNNClass(RecurrentNetwork, nn.Module)``::

        def __init__(self, obs_space, num_outputs):
            nn.Module.__init__(self)
            super().__init__(obs_space, action_space, num_outputs,
                             model_config, name)
            self.obs_size = _get_size(obs_space)
            self.rnn_hidden_dim = model_config["lstm_cell_size"]
            self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)
            self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
            self.fc2 = nn.Linear(self.rnn_hidden_dim, num_outputs)

            self.value_branch = nn.Linear(self.rnn_hidden_dim, 1)
            self._cur_value = None

        @override(ModelV2)
        def get_initial_state(self):
            # Place hidden states on same device as model.
            h = [self.fc1.weight.new(
                1, self.rnn_hidden_dim).zero_().squeeze(0)]
            return h

        @override(ModelV2)
        def value_function(self):
            assert self._cur_value is not None, "must call forward() first"
            return self._cur_value

        @override(RecurrentNetwork)
        def forward_rnn(self, input_dict, state, seq_lens):
            x = nn.functional.relu(self.fc1(input_dict["obs_flat"].float()))
            h_in = state[0].reshape(-1, self.rnn_hidden_dim)
            h = self.rnn(x, h_in)
            q = self.fc2(h)
            self._cur_value = self.value_branch(h).squeeze(1)
            return q, [h]
    """

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        flat_inputs = input_dict["obs_flat"].float()
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    def forward_rnn(self, inputs: TensorType, state: List[TensorType],
                    seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """Call the model with the given input tensors and state.

        Args:
            inputs (dict): Observation tensor with shape [B, T, obs_size].
            state (list): List of state tensors, each with shape [B, size].
            seq_lens (Tensor): 1D tensor holding input sequence lengths.
                Note: len(seq_lens) == B.

        Returns:
            (outputs, new_state): The model output tensor of shape
                [B, T, num_outputs] and the list of new state tensors each with
                shape [B, size].

        Examples:
            def forward_rnn(self, inputs, state, seq_lens):
                model_out, h, c = self.rnn_model([inputs, seq_lens] + state)
                return model_out, [h, c]
        """
        raise NotImplementedError("You must implement this for an RNN model")



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

    def forward(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self._hidden_layers.train(mode=bool(input_dict.get("is_training", False)))
        self._hidden_layers_v.train(mode=bool(input_dict.get("is_training", False)))

        self._hidden_out = self._hidden_layers(input_dict["obs"])
        self._hidden_out_v = self._hidden_layers_v(input_dict["obs"])

        logits = self._logits(self._hidden_out)
        return logits, []

    
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

        # self.action_space_struct = get_base_struct_from_space(self.action_space)
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

    
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self._value_branch(self._features), [-1])

    def forward_hidden(self, input_dict, state, seq_lens):
        # Set the correct train-mode for our hidden module (only important
        # b/c we have some batch-norm layers).
        self.hidden.train(mode=bool(input_dict.get("is_training", False)))

        hidden_out = self.hidden(input_dict["obs"])

        return hidden_out, []

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


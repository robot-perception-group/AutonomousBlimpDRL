import contextlib
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import tree  # pip install dm_tree
from gym.spaces import Box, Discrete, MultiDiscrete, Space
from rl.rllib_script.agent.model.sample_batch import SampleBatch

TensorType = Any
ModelConfigDict = dict
LocalOptimizer = Union["tf.keras.optimizers.Optimizer",
                       "torch.optim.Optimizer"]
TrainerConfigDict = dict
TensorStructType = Union[TensorType, dict, tuple]
ModelInputDict = Dict[str, TensorType]
OBS_VALIDATION_INTERVAL = 100
MODEL_DEFAULTS: ModelConfigDict = {
    # Experimental flag.
    # If True, try to use a native (tf.keras.Model or torch.Module) default
    # model instead of our built-in ModelV2 defaults.
    # If False (default), use "classic" ModelV2 default models.
    # Note that this currently only works for:
    # 1) framework != torch AND
    # 2) fully connected and CNN default networks as well as
    # auto-wrapped LSTM- and attention nets.
    "_use_default_native_models": False,
    # Experimental flag.
    # If True, user specified no preprocessor to be created
    # (via config.preprocessor_pref=None). If True, observations will arrive
    # in model as they are returned by the env.
    "_no_preprocessing": False,

    # === Built-in options ===
    # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
    # These are used if no custom model is specified and the input space is 1D.
    # Number of hidden layers to be used.
    "fcnet_hiddens": [256, 256],
    # Activation function descriptor.
    # Supported values are: "tanh", "relu", "swish" (or "silu"),
    # "linear" (or None).
    "fcnet_activation": "tanh",

    # VisionNetwork (tf and torch): rllib.models.tf|torch.visionnet.py
    # These are used if no custom model is specified and the input space is 2D.
    # Filter config: List of [out_channels, kernel, stride] for each filter.
    # Example:
    # Use None for making RLlib try to find a default filter setup given the
    # observation space.
    "conv_filters": None,
    # Activation function descriptor.
    # Supported values are: "tanh", "relu", "swish" (or "silu"),
    # "linear" (or None).
    "conv_activation": "relu",

    # Some default models support a final FC stack of n Dense layers with given
    # activation:
    # - Complex observation spaces: Image components are fed through
    #   VisionNets, flat Boxes are left as-is, Discrete are one-hot'd, then
    #   everything is concated and pushed through this final FC stack.
    # - VisionNets (CNNs), e.g. after the CNN stack, there may be
    #   additional Dense layers.
    # - FullyConnectedNetworks will have this additional FCStack as well
    # (that's why it's empty by default).
    "post_fcnet_hiddens": [],
    "post_fcnet_activation": "relu",

    # For DiagGaussian action distributions, make the second half of the model
    # outputs floating bias variables instead of state-dependent. This only
    # has an effect is using the default fully connected net.
    "free_log_std": False,
    # Whether to skip the final linear layer used to resize the hidden layer
    # outputs to size `num_outputs`. If True, then the last hidden layer
    # should already match num_outputs.
    "no_final_linear": False,
    # Whether layers should be shared for the value function.
    "vf_share_layers": True,

    # == LSTM ==
    # Whether to wrap the model with an LSTM.
    "use_lstm": False,
    # Max seq len for training the LSTM, defaults to 20.
    "max_seq_len": 20,
    # Size of the LSTM cell.
    "lstm_cell_size": 256,
    # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
    "lstm_use_prev_action": False,
    # Whether to feed r_{t-1} to LSTM.
    "lstm_use_prev_reward": False,
    # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
    "_time_major": False,

    # == Attention Nets (experimental: torch-version is untested) ==
    # Whether to use a GTrXL ("Gru transformer XL"; attention net) as the
    # wrapper Model around the default Model.
    "use_attention": False,
    # The number of transformer units within GTrXL.
    # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
    # b) a position-wise MLP.
    "attention_num_transformer_units": 1,
    # The input and output size of each transformer unit.
    "attention_dim": 64,
    # The number of attention heads within the MultiHeadAttention units.
    "attention_num_heads": 1,
    # The dim of a single head (within the MultiHeadAttention units).
    "attention_head_dim": 32,
    # The memory sizes for inference and training.
    "attention_memory_inference": 50,
    "attention_memory_training": 50,
    # The output dim of the position-wise MLP.
    "attention_position_wise_mlp_dim": 32,
    # The initial bias values for the 2 GRU gates within a transformer unit.
    "attention_init_gru_gate_bias": 2.0,
    # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
    "attention_use_n_prev_actions": 0,
    # Whether to feed r_{t-n:t-1} to GTrXL.
    "attention_use_n_prev_rewards": 0,

    # == Atari ==
    # Set to True to enable 4x stacking behavior.
    "framestack": True,
    # Final resized frame dimension
    "dim": 84,
    # (deprecated) Converts ATARI frame to 1 Channel Grayscale image
    "grayscale": False,
    # (deprecated) Changes frame to range from [-1, 1] if true
    "zero_mean": True,

    # === Options for custom models ===
    # Name of a custom model to use
    "custom_model": None,
    # Extra options to pass to the custom classes. These will be available to
    # the Model's constructor in the model_config field. Also, they will be
    # attempted to be passed as **kwargs to ModelV2 models. For an example,
    # see rllib/models/[tf|torch]/attention_net.py.
    "custom_model_config": {},
    # Name of a custom action distribution to use.
    "custom_action_dist": None,
    # Custom preprocessors are deprecated. Please use a wrapper class around
    # your environment instead to preprocess observations.
    "custom_preprocessor": None,

    # Deprecated keys:
    # Use `lstm_use_prev_action` or `lstm_use_prev_reward` instead.
    "lstm_use_prev_action_reward": None,
}

class Preprocessor:
    """Defines an abstract observation preprocessor function.

    Attributes:
        shape (List[int]): Shape of the preprocessed output.
    """

    def __init__(self, obs_space: gym.Space, options: dict = None):
        legacy_patch_shapes(obs_space)
        self._obs_space = obs_space
        if not options:
            self._options = MODEL_DEFAULTS.copy()
        else:
            self._options = options
        self.shape = self._init_shape(obs_space, self._options)
        self._size = int(np.product(self.shape))
        self._i = 0

    def _init_shape(self, obs_space: gym.Space, options: dict) -> List[int]:
        """Returns the shape after preprocessing."""
        raise NotImplementedError

    def transform(self, observation: TensorType) -> np.ndarray:
        """Returns the preprocessed observation."""
        raise NotImplementedError

    def write(self, observation: TensorType, array: np.ndarray,
              offset: int) -> None:
        """Alternative to transform for more efficient flattening."""
        array[offset:offset + self._size] = self.transform(observation)

    def check_shape(self, observation: Any) -> None:
        """Checks the shape of the given observation."""
        if self._i % OBS_VALIDATION_INTERVAL == 0:
            # Convert lists to np.ndarrays.
            if type(observation) is list and isinstance(
                    self._obs_space, gym.spaces.Box):
                observation = np.array(observation)
            # Ignore float32/float64 diffs.
            if isinstance(self._obs_space, gym.spaces.Box) and \
                    self._obs_space.dtype != observation.dtype:
                observation = observation.astype(self._obs_space.dtype)
            try:
                if not self._obs_space.contains(observation):
                    raise ValueError(
                        "Observation ({} dtype={}) outside given space ({})!",
                        observation, observation.dtype if isinstance(
                            self._obs_space,
                            gym.spaces.Box) else None, self._obs_space)
            except AttributeError:
                raise ValueError(
                    "Observation for a Box/MultiBinary/MultiDiscrete space "
                    "should be an np.array, not a Python list.", observation)
        self._i += 1

    @property
    def size(self) -> int:
        return self._size

    @property
    def observation_space(self) -> gym.Space:
        obs_space = gym.spaces.Box(-1., 1., self.shape, dtype=np.float32)
        # Stash the unwrapped space so that we can unwrap dict and tuple spaces
        # automatically in modelv2.py
        classes = (DictFlatteningPreprocessor, OneHotPreprocessor,
                   RepeatedValuesPreprocessor, TupleFlatteningPreprocessor)
        if isinstance(self, classes):
            obs_space.original_space = self._obs_space
        return obs_space

class NoPreprocessor(Preprocessor):
    def _init_shape(self, obs_space: gym.Space, options: dict) -> List[int]:
        return self._obs_space.shape

    def transform(self, observation: TensorType) -> np.ndarray:
        self.check_shape(observation)
        return observation

    def write(self, observation: TensorType, array: np.ndarray,
              offset: int) -> None:
        array[offset:offset + self._size] = np.array(
            observation, copy=False).ravel()

    @property
    def observation_space(self) -> gym.Space:
        return self._obs_space



def legacy_patch_shapes(space: gym.Space) -> List[int]:
    """Assigns shapes to spaces that don't have shapes.

    This is only needed for older gym versions that don't set shapes properly
    for Tuple and Discrete spaces.
    """

    if not hasattr(space, "shape"):
        if isinstance(space, gym.spaces.Discrete):
            space.shape = ()
        elif isinstance(space, gym.spaces.Tuple):
            shapes = []
            for s in space.spaces:
                shape = legacy_patch_shapes(s)
                shapes.append(shape)
            space.shape = tuple(shapes)

    return space.shape


def get_preprocessor(space: gym.Space) -> type:
    """Returns an appropriate preprocessor class for the given space."""

    legacy_patch_shapes(space)
    obs_shape = space.shape

    if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        preprocessor = OneHotPreprocessor
    elif obs_shape == ATARI_OBS_SHAPE:
        preprocessor = GenericPixelPreprocessor
    elif obs_shape == ATARI_RAM_OBS_SHAPE:
        preprocessor = AtariRamPreprocessor
    elif isinstance(space, gym.spaces.Tuple):
        preprocessor = TupleFlatteningPreprocessor
    elif isinstance(space, gym.spaces.Dict):
        preprocessor = DictFlatteningPreprocessor
    elif isinstance(space, Repeated):
        preprocessor = RepeatedValuesPreprocessor
    else:
        preprocessor = NoPreprocessor

    return preprocessor



class RepeatedValues:
    """Represents a variable-length list of items from spaces.Repeated.

    RepeatedValues are created when you use spaces.Repeated, and are
    accessible as part of input_dict["obs"] in ModelV2 forward functions.

    Example:
        Suppose the gym space definition was:
            Repeated(Repeated(Box(K), N), M)

        Then in the model forward function, input_dict["obs"] is of type:
            RepeatedValues(RepeatedValues(<Tensor shape=(B, M, N, K)>))

        The tensor is accessible via:
            input_dict["obs"].values.values

        And the actual data lengths via:
            # outer repetition, shape [B], range [0, M]
            input_dict["obs"].lengths
                -and-
            # inner repetition, shape [B, M], range [0, N]
            input_dict["obs"].values.lengths

    Attributes:
        values (Tensor): The padded data tensor of shape [B, max_len, ..., sz],
            where B is the batch dimension, max_len is the max length of this
            list, followed by any number of sub list max lens, followed by the
            actual data size.
        lengths (List[int]): Tensor of shape [B, ...] that represents the
            number of valid items in each list. When the list is nested within
            other lists, there will be extra dimensions for the parent list
            max lens.
        max_len (int): The max number of items allowed in each list.

    TODO(ekl): support conversion to tf.RaggedTensor.
    """

    def __init__(self, values: TensorType, lengths: List[int], max_len: int):
        self.values = values
        self.lengths = lengths
        self.max_len = max_len
        self._unbatched_repr = None

    def unbatch_all(self) -> List[List[TensorType]]:
        """Unbatch both the repeat and batch dimensions into Python lists.

        This is only supported in PyTorch / TF eager mode.

        This lets you view the data unbatched in its original form, but is
        not efficient for processing.

        Examples:
            >>> batch = RepeatedValues(<Tensor shape=(B, N, K)>)
            >>> items = batch.unbatch_all()
            >>> print(len(items) == B)
            True
            >>> print(max(len(x) for x in items) <= N)
            True
            >>> print(items)
            ... [[<Tensor_1 shape=(K)>, ..., <Tensor_N, shape=(K)>],
            ...  ...
            ...  [<Tensor_1 shape=(K)>, <Tensor_2 shape=(K)>],
            ...  ...
            ...  [<Tensor_1 shape=(K)>],
            ...  ...
            ...  [<Tensor_1 shape=(K)>, ..., <Tensor_N shape=(K)>]]
        """

        if self._unbatched_repr is None:
            B = _get_batch_dim_helper(self.values)
            if B is None:
                raise ValueError(
                    "Cannot call unbatch_all() when batch_dim is unknown. "
                    "This is probably because you are using TF graph mode.")
            else:
                B = int(B)
            slices = self.unbatch_repeat_dim()
            result = []
            for i in range(B):
                if hasattr(self.lengths[i], "item"):
                    dynamic_len = int(self.lengths[i].item())
                else:
                    dynamic_len = int(self.lengths[i].numpy())
                dynamic_slice = []
                for j in range(dynamic_len):
                    dynamic_slice.append(_batch_index_helper(slices, i, j))
                result.append(dynamic_slice)
            self._unbatched_repr = result

        return self._unbatched_repr

    def unbatch_repeat_dim(self) -> List[TensorType]:
        """Unbatches the repeat dimension (the one `max_len` in size).

        This removes the repeat dimension. The result will be a Python list of
        with length `self.max_len`. Note that the data is still padded.

        Examples:
            >>> batch = RepeatedValues(<Tensor shape=(B, N, K)>)
            >>> items = batch.unbatch()
            >>> len(items) == batch.max_len
            True
            >>> print(items)
            ... [<Tensor_1 shape=(B, K)>, ..., <Tensor_N shape=(B, K)>]
        """
        return _unbatch_helper(self.values, self.max_len)

    def __repr__(self):
        return "RepeatedValues(value={}, lengths={}, max_len={})".format(
            repr(self.values), repr(self.lengths), self.max_len)

    def __str__(self):
        return repr(self)


def _get_batch_dim_helper(v: TensorStructType) -> int:
    """Tries to find the batch dimension size of v, or None."""
    if isinstance(v, dict):
        for u in v.values():
            return _get_batch_dim_helper(u)
    elif isinstance(v, tuple):
        return _get_batch_dim_helper(v[0])
    elif isinstance(v, RepeatedValues):
        return _get_batch_dim_helper(v.values)
    else:
        B = v.shape[0]
        if hasattr(B, "value"):
            B = B.value  # TensorFlow
        return B


def _unbatch_helper(v: TensorStructType, max_len: int) -> TensorStructType:
    """Recursively unpacks the repeat dimension (max_len)."""
    if isinstance(v, dict):
        return {k: _unbatch_helper(u, max_len) for (k, u) in v.items()}
    elif isinstance(v, tuple):
        return tuple(_unbatch_helper(u, max_len) for u in v)
    elif isinstance(v, RepeatedValues):
        unbatched = _unbatch_helper(v.values, max_len)
        return [
            RepeatedValues(u, v.lengths[:, i, ...], v.max_len)
            for i, u in enumerate(unbatched)
        ]
    else:
        return [v[:, i, ...] for i in range(max_len)]


def _batch_index_helper(v: TensorStructType, i: int,
                        j: int) -> TensorStructType:
    """Selects the item at the ith batch index and jth repetition."""
    if isinstance(v, dict):
        return {k: _batch_index_helper(u, i, j) for (k, u) in v.items()}
    elif isinstance(v, tuple):
        return tuple(_batch_index_helper(u, i, j) for u in v)
    elif isinstance(v, list):
        # This is the output of unbatch_repeat_dim(). Unfortunately we have to
        # process it here instead of in unbatch_all(), since it may be buried
        # under a dict / tuple.
        return _batch_index_helper(v[j], i, j)
    elif isinstance(v, RepeatedValues):
        unbatched = v.unbatch_all()
        # Don't need to select j here; that's already done in unbatch_all.
        return unbatched[i]
    else:
        return v[i, ...]


class NullContextManager:
    """No-op context manager"""

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass




class Repeated(gym.Space):
    """Represents a variable-length list of child spaces.

    Example:
        self.observation_space = spaces.Repeated(spaces.Box(4,), max_len=10)
            --> from 0 to 10 boxes of shape (4,)

    See also: documentation for rllib.models.RepeatedValues, which shows how
        the lists are represented as batched input for ModelV2 classes.
    """

    def __init__(self, child_space: gym.Space, max_len: int):
        super().__init__()
        self.child_space = child_space
        self.max_len = max_len

    def sample(self):
        return [
            self.child_space.sample()
            for _ in range(self.np_random.randint(1, self.max_len + 1))
        ]

    def contains(self, x):
        return (isinstance(x, list) and len(x) <= self.max_len
                and all(self.child_space.contains(c) for c in x))

class ModelV2:
    """Defines an abstract neural network model for use with RLlib.

    Custom models should extend either TFModelV2 or TorchModelV2 instead of
    this class directly.

    Data flow:
        obs -> forward() -> model_out
               value_function() -> V(s)
    """

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, framework: str):
        """Initializes a ModelV2 object.

        This method should create any variables used by the model.

        Args:
            obs_space (gym.spaces.Space): Observation space of the target gym
                env. This may have an `original_space` attribute that
                specifies how to unflatten the tensor into a ragged tensor.
            action_space (gym.spaces.Space): Action space of the target gym
                env.
            num_outputs (int): Number of output units of the model.
            model_config (ModelConfigDict): Config for the model, documented
                in ModelCatalog.
            name (str): Name (scope) for the model.
            framework (str): Either "tf" or "torch".
        """

        self.obs_space: gym.spaces.Space = obs_space
        self.action_space: gym.spaces.Space = action_space
        self.num_outputs: int = num_outputs
        self.model_config: ModelConfigDict = model_config
        self.name: str = name or "default_model"
        self.framework: str = framework
        self._last_output = None
        self.time_major = self.model_config.get("_time_major")
        # Basic view requirement for all models: Use the observation as input.
        self.view_requirements = {
            SampleBatch.OBS: ViewRequirement(shift=0, space=self.obs_space),
        }

    # TODO: (sven): Get rid of `get_initial_state` once Trajectory
    #  View API is supported across all of RLlib.
    def get_initial_state(self) -> List[np.ndarray]:
        """Get the initial recurrent state values for the model.

        Returns:
            List[np.ndarray]: List of np.array objects containing the initial
                hidden state of an RNN, if applicable.

        Examples:
            >>> def get_initial_state(self):
            >>>    return [
            >>>        np.zeros(self.cell_size, np.float32),
            >>>        np.zeros(self.cell_size, np.float32),
            >>>    ]
        """
        return []

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """Call the model with the given input tensors and state.

        Any complex observations (dicts, tuples, etc.) will be unpacked by
        __call__ before being passed to forward(). To access the flattened
        observation tensor, refer to input_dict["obs_flat"].

        This method can be called any number of times. In eager execution,
        each call to forward() will eagerly evaluate the model. In symbolic
        execution, each call to forward creates a computation graph that
        operates over the variables of this model (i.e., shares weights).

        Custom models should override this instead of __call__.

        Args:
            input_dict (dict): dictionary of input tensors, including "obs",
                "obs_flat", "prev_action", "prev_reward", "is_training",
                "eps_id", "agent_id", "infos", and "t".
            state (list): list of state tensors with sizes matching those
                returned by get_initial_state + the batch dimension
            seq_lens (Tensor): 1d tensor holding input sequence lengths

        Returns:
            (outputs, state): The model output tensor of size
                [BATCH, num_outputs], and the new RNN state.

        Examples:
            >>> def forward(self, input_dict, state, seq_lens):
            >>>     model_out, self._value_out = self.base_model(
            ...         input_dict["obs"])
            >>>     return model_out, state
        """
        raise NotImplementedError

    def value_function(self) -> TensorType:
        """Returns the value function output for the most recent forward pass.

        Note that a `forward` call has to be performed first, before this
        methods can return anything and thus that calling this method does not
        cause an extra forward pass through the network.

        Returns:
            value estimate tensor of shape [BATCH].
        """
        raise NotImplementedError

    def custom_loss(self, policy_loss: TensorType,
                    loss_inputs: Dict[str, TensorType]) -> TensorType:
        """Override to customize the loss function used to optimize this model.

        This can be used to incorporate self-supervised losses (by defining
        a loss over existing input and output tensors of this model), and
        supervised losses (by defining losses over a variable-sharing copy of
        this model's layers).

        You can find an runnable example in examples/custom_loss.py.

        Args:
            policy_loss (Union[List[Tensor],Tensor]): List of or single policy
                loss(es) from the policy.
            loss_inputs (dict): map of input placeholders for rollout data.

        Returns:
            Union[List[Tensor],Tensor]: List of or scalar tensor for the
                customized loss(es) for this model.
        """
        return policy_loss

    def metrics(self) -> Dict[str, TensorType]:
        """Override to return custom metrics from your model.

        The stats will be reported as part of the learner stats, i.e.,
        info.learner.[policy_id, e.g. "default_policy"].model.key1=metric1

        Returns:
            Dict[str, TensorType]: The custom metrics for this model.
        """
        return {}

    def __call__(
            self,
            input_dict: Union[SampleBatch, ModelInputDict],
            state: List[Any] = None,
            seq_lens: TensorType = None) -> (TensorType, List[TensorType]):
        """Call the model with the given input tensors and state.

        This is the method used by RLlib to execute the forward pass. It calls
        forward() internally after unpacking nested observation tensors.

        Custom models should override forward() instead of __call__.

        Args:
            input_dict (Union[SampleBatch, ModelInputDict]): Dictionary of
                input tensors.
            state (list): list of state tensors with sizes matching those
                returned by get_initial_state + the batch dimension
            seq_lens (Tensor): 1D tensor holding input sequence lengths.

        Returns:
            (outputs, state): The model output tensor of size
                [BATCH, output_spec.size] or a list of tensors corresponding to
                output_spec.shape_list, and a list of state tensors of
                [BATCH, state_size_i].
        """

        # Original observations will be stored in "obs".
        # Flattened (preprocessed) obs will be stored in "obs_flat".

        # SampleBatch case: Models can now be called directly with a
        # SampleBatch (which also includes tracking-dict case (deprecated now),
        # where tensors get automatically converted).
        if isinstance(input_dict, SampleBatch):
            restored = input_dict.copy(shallow=True)
            # Backward compatibility.
            if seq_lens is None:
                seq_lens = input_dict.get(SampleBatch.SEQ_LENS)
            if not state:
                state = []
                i = 0
                while "state_in_{}".format(i) in input_dict:
                    state.append(input_dict["state_in_{}".format(i)])
                    i += 1
            input_dict["is_training"] = input_dict.is_training
        else:
            restored = input_dict.copy()

        # No Preprocessor used: `config.preprocessor_pref`=None.
        # TODO: This is unnecessary for when no preprocessor is used.
        #  Obs are not flat then anymore. However, we'll keep this
        #  here for backward-compatibility until Preprocessors have
        #  been fully deprecated.
        if self.model_config.get("_no_preprocessing"):
            restored["obs_flat"] = input_dict["obs"]
        # Input to this Model went through a Preprocessor.
        # Generate extra keys: "obs_flat" (vs "obs", which will hold the
        # original obs).
        else:
            restored["obs"] = restore_original_dimensions(
                input_dict["obs"], self.obs_space, self.framework)
            try:
                if len(input_dict["obs"].shape) > 2:
                    restored["obs_flat"] = flatten(input_dict["obs"],
                                                   self.framework)
                else:
                    restored["obs_flat"] = input_dict["obs"]
            except AttributeError:
                restored["obs_flat"] = input_dict["obs"]

        with self.context():
            res = self.forward(restored, state or [], seq_lens)

        if ((not isinstance(res, list) and not isinstance(res, tuple))
                or len(res) != 2):
            raise ValueError(
                "forward() must return a tuple of (output, state) tensors, "
                "got {}".format(res))
        outputs, state_out = res

        if not isinstance(state_out, list):
            raise ValueError(
                "State output is not a list: {}".format(state_out))

        self._last_output = outputs
        return outputs, state_out if len(state_out) > 0 else (state or [])

    # TODO: (sven) obsolete this method at some point (replace by
    #  simply calling model directly with a sample_batch as only input).
    def from_batch(self, train_batch: SampleBatch,
                   is_training: bool = True) -> (TensorType, List[TensorType]):
        """Convenience function that calls this model with a tensor batch.

        All this does is unpack the tensor batch to call this model with the
        right input dict, state, and seq len arguments.
        """

        input_dict = train_batch.copy()
        input_dict["is_training"] = is_training
        states = []
        i = 0
        while "state_in_{}".format(i) in input_dict:
            states.append(input_dict["state_in_{}".format(i)])
            i += 1
        ret = self.__call__(input_dict, states,
                            input_dict.get(SampleBatch.SEQ_LENS))
        return ret

    def import_from_h5(self, h5_file: str) -> None:
        """Imports weights from an h5 file.

        Args:
            h5_file (str): The h5 file name to import weights from.

        Example:
            >>> trainer = MyTrainer()
            >>> trainer.import_policy_model_from_h5("/tmp/weights.h5")
            >>> for _ in range(10):
            >>>     trainer.train()
        """
        raise NotImplementedError

    def last_output(self) -> TensorType:
        """Returns the last output returned from calling the model."""
        return self._last_output

    def context(self) -> contextlib.AbstractContextManager:
        """Returns a contextmanager for the current forward pass."""
        return NullContextManager()

    def variables(self, as_dict: bool = False
                  ) -> Union[List[TensorType], Dict[str, TensorType]]:
        """Returns the list (or a dict) of variables for this model.

        Args:
            as_dict(bool): Whether variables should be returned as dict-values
                (using descriptive str keys).

        Returns:
            Union[List[any],Dict[str,any]]: The list (or dict if `as_dict` is
                True) of all variables of this ModelV2.
        """
        raise NotImplementedError

    def trainable_variables(
            self, as_dict: bool = False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        """Returns the list of trainable variables for this model.

        Args:
            as_dict(bool): Whether variables should be returned as dict-values
                (using descriptive keys).

        Returns:
            Union[List[any],Dict[str,any]]: The list (or dict if `as_dict` is
                True) of all trainable (tf)/requires_grad (torch) variables
                of this ModelV2.
        """
        raise NotImplementedError

    def is_time_major(self) -> bool:
        """If True, data for calling this ModelV2 must be in time-major format.

        Returns
            bool: Whether this ModelV2 requires a time-major (TxBx...) data
                format.
        """
        return self.time_major is True


def flatten(obs: TensorType, framework: str) -> TensorType:
    """Flatten the given tensor."""
    if framework in ["tf2", "tf", "tfe"]:
        return tf1.keras.layers.Flatten()(obs)
    elif framework == "torch":
        assert torch is not None
        return torch.flatten(obs, start_dim=1)
    else:
        raise NotImplementedError("flatten", framework)


def restore_original_dimensions(obs: TensorType,
                                obs_space: gym.spaces.Space,
                                tensorlib) -> TensorStructType:
    """Unpacks Dict and Tuple space observations into their original form.

    This is needed since we flatten Dict and Tuple observations in transit
    within a SampleBatch. Before sending them to the model though, we should
    unflatten them into Dicts or Tuples of tensors.

    Args:
        obs (TensorType): The flattened observation tensor.
        obs_space (gym.spaces.Space): The flattened obs space. If this has the
            `original_space` attribute, we will unflatten the tensor to that
            shape.
        tensorlib: The library used to unflatten (reshape) the array/tensor.

    Returns:
        single tensor or dict / tuple of tensors matching the original
        observation space.
    """

    if tensorlib in ["tf", "tfe", "tf2"]:
        assert tf is not None
        tensorlib = tf
    elif tensorlib == "torch":
        assert torch is not None
        tensorlib = torch
    original_space = getattr(obs_space, "original_space", obs_space)
    return _unpack_obs(obs, original_space, tensorlib=tensorlib)


# Cache of preprocessors, for if the user is calling unpack obs often.
_cache = {}


def _unpack_obs(obs: TensorType, space: gym.Space,
                tensorlib) -> TensorStructType:
    """Unpack a flattened Dict or Tuple observation array/tensor.

    Args:
        obs: The flattened observation tensor, with last dimension equal to
            the flat size and any number of batch dimensions. For example, for
            Box(4,), the obs may have shape [B, 4], or [B, N, M, 4] in case
            the Box was nested under two Repeated spaces.
        space: The original space prior to flattening
        tensorlib: The library used to unflatten (reshape) the array/tensor
    """

    if isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple, Repeated)):
        if id(space) in _cache:
            prep = _cache[id(space)]
        else:
            prep = get_preprocessor(space)(space)
            # Make an attempt to cache the result, if enough space left.
            if len(_cache) < 999:
                _cache[id(space)] = prep
        # Already unpacked?
        if (isinstance(space, gym.spaces.Tuple) and
                isinstance(obs, (list, tuple))) or \
                (isinstance(space, gym.spaces.Dict) and isinstance(obs, dict)):
            return obs
        elif len(obs.shape) < 2 or obs.shape[-1] != prep.shape[0]:
            raise ValueError(
                "Expected flattened obs shape of [..., {}], got {}".format(
                    prep.shape[0], obs.shape))
        offset = 0
        if tensorlib == tf:
            batch_dims = [
                v if isinstance(v, int) else v.value for v in obs.shape[:-1]
            ]
            batch_dims = [-1 if v is None else v for v in batch_dims]
        else:
            batch_dims = list(obs.shape[:-1])
        if isinstance(space, gym.spaces.Tuple):
            assert len(prep.preprocessors) == len(space.spaces), \
                (len(prep.preprocessors) == len(space.spaces))
            u = []
            for p, v in zip(prep.preprocessors, space.spaces):
                obs_slice = obs[..., offset:offset + p.size]
                offset += p.size
                u.append(
                    _unpack_obs(
                        tensorlib.reshape(obs_slice,
                                          batch_dims + list(p.shape)),
                        v,
                        tensorlib=tensorlib))
        elif isinstance(space, gym.spaces.Dict):
            assert len(prep.preprocessors) == len(space.spaces), \
                (len(prep.preprocessors) == len(space.spaces))
            u = OrderedDict()
            for p, (k, v) in zip(prep.preprocessors, space.spaces.items()):
                obs_slice = obs[..., offset:offset + p.size]
                offset += p.size
                u[k] = _unpack_obs(
                    tensorlib.reshape(obs_slice, batch_dims + list(p.shape)),
                    v,
                    tensorlib=tensorlib)
        # Repeated space.
        else:
            assert isinstance(prep, RepeatedValuesPreprocessor), prep
            child_size = prep.child_preprocessor.size
            # The list lengths are stored in the first slot of the flat obs.
            lengths = obs[..., 0]
            # [B, ..., 1 + max_len * child_sz] -> [B, ..., max_len, child_sz]
            with_repeat_dim = tensorlib.reshape(
                obs[..., 1:], batch_dims + [space.max_len, child_size])
            # Retry the unpack, dropping the List container space.
            u = _unpack_obs(
                with_repeat_dim, space.child_space, tensorlib=tensorlib)
            return RepeatedValues(
                u, lengths=lengths, max_len=prep._obs_space.max_len)
        return u
    else:
        return obs


def _get_batch_dim_helper(v: TensorStructType) -> int:
    """Tries to find the batch dimension size of v, or None."""
    if isinstance(v, dict):
        for u in v.values():
            return _get_batch_dim_helper(u)
    elif isinstance(v, tuple):
        return _get_batch_dim_helper(v[0])
    elif isinstance(v, RepeatedValues):
        return _get_batch_dim_helper(v.values)
    else:
        B = v.shape[0]
        if hasattr(B, "value"):
            B = B.value  # TensorFlow
        return B

def _batch_index_helper(v: TensorStructType, i: int,
                        j: int) -> TensorStructType:
    """Selects the item at the ith batch index and jth repetition."""
    if isinstance(v, dict):
        return {k: _batch_index_helper(u, i, j) for (k, u) in v.items()}
    elif isinstance(v, tuple):
        return tuple(_batch_index_helper(u, i, j) for u in v)
    elif isinstance(v, list):
        # This is the output of unbatch_repeat_dim(). Unfortunately we have to
        # process it here instead of in unbatch_all(), since it may be buried
        # under a dict / tuple.
        return _batch_index_helper(v[j], i, j)
    elif isinstance(v, RepeatedValues):
        unbatched = v.unbatch_all()
        # Don't need to select j here; that's already done in unbatch_all.
        return unbatched[i]
    else:
        return v[i, ...]

def _unbatch_helper(v: TensorStructType, max_len: int) -> TensorStructType:
    """Recursively unpacks the repeat dimension (max_len)."""
    if isinstance(v, dict):
        return {k: _unbatch_helper(u, max_len) for (k, u) in v.items()}
    elif isinstance(v, tuple):
        return tuple(_unbatch_helper(u, max_len) for u in v)
    elif isinstance(v, RepeatedValues):
        unbatched = _unbatch_helper(v.values, max_len)
        return [
            RepeatedValues(u, v.lengths[:, i, ...], v.max_len)
            for i, u in enumerate(unbatched)
        ]
    else:
        return [v[:, i, ...] for i in range(max_len)]



class RepeatedValues:
    """Represents a variable-length list of items from spaces.Repeated.

    RepeatedValues are created when you use spaces.Repeated, and are
    accessible as part of input_dict["obs"] in ModelV2 forward functions.

    Example:
        Suppose the gym space definition was:
            Repeated(Repeated(Box(K), N), M)

        Then in the model forward function, input_dict["obs"] is of type:
            RepeatedValues(RepeatedValues(<Tensor shape=(B, M, N, K)>))

        The tensor is accessible via:
            input_dict["obs"].values.values

        And the actual data lengths via:
            # outer repetition, shape [B], range [0, M]
            input_dict["obs"].lengths
                -and-
            # inner repetition, shape [B, M], range [0, N]
            input_dict["obs"].values.lengths

    Attributes:
        values (Tensor): The padded data tensor of shape [B, max_len, ..., sz],
            where B is the batch dimension, max_len is the max length of this
            list, followed by any number of sub list max lens, followed by the
            actual data size.
        lengths (List[int]): Tensor of shape [B, ...] that represents the
            number of valid items in each list. When the list is nested within
            other lists, there will be extra dimensions for the parent list
            max lens.
        max_len (int): The max number of items allowed in each list.

    TODO(ekl): support conversion to tf.RaggedTensor.
    """

    def __init__(self, values: TensorType, lengths: List[int], max_len: int):
        self.values = values
        self.lengths = lengths
        self.max_len = max_len
        self._unbatched_repr = None

    def unbatch_all(self) -> List[List[TensorType]]:
        """Unbatch both the repeat and batch dimensions into Python lists.

        This is only supported in PyTorch / TF eager mode.

        This lets you view the data unbatched in its original form, but is
        not efficient for processing.

        Examples:
            >>> batch = RepeatedValues(<Tensor shape=(B, N, K)>)
            >>> items = batch.unbatch_all()
            >>> print(len(items) == B)
            True
            >>> print(max(len(x) for x in items) <= N)
            True
            >>> print(items)
            ... [[<Tensor_1 shape=(K)>, ..., <Tensor_N, shape=(K)>],
            ...  ...
            ...  [<Tensor_1 shape=(K)>, <Tensor_2 shape=(K)>],
            ...  ...
            ...  [<Tensor_1 shape=(K)>],
            ...  ...
            ...  [<Tensor_1 shape=(K)>, ..., <Tensor_N shape=(K)>]]
        """

        if self._unbatched_repr is None:
            B = _get_batch_dim_helper(self.values)
            if B is None:
                raise ValueError(
                    "Cannot call unbatch_all() when batch_dim is unknown. "
                    "This is probably because you are using TF graph mode.")
            else:
                B = int(B)
            slices = self.unbatch_repeat_dim()
            result = []
            for i in range(B):
                if hasattr(self.lengths[i], "item"):
                    dynamic_len = int(self.lengths[i].item())
                else:
                    dynamic_len = int(self.lengths[i].numpy())
                dynamic_slice = []
                for j in range(dynamic_len):
                    dynamic_slice.append(_batch_index_helper(slices, i, j))
                result.append(dynamic_slice)
            self._unbatched_repr = result

        return self._unbatched_repr

    def unbatch_repeat_dim(self) -> List[TensorType]:
        """Unbatches the repeat dimension (the one `max_len` in size).

        This removes the repeat dimension. The result will be a Python list of
        with length `self.max_len`. Note that the data is still padded.

        Examples:
            >>> batch = RepeatedValues(<Tensor shape=(B, N, K)>)
            >>> items = batch.unbatch()
            >>> len(items) == batch.max_len
            True
            >>> print(items)
            ... [<Tensor_1 shape=(B, K)>, ..., <Tensor_N shape=(B, K)>]
        """
        return _unbatch_helper(self.values, self.max_len)

    def __repr__(self):
        return "RepeatedValues(value={}, lengths={}, max_len={})".format(
            repr(self.values), repr(self.lengths), self.max_len)

    def __str__(self):
        return repr(self)


def convert_to_torch_tensor(x, device=None):
    """Converts any struct to torch.Tensors.

    x (any): Any (possibly nested) struct, the values in which will be
        converted and returned as a new struct with all leaves converted
        to torch tensors.

    Returns:
        Any: A new struct with the same structure as `stats`, but with all
            values converted to torch Tensor types.
    """

    def mapping(item):
        # Already torch tensor -> make sure it's on right device.
        if torch.is_tensor(item):
            return item if device is None else item.to(device)
        # Special handling of "Repeated" values.
        elif isinstance(item, RepeatedValues):
            return RepeatedValues(
                tree.map_structure(mapping, item.values), item.lengths,
                item.max_len)
        # Numpy arrays.
        if isinstance(item, np.ndarray):
            # np.object_ type (e.g. info dicts in train batch): leave as-is.
            if item.dtype == np.object_:
                return item
            # Non-writable numpy-arrays will cause PyTorch warning.
            elif item.flags.writeable is False:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tensor = torch.from_numpy(item)
            # Already numpy: Wrap as torch tensor.
            else:
                tensor = torch.from_numpy(item)
        # Everything else: Convert to numpy, then wrap as torch tensor.
        else:
            tensor = torch.from_numpy(np.asarray(item))
        # Floatify all float64 tensors.
        if tensor.dtype == torch.double:
            tensor = tensor.float()
        return tensor if device is None else tensor.to(device)

    return tree.map_structure(mapping, x)


def convert_to_non_torch_type(stats):
    """Converts values in `stats` to non-Tensor numpy or python types.

    Args:
        stats (any): Any (possibly nested) struct, the values in which will be
            converted and returned as a new struct with all torch tensors
            being converted to numpy types.

    Returns:
        Any: A new struct with the same structure as `stats`, but with all
            values converted to non-torch Tensor types.
    """

    # The mapping function used to numpyize torch Tensors.
    def mapping(item):
        if isinstance(item, torch.Tensor):
            return item.cpu().item() if len(item.size()) == 0 else \
                item.detach().cpu().numpy()
        else:
            return item

    return tree.map_structure(mapping, stats)

class Simplex(gym.Space):
    """Represents a d - 1 dimensional Simplex in R^d.

    That is, all coordinates are in [0, 1] and sum to 1.
    The dimension d of the simplex is assumed to be shape[-1].

    Additionally one can specify the underlying distribution of
    the simplex as a Dirichlet distribution by providing concentration
    parameters. By default, sampling is uniform, i.e. concentration is
    all 1s.

    Example usage:
    self.action_space = spaces.Simplex(shape=(3, 4))
        --> 3 independent 4d Dirichlet with uniform concentration
    """

    def __init__(self, shape, concentration=None, dtype=np.float32):
        assert type(shape) in [tuple, list]
        self.shape = shape
        self.dtype = dtype
        self.dim = shape[-1]

        if concentration is not None:
            assert concentration.shape == shape[:-1]
        else:
            self.concentration = [1] * self.dim

        super().__init__(shape, dtype)

    def seed(self, seed=None):
        if self.np_random is None:
            self.np_random = np.random.RandomState()
        self.np_random.seed(seed)

    def sample(self):
        return np.random.dirichlet(
            self.concentration, size=self.shape[:-1]).astype(self.dtype)

    def contains(self, x):
        return x.shape == self.shape and np.allclose(
            np.sum(x, axis=-1), np.ones_like(x[..., 0]))

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Simplex({}; {})".format(self.shape, self.concentration)

    def __eq__(self, other):
        return np.allclose(self.concentration,
                           other.concentration) and self.shape == other.shape

class FlexDict(gym.spaces.Dict):
    """Gym Dictionary with arbitrary keys updatable after instantiation

    Example:
       space = FlexDict({})
       space['key'] = spaces.Box(4,)
    See also: documentation for gym.spaces.Dict
    """

    def __init__(self, spaces=None, **spaces_kwargs):
        err = "Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)"
        assert (spaces is None) or (not spaces_kwargs), err

        if spaces is None:
            spaces = spaces_kwargs

        self.spaces = spaces
        for space in spaces.values():
            self.assertSpace(space)

        # None for shape and dtype, since it'll require special handling
        self.np_random = None
        self.shape = None
        self.dtype = None
        self.seed()

    def assertSpace(self, space):
        err = "Values of the dict should be instances of gym.Space"
        assert issubclass(type(space), gym.spaces.Space), err

    def sample(self):
        return {k: space.sample() for k, space in self.spaces.items()}

    def __getitem__(self, key):
        return self.spaces[key]

    def __setitem__(self, key, space):
        self.assertSpace(space)
        self.spaces[key] = space

    def __repr__(self):
        return "FlexDict(" + ", ".join(
            [str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"


def flatten_space(space: gym.Space) -> List[gym.Space]:
    """Flattens a gym.Space into its primitive components.

    Primitive components are any non Tuple/Dict spaces.

    Args:
        space (gym.Space): The gym.Space to flatten. This may be any
            supported type (including nested Tuples and Dicts).

    Returns:
        List[gym.Space]: The flattened list of primitive Spaces. This list
            does not contain Tuples or Dicts anymore.
    """

    def _helper_flatten(space_, return_list):
        # from ray.rllib.utils.spaces.flexdict import FlexDict
        if isinstance(space_, Tuple):
            for s in space_:
                _helper_flatten(s, return_list)
        elif isinstance(space_, (Dict, FlexDict)):
            for k in space_.spaces:
                _helper_flatten(space_[k], return_list)
        else:
            return_list.append(space_)

    ret = []
    _helper_flatten(space, ret)
    return ret



def get_action_shape(action_space: gym.Space,
                         framework: str = "tf") -> (np.dtype, List[int]):
        """Returns action tensor dtype and shape for the action space.

        Args:
            action_space (Space): Action space of the target gym env.
            framework (str): The framework identifier. One of "tf" or "torch".

        Returns:
            (dtype, shape): Dtype and shape of the actions tensor.
        """
        dl_lib = torch if framework == "torch" else tf

        if isinstance(action_space, Discrete):
            return action_space.dtype, (None, )
        elif isinstance(action_space, (Box, Simplex)):
            return dl_lib.float32, (None, ) + action_space.shape
        elif isinstance(action_space, MultiDiscrete):
            return action_space.dtype, (None, ) + action_space.shape
        elif isinstance(action_space, (Tuple, Dict)):
            flat_action_space = flatten_space(action_space)
            size = 0
            all_discrete = True
            for i in range(len(flat_action_space)):
                if isinstance(flat_action_space[i], Discrete):
                    size += 1
                else:
                    all_discrete = False
                    size += np.product(flat_action_space[i].shape)
            size = int(size)
            return dl_lib.int64 if all_discrete else dl_lib.float32, \
                (None, size)
        else:
            raise NotImplementedError(
                "Action space {} not supported".format(action_space))

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

class Exploration:
    """Implements an exploration strategy for Policies.

    An Exploration takes model outputs, a distribution, and a timestep from
    the agent and computes an action to apply to the environment using an
    implemented exploration schema.
    """

    def __init__(self, action_space: Space, *, framework: str,
                 policy_config: TrainerConfigDict, model: ModelV2,
                 num_workers: int, worker_index: int):
        """
        Args:
            action_space (Space): The action space in which to explore.
            framework (str): One of "tf" or "torch".
            policy_config (TrainerConfigDict): The Policy's config dict.
            model (ModelV2): The Policy's model.
            num_workers (int): The overall number of workers used.
            worker_index (int): The index of the worker using this class.
        """
        self.action_space = action_space
        self.policy_config = policy_config
        self.model = model
        self.num_workers = num_workers
        self.worker_index = worker_index
        self.framework = framework
        # The device on which the Model has been placed.
        # This Exploration will be on the same device.
        self.device = None
        if isinstance(self.model, nn.Module):
            params = list(self.model.parameters())
            if params:
                self.device = params[0].device

    def before_compute_actions(
            self,
            *,
            timestep: Optional[Union[TensorType, int]] = None,
            explore: Optional[Union[TensorType, bool]] = None,
            tf_sess: Optional["tf.Session"] = None,
            **kwargs):
        """Hook for preparations before policy.compute_actions() is called.

        Args:
            timestep (Optional[Union[TensorType, int]]): An optional timestep
                tensor.
            explore (Optional[Union[TensorType, bool]]): An optional explore
                boolean flag.
            tf_sess (Optional[tf.Session]): The tf-session object to use.
            **kwargs: Forward compatibility kwargs.
        """
        pass

    # yapf: disable
    # __sphinx_doc_begin_get_exploration_action__

    def get_exploration_action(self,
                               *,
                               action_distribution: ActionDistribution,
                               timestep: Union[TensorType, int],
                               explore: bool = True):
        """Returns a (possibly) exploratory action and its log-likelihood.

        Given the Model's logits outputs and action distribution, returns an
        exploratory action.

        Args:
            action_distribution (ActionDistribution): The instantiated
                ActionDistribution object to work with when creating
                exploration actions.
            timestep (Union[TensorType, int]): The current sampling time step.
                It can be a tensor for TF graph mode, otherwise an integer.
            explore (Union[TensorType, bool]): True: "Normal" exploration
                behavior. False: Suppress all exploratory behavior and return
                a deterministic action.

        Returns:
            Tuple:
            - The chosen exploration action or a tf-op to fetch the exploration
              action from the graph.
            - The log-likelihood of the exploration action.
        """
        pass

    # __sphinx_doc_end_get_exploration_action__
    # yapf: enable

    def on_episode_start(self,
                         policy: "Policy",
                         *,
                         environment = None,
                         episode: int = None,
                         tf_sess: Optional["tf.Session"] = None):
        """Handles necessary exploration logic at the beginning of an episode.

        Args:
            policy (Policy): The Policy object that holds this Exploration.
            environment (BaseEnv): The environment object we are acting in.
            episode (int): The number of the episode that is starting.
            tf_sess (Optional[tf.Session]): In case of tf, the session object.
        """
        pass

    def on_episode_end(self,
                       policy: "Policy",
                       *,
                       environment = None,
                       episode: int = None,
                       tf_sess: Optional["tf.Session"] = None):
        """Handles necessary exploration logic at the end of an episode.

        Args:
            policy (Policy): The Policy object that holds this Exploration.
            environment (BaseEnv): The environment object we are acting in.
            episode (int): The number of the episode that is starting.
            tf_sess (Optional[tf.Session]): In case of tf, the session object.
        """
        pass

    def postprocess_trajectory(self,
                               policy: "Policy",
                               sample_batch: SampleBatch,
                               tf_sess: Optional["tf.Session"] = None):
        """Handles post-processing of done episode trajectories.

        Changes the given batch in place. This callback is invoked by the
        sampler after policy.postprocess_trajectory() is called.

        Args:
            policy (Policy): The owning policy object.
            sample_batch (SampleBatch): The SampleBatch object to post-process.
            tf_sess (Optional[tf.Session]): An optional tf.Session object.
        """
        return sample_batch

    def get_exploration_optimizer(self, optimizers: List[LocalOptimizer]) -> \
            List[LocalOptimizer]:
        """May add optimizer(s) to the Policy's own `optimizers`.

        The number of optimizers (Policy's plus Exploration's optimizers) must
        match the number of loss terms produced by the Policy's loss function
        and the Exploration component's loss terms.

        Args:
            optimizers (List[LocalOptimizer]): The list of the Policy's
                local optimizers.

        Returns:
            List[LocalOptimizer]: The updated list of local optimizers to use
                on the different loss terms.
        """
        return optimizers

    def get_state(self, sess: Optional["tf.Session"] = None) -> \
            Dict[str, TensorType]:
        """Returns the current exploration state.

        Args:
            sess (Optional[tf.Session]): An optional tf Session object to use.

        Returns:
            Dict[str, TensorType]: The Exploration object's current state.
        """
        return {}

    def set_state(self, state: object,
                  sess: Optional["tf.Session"] = None) -> None:
        """Sets the Exploration object's state to the given values.

        Note that some exploration components are stateless, even though they
        decay some values over time (e.g. EpsilonGreedy). However the decay is
        only dependent on the current global timestep of the policy and we
        therefore don't need to keep track of it.

        Args:
            state (object): The state to set this Exploration to.
            sess (Optional[tf.Session]): An optional tf Session object to use.
        """
        pass

    def get_info(self, sess: Optional["tf.Session"] = None):
        return self.get_state(sess)


class Random(Exploration):
    """A random action selector (deterministic/greedy for explore=False).

    If explore=True, returns actions randomly from `self.action_space` (via
    Space.sample()).
    If explore=False, returns the greedy/max-likelihood action.
    """

    def __init__(self, action_space: Space, *, model: ModelV2,
                 framework: Optional[str], **kwargs):
        """Initialize a Random Exploration object.

        Args:
            action_space (Space): The gym action space used by the environment.
            framework (Optional[str]): One of None, "tf", "tfe", "torch".
        """
        super().__init__(
            action_space=action_space,
            model=model,
            framework=framework,
            **kwargs)

        self.action_space_struct = get_base_struct_from_space(
            self.action_space)

    def get_exploration_action(self,
                               *,
                               action_distribution: ActionDistribution,
                               timestep: Union[int, TensorType],
                               explore: bool = True):
        # Instantiate the distribution object.
        if self.framework in ["tf2", "tf", "tfe"]:
            return self.get_tf_exploration_action_op(action_distribution,
                                                     explore)
        else:
            return self.get_torch_exploration_action(action_distribution,
                                                     explore)

    def get_tf_exploration_action_op(
            self, action_dist: ActionDistribution,
            explore: Optional[Union[bool, TensorType]]):
        def true_fn():
            batch_size = 1
            req = force_tuple(
                action_dist.required_model_output_shape(
                    self.action_space, getattr(self.model, "model_config",
                                               None)))
            # Add a batch dimension?
            if len(action_dist.inputs.shape) == len(req) + 1:
                batch_size = tf.shape(action_dist.inputs)[0]

            # Function to produce random samples from primitive space
            # components: (Multi)Discrete or Box.
            def random_component(component):
                # Have at least an additional shape of (1,), even if the
                # component is Box(-1.0, 1.0, shape=()).
                shape = component.shape or (1, )

                if isinstance(component, Discrete):
                    return tf.random.uniform(
                        shape=(batch_size, ) + component.shape,
                        maxval=component.n,
                        dtype=component.dtype)
                elif isinstance(component, MultiDiscrete):
                    return tf.concat(
                        [
                            tf.random.uniform(
                                shape=(batch_size, 1),
                                maxval=n,
                                dtype=component.dtype) for n in component.nvec
                        ],
                        axis=1)
                elif isinstance(component, Box):
                    if component.bounded_above.all() and \
                            component.bounded_below.all():
                        if component.dtype.name.startswith("int"):
                            return tf.random.uniform(
                                shape=(batch_size, ) + shape,
                                minval=component.low.flat[0],
                                maxval=component.high.flat[0],
                                dtype=component.dtype)
                        else:
                            return tf.random.uniform(
                                shape=(batch_size, ) + shape,
                                minval=component.low,
                                maxval=component.high,
                                dtype=component.dtype)
                    else:
                        return tf.random.normal(
                            shape=(batch_size, ) + shape,
                            dtype=component.dtype)
                else:
                    assert isinstance(component, Simplex), \
                        "Unsupported distribution component '{}' for random " \
                        "sampling!".format(component)
                    return tf.nn.softmax(
                        tf.random.uniform(
                            shape=(batch_size, ) + shape,
                            minval=0.0,
                            maxval=1.0,
                            dtype=component.dtype))

            actions = tree.map_structure(random_component,
                                         self.action_space_struct)
            return actions

        def false_fn():
            return action_dist.deterministic_sample()

        action = tf.cond(
            pred=tf.constant(explore, dtype=tf.bool)
            if isinstance(explore, bool) else explore,
            true_fn=true_fn,
            false_fn=false_fn)

        logp = zero_logps_from_actions(action)
        return action, logp

    def get_torch_exploration_action(self, action_dist: ActionDistribution,
                                     explore: bool):
        if explore:
            req = force_tuple(
                action_dist.required_model_output_shape(
                    self.action_space, getattr(self.model, "model_config",
                                               None)))
            # Add a batch dimension?
            if len(action_dist.inputs.shape) == len(req) + 1:
                batch_size = action_dist.inputs.shape[0]
                a = np.stack(
                    [self.action_space.sample() for _ in range(batch_size)])
            else:
                a = self.action_space.sample()
            # Convert action to torch tensor.
            action = torch.from_numpy(a).to(self.device)
        else:
            action = action_dist.deterministic_sample()
        logp = torch.zeros(
            (action.size()[0], ), dtype=torch.float32, device=self.device)
        return action, logp

def get_variable(value,
                 framework: str = "tf",
                 trainable: bool = False,
                 tf_name: str = "unnamed-variable",
                 torch_tensor: bool = False,
                 device: Optional[str] = None,
                 shape = None,
                 dtype: Optional[Any] = None):
    """
    Args:
        value (any): The initial value to use. In the non-tf case, this will
            be returned as is. In the tf case, this could be a tf-Initializer
            object.
        framework (str): One of "tf", "torch", or None.
        trainable (bool): Whether the generated variable should be
            trainable (tf)/require_grad (torch) or not (default: False).
        tf_name (str): For framework="tf": An optional name for the
            tf.Variable.
        torch_tensor (bool): For framework="torch": Whether to actually create
            a torch.tensor, or just a python value (default).
        device (Optional[torch.Device]): An optional torch device to use for
            the created torch tensor.
        shape (Optional[TensorShape]): An optional shape to use iff `value`
            does not have any (e.g. if it's an initializer w/o explicit value).
        dtype (Optional[TensorType]): An optional dtype to use iff `value` does
            not have any (e.g. if it's an initializer w/o explicit value).
            This should always be a numpy dtype (e.g. np.float32, np.int64).

    Returns:
        any: A framework-specific variable (tf.Variable, torch.tensor, or
            python primitive).
    """
    if framework in ["tf2", "tf", "tfe"]:
        import tensorflow as tf
        dtype = dtype or getattr(
            value, "dtype", tf.float32
            if isinstance(value, float) else tf.int32
            if isinstance(value, int) else None)
        return tf.compat.v1.get_variable(
            tf_name,
            initializer=value,
            dtype=dtype,
            trainable=trainable,
            **({} if shape is None else {
                "shape": shape
            }))
    elif framework == "torch" and torch_tensor is True:
        var_ = torch.from_numpy(value)
        if dtype in [torch.float32, np.float32]:
            var_ = var_.float()
        elif dtype in [torch.int32, np.int32]:
            var_ = var_.int()
        elif dtype in [torch.float64, np.float64]:
            var_ = var_.double()

        if device:
            var_ = var_.to(device)
        var_.requires_grad = trainable
        return var_
    # torch or None: Return python primitive.
    return value

class StochasticSampling(Exploration):
    """An exploration that simply samples from a distribution.
    The sampling can be made deterministic by passing explore=False into
    the call to `get_exploration_action`.
    Also allows for scheduled parameters for the distributions, such as
    lowering stddev, temperature, etc.. over time.
    """

    def __init__(
        self,
        action_space: gym.spaces.Space,
        *,
        framework: str,
        model: ModelV2,
        random_timesteps: int = 0,
        **kwargs
    ):
        """Initializes a StochasticSampling Exploration object.
        Args:
            action_space: The gym action space used by the environment.
            framework: One of None, "tf", "torch".
            model: The ModelV2 used by the owning Policy.
            random_timesteps: The number of timesteps for which to act
                completely randomly. Only after this number of timesteps,
                actual samples will be drawn to get exploration actions.
        """
        assert framework is not None
        super().__init__(action_space, model=model, framework=framework, **kwargs)

        # Create the Random exploration module (used for the first n
        # timesteps).
        self.random_timesteps = random_timesteps
        self.random_exploration = Random(
            action_space, model=self.model, framework=self.framework, **kwargs
        )

        # The current timestep value (tf-var or python int).
        self.last_timestep = get_variable(
            np.array(0, np.int64),
            framework=self.framework,
            tf_name="timestep",
            dtype=np.int64,
        )

    def get_exploration_action(
        self,
        *,
        action_distribution: ActionDistribution,
        timestep: Optional[Union[int, TensorType]] = None,
        explore: bool = True
    ):
        if self.framework == "torch":
            return self._get_torch_exploration_action(
                action_distribution, timestep, explore
            )
        else:
            return self._get_tf_exploration_action_op(
                action_distribution, timestep, explore
            )

    def _get_tf_exploration_action_op(self, action_dist, timestep, explore):
        ts = self.last_timestep + 1

        stochastic_actions = tf.cond(
            pred=tf.convert_to_tensor(ts < self.random_timesteps),
            true_fn=lambda: (
                self.random_exploration.get_tf_exploration_action_op(
                    action_dist, explore=True
                )[0]
            ),
            false_fn=lambda: action_dist.sample(),
        )
        deterministic_actions = action_dist.deterministic_sample()

        action = tf.cond(
            tf.constant(explore) if isinstance(explore, bool) else explore,
            true_fn=lambda: stochastic_actions,
            false_fn=lambda: deterministic_actions,
        )

        logp = tf.cond(
            tf.math.logical_and(
                explore, tf.convert_to_tensor(ts >= self.random_timesteps)
            ),
            true_fn=lambda: action_dist.sampled_action_logp(),
            false_fn=functools.partial(zero_logps_from_actions, deterministic_actions),
        )

        # Increment `last_timestep` by 1 (or set to `timestep`).
        if self.framework in ["tf2", "tfe"]:
            self.last_timestep.assign_add(1)
            return action, logp
        else:
            assign_op = (
                tf1.assign_add(self.last_timestep, 1)
                if timestep is None
                else tf1.assign(self.last_timestep, timestep)
            )
            with tf1.control_dependencies([assign_op]):
                return action, logp

    def _get_torch_exploration_action(
        self,
        action_dist: ActionDistribution,
        timestep: Union[TensorType, int],
        explore: Union[TensorType, bool],
    ):
        # Set last timestep or (if not given) increase by one.
        self.last_timestep = (
            timestep if timestep is not None else self.last_timestep + 1
        )

        # Apply exploration.
        if explore:
            # Random exploration phase.
            if self.last_timestep < self.random_timesteps:
                action, logp = self.random_exploration.get_torch_exploration_action(
                    action_dist, explore=True
                )
            # Take a sample from our distribution.
            else:
                action = action_dist.sample()
                logp = action_dist.sampled_action_logp()

        # No exploration -> Return deterministic actions.
        else:
            action = action_dist.deterministic_sample()
            logp = torch.zeros_like(action_dist.sampled_action_logp())

        return action, logp

def unsquash_action(action, action_space_struct):
    """Unsquashes all components in `action` according to the given Space.

    Inverse of `normalize_action()`. Useful for mapping policy action
    outputs (normalized between -1.0 and 1.0) to an env's action space.
    Unsquashing results in cont. action component values between the
    given Space's bounds (`low` and `high`). This only applies to Box
    components within the action space, whose dtype is float32 or float64.

    Args:
        action (Any): The action to be unsquashed. This could be any complex
            action, e.g. a dict or tuple.
        action_space_struct (Any): The action space struct,
            e.g. `{"a": Box()}` for a space: Dict({"a": Box()}).

    Returns:
        Any: The input action, but unsquashed, according to the space's
            bounds. An unsquashed action is ready to be sent to the
            environment (`BaseEnv.send_actions([unsquashed actions])`).
    """

    def map_(a, s):
        if isinstance(s, gym.spaces.Box) and \
                (s.dtype == np.float32 or s.dtype == np.float64):
            # Assuming values are roughly between -1.0 and 1.0 ->
            # unsquash them to the given bounds.
            a = s.low + (a + 1.0) * (s.high - s.low) / 2.0
            # Clip to given bounds, just in case the squashed values were
            # outside [-1.0, 1.0].
            a = np.clip(a, s.low, s.high)
        return a

    return tree.map_structure(map_, action, action_space_struct)


def unbatch(batches_struct):
    """Converts input from (nested) struct of batches to batch of structs.

    Input: Struct of different batches (each batch has size=3):
        {"a": [1, 2, 3], "b": ([4, 5, 6], [7.0, 8.0, 9.0])}
    Output: Batch (list) of structs (each of these structs representing a
        single action):
        [
            {"a": 1, "b": (4, 7.0)},  <- action 1
            {"a": 2, "b": (5, 8.0)},  <- action 2
            {"a": 3, "b": (6, 9.0)},  <- action 3
        ]

    Args:
        batches_struct (any): The struct of component batches. Each leaf item
            in this struct represents the batch for a single component
            (in case struct is tuple/dict).
            Alternatively, `batches_struct` may also simply be a batch of
            primitives (non tuple/dict).

    Returns:
        List[struct[components]]: The list of rows. Each item
            in the returned list represents a single (maybe complex) struct.
    """
    flat_batches = tree.flatten(batches_struct)

    out = []
    for batch_pos in range(len(flat_batches[0])):
        out.append(
            tree.unflatten_as(
                batches_struct,
                [flat_batches[i][batch_pos]
                 for i in range(len(flat_batches))]))
    return out


def get_dummy_batch_for_space(
        space: gym.Space,
        batch_size: int = 32,
        fill_value: Union[float, int, str] = 0.0,
        time_size: Optional[int] = None,
        time_major: bool = False,
) -> np.ndarray:
    """Returns batched dummy data (using `batch_size`) for the given `space`.

    Note: The returned batch will not pass a `space.contains(batch)` test
    as an additional batch dimension has to be added as dim=0.

    Args:
        space (gym.Space): The space to get a dummy batch for.
        batch_size(int): The required batch size (B). Note that this can also
            be 0 (only if `time_size` is None!), which will result in a
            non-batched sample for the given space (no batch dim).
        fill_value (Union[float, int, str]): The value to fill the batch with
            or "random" for random values.
        time_size (Optional[int]): If not None, add an optional time axis
            of `time_size` size to the returned batch.
        time_major (bool): If True AND `time_size` is not None, return batch
            as shape [T x B x ...], otherwise as [B x T x ...]. If `time_size`
            if None, ignore this setting and return [B x ...].

    Returns:
        The dummy batch of size `bqtch_size` matching the given space.
    """
    # Complex spaces. Perform recursive calls of this function.
    if isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple)):
        return tree.map_structure(
            lambda s: get_dummy_batch_for_space(s, batch_size, fill_value),
            get_base_struct_from_space(space),
        )
    # Primivite spaces: Box, Discrete, MultiDiscrete.
    # Random values: Use gym's sample() method.
    elif fill_value == "random":
        if time_size is not None:
            assert batch_size > 0 and time_size > 0
            if time_major:
                return np.array(
                    [[space.sample() for _ in range(batch_size)]
                     for t in range(time_size)],
                    dtype=space.dtype)
            else:
                return np.array(
                    [[space.sample() for t in range(time_size)]
                     for _ in range(batch_size)],
                    dtype=space.dtype)
        else:
            return np.array(
                [space.sample() for _ in range(batch_size)]
                if batch_size > 0 else space.sample(),
                dtype=space.dtype)
    # Fill value given: Use np.full.
    else:
        if time_size is not None:
            assert batch_size > 0 and time_size > 0
            if time_major:
                shape = [time_size, batch_size]
            else:
                shape = [batch_size, time_size]
        else:
            shape = [batch_size] if batch_size > 0 else []
        return np.full(
            shape + list(space.shape),
            fill_value=fill_value,
            dtype=space.dtype)


def clip_action(action, action_space):
    """Clips all components in `action` according to the given Space.

    Only applies to Box components within the action space.

    Args:
        action (Any): The action to be clipped. This could be any complex
            action, e.g. a dict or tuple.
        action_space (Any): The action space struct,
            e.g. `{"a": Distrete(2)}` for a space: Dict({"a": Discrete(2)}).

    Returns:
        Any: The input action, but clipped by value according to the space's
            bounds.
    """

    def map_(a, s):
        if isinstance(s, gym.spaces.Box):
            a = np.clip(a, s.low, s.high)
        return a

    return tree.map_structure(map_, action, action_space)


class ViewRequirement:
    """Single view requirement (for one column in an SampleBatch/input_dict).

    Policies and ModelV2s return a Dict[str, ViewRequirement] upon calling
    their `[train|inference]_view_requirements()` methods, where the str key
    represents the column name (C) under which the view is available in the
    input_dict/SampleBatch and ViewRequirement specifies the actual underlying
    column names (in the original data buffer), timestep shifts, and other
    options to build the view.

    Examples:
        >>> # The default ViewRequirement for a Model is:
        >>> req = [ModelV2].view_requirements
        >>> print(req)
        {"obs": ViewRequirement(shift=0)}
    """

    def __init__(self,
                 data_col: Optional[str] = None,
                 space: gym.Space = None,
                 shift: Union[int, str, List[int]] = 0,
                 index: Optional[int] = None,
                 batch_repeat_value: int = 1,
                 used_for_compute_actions: bool = True,
                 used_for_training: bool = True):
        """Initializes a ViewRequirement object.

        Args:
            data_col (Optional[str]): The data column name from the SampleBatch
                (str key). If None, use the dict key under which this
                ViewRequirement resides.
            space (gym.Space): The gym Space used in case we need to pad data
                in inaccessible areas of the trajectory (t<0 or t>H).
                Default: Simple box space, e.g. rewards.
            shift (Union[int, str, List[int]]): Single shift value or
                list of relative positions to use (relative to the underlying
                `data_col`).
                Example: For a view column "prev_actions", you can set
                `data_col="actions"` and `shift=-1`.
                Example: For a view column "obs" in an Atari framestacking
                fashion, you can set `data_col="obs"` and
                `shift=[-3, -2, -1, 0]`.
                Example: For the obs input to an attention net, you can specify
                a range via a str: `shift="-100:0"`, which will pass in
                the past 100 observations plus the current one.
            index (Optional[int]): An optional absolute position arg,
                used e.g. for the location of a requested inference dict within
                the trajectory. Negative values refer to counting from the end
                of a trajectory.
            used_for_compute_actions (bool): Whether the data will be used for
                creating input_dicts for `Policy.compute_actions()` calls (or
                `Policy.compute_actions_from_input_dict()`).
            used_for_training (bool): Whether the data will be used for
                training. If False, the column will not be copied into the
                final train batch.
        """
        self.data_col = data_col
        self.space = space if space is not None else gym.spaces.Box(
            float("-inf"), float("inf"), shape=())

        self.shift = shift
        if isinstance(self.shift, (list, tuple)):
            self.shift = np.array(self.shift)

        # Special case: Providing a (probably larger) range of indices, e.g.
        # "-100:0" (past 100 timesteps plus current one).
        self.shift_from = self.shift_to = None
        if isinstance(self.shift, str):
            f, t = self.shift.split(":")
            self.shift_from = int(f)
            self.shift_to = int(t)

        self.index = index
        self.batch_repeat_value = batch_repeat_value

        self.used_for_compute_actions = used_for_compute_actions
        self.used_for_training = used_for_training

def one_hot(x: TensorType, space: gym.Space) -> TensorType:
    """Returns a one-hot tensor, given and int tensor and a space.
    Handles the MultiDiscrete case as well.
    Args:
        x: The input tensor.
        space: The space to use for generating the one-hot tensor.
    Returns:
        The resulting one-hot tensor.
    Raises:
        ValueError: If the given space is not a discrete one.
    Examples:
        >>> x = torch.IntTensor([0, 3])  # batch-dim=2
        >>> # Discrete space with 4 (one-hot) slots per batch item.
        >>> s = gym.spaces.Discrete(4)
        >>> one_hot(x, s)
        tensor([[1, 0, 0, 0], [0, 0, 0, 1]])
        >>> x = torch.IntTensor([[0, 1, 2, 3]])  # batch-dim=1
        >>> # MultiDiscrete space with 5 + 4 + 4 + 7 = 20 (one-hot) slots
        >>> # per batch item.
        >>> s = gym.spaces.MultiDiscrete([5, 4, 4, 7])
        >>> one_hot(x, s)
        tensor([[1, 0, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1, 0, 0, 0]])
    """
    if isinstance(space, Discrete):
        return nn.functional.one_hot(x.long(), space.n)
    elif isinstance(space, MultiDiscrete):
        return torch.cat(
            [
                nn.functional.one_hot(x[:, i].long(), n)
                for i, n in enumerate(space.nvec)
            ],
            dim=-1,
        )
    else:
        raise ValueError("Unsupported space for `one_hot`: {}".format(space))

def get_base_struct_from_space(space):
    """Returns a Tuple/Dict Space as native (equally structured) py tuple/dict.

    Args:
        space (gym.Space): The Space to get the python struct for.

    Returns:
        Union[dict,tuple,gym.Space]: The struct equivalent to the given Space.
            Note that the returned struct still contains all original
            "primitive" Spaces (e.g. Box, Discrete).

    Examples:
        >>> get_base_struct_from_space(Dict({
        >>>     "a": Box(),
        >>>     "b": Tuple([Discrete(2), Discrete(3)])
        >>> }))
        >>> # Will return: dict(a=Box(), b=tuple(Discrete(2), Discrete(3)))
    """

    def _helper_struct(space_):
        if isinstance(space_, Tuple):
            return tuple(_helper_struct(s) for s in space_)
        elif isinstance(space_, Dict):
            return {k: _helper_struct(space_[k]) for k in space_.spaces}
        else:
            return space_

    return _helper_struct(space)

def normc_initializer(std: float = 1.0) -> Any:
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(
            tensor.data.pow(2).sum(1, keepdim=True))

    return initializer


class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 initializer: Any = None,
                 activation_fn: Any = None,
                 use_bias: bool = True,
                 bias_init: float = 0.0):
        """Creates a standard FC layer, similar to torch.nn.Linear

        Args:
            in_size(int): Input size for FC Layer
            out_size (int): Output size for FC Layer
            initializer (Any): Initializer function for FC layer weights
            activation_fn (Any): Activation function at the end of layer
            use_bias (bool): Whether to add bias weights or not
            bias_init (float): Initalize bias weights to bias_init const
        """
        super(SlimFC, self).__init__()
        layers = []
        # Actual nn.Linear layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer is None:
            initializer = nn.init.xavier_uniform_
        initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        # Activation function (if any; default=None (linear)).
        # if isinstance(activation_fn, str):
        #     activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)



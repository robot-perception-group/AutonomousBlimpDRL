import os
import pickle
import json
import numpy as np
from rl.rllib_script.agent.model.tx2_model import TorchBatchNormRNNModel
from rl.rllib_script.agent.model.action_dist import TorchDiagGaussian
from rl.rllib_script.agent.torch_policy import MyTorchPolicy, ppo_surrogate_loss
import inspect

checkpoint_path = os.path.expanduser(
    "~/catkin_ws/src/AutonomousBlimpDRL/RL/rl/trained_model/PPO_ResidualPlanarNavigateEnv_75b5a_00000_0_2022-02-21_17-15-18/checkpoint_001080/checkpoint-1080"
)


###########################################

dist_cls = TorchDiagGaussian

checkpoint_base_dir = os.path.dirname(checkpoint_path)
run_base_dir = os.path.dirname(os.path.dirname(checkpoint_path))

config_path = os.path.join(checkpoint_path)
with open(config_path, "rb") as f:
    obj = pickle.load(f)

config_path = os.path.join(run_base_dir, "params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)


########################################### save policy

worker = pickle.loads(obj["worker"])
print(worker)

weights = worker["state"]["default_policy"]["weights"]
observation_space = worker["policy_specs"]["default_policy"][1]
action_space = worker["policy_specs"]["default_policy"][2]


###tmp
# weights["lstm.weight_ih_l0"] = weights.pop("rnn.weight_ih_l0")
# weights["lstm.weight_hh_l0"] = weights.pop("rnn.weight_hh_l0")
# weights["lstm.bias_ih_l0"] = weights.pop("rnn.bias_ih_l0")
# weights["lstm.bias_hh_l0"] = weights.pop("rnn.bias_hh_l0")
###

to_pickle = {
    "weights": weights,
    "observation_space": observation_space,
    "action_space": action_space,
}
file_to_store = open(
    os.path.join(checkpoint_base_dir, "mypolicy.pickle"),
    "wb",
)
pickle.dump(to_pickle, file_to_store)
file_to_store.close()


file_to_read = open(os.path.join(checkpoint_base_dir, "mypolicy.pickle"), "rb")
loaded = pickle.load(file_to_read)

print(loaded)

weights = loaded["weights"]
observation_space = loaded["observation_space"]
action_space = loaded["action_space"]


########################################### save config
myconfig = {}
for k, v in config.items():
    if not inspect.isclass(v):
        # save the parameter if it is not a (ray) object
        myconfig[k] = config[k]


file_to_store = open(
    os.path.join(checkpoint_base_dir, "myconfig.pickle"),
    "wb",
)
pickle.dump(myconfig, file_to_store)
file_to_store.close()

file_to_read = open(os.path.join(checkpoint_base_dir, "myconfig.pickle"), "rb")
loaded = pickle.load(file_to_read)
print(loaded)

########################################### test


num_outputs = dist_cls.required_model_output_shape(action_space, config)
model_config = config["model"]
name = config["model"]["custom_model"]

model = TorchBatchNormRNNModel(
    obs_space=observation_space,
    action_space=action_space,
    num_outputs=num_outputs,
    model_config=model_config,
    name=name,
)


loss = ppo_surrogate_loss
action_distribution_class = dist_cls
action_sampler_fn = None
action_distribution_fn = None
max_seq_len = 20
get_batch_divisibility_req = None

policy = MyTorchPolicy(
    observation_space=observation_space,
    action_space=action_space,
    config=config,
    model=model,
    loss=loss,
    action_distribution_class=action_distribution_class,
    action_sampler_fn=action_sampler_fn,
    action_distribution_fn=action_distribution_fn,
    max_seq_len=max_seq_len,
    get_batch_divisibility_req=get_batch_divisibility_req,
)

policy.set_weights(weights)

cell_size = config["model"]["custom_model_config"].get("lstm_cell_size", 64)
state = [np.zeros(cell_size, np.float32), np.zeros(cell_size, np.float32)]
prev_action = np.zeros(4)
prev_reward = np.zeros(1)
obs = np.zeros(16)
action, state, _ = policy.compute_single_action(
    obs,
    state=state,
    prev_action=prev_action,
    prev_reward=prev_reward,
)

print(action)

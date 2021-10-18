import pickle
import os

default_exp_config = {
    "logdir": None,
    "timestep": 2e6,
    "callback": None,
    "final_model_save_path": None,
}

default_env_config = {}

default_tqc_agent_config = dict(
    learning_rate=0.0003,
    buffer_size=1000000,
    learning_starts=100,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    action_noise=None,
    replay_buffer_class=None,
    replay_buffer_kwargs=None,
    optimize_memory_usage=False,
    ent_coef="auto",
    target_update_interval=1,
    target_entropy="auto",
    top_quantiles_to_drop_per_net=2,
    use_sde=False,
    sde_sample_freq=-1,
    use_sde_at_warmup=False,
    tensorboard_log=None,
    create_eval_env=False,
    policy_kwargs=None,
    verbose=1,
    seed=None,
    device="auto",
    _init_setup_model=True,
)

default_qrdqn_agent_config = dict(
    learning_rate=5e-5,
    buffer_size=int(1e6),
    learning_starts=5e4,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1e4,
    exploration_fraction=5e-3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    max_grad_norm=None,
    verbose=1,
)

default_ppo_agent_config = dict(
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    sde_sample_freq=-1,
    target_kl=None,
    tensorboard_log=None,
    create_eval_env=False,
    policy_kwargs=None,
    verbose=1,
    seed=None,
)


def generate_config(
    agent_name: str = None,
    exp_config: dict = {},
    env_config: dict = {},
    agent_config: dict = {},
) -> dict:

    final_exp_config, final_env_config, final_agent_config = {}, {}, {}

    final_exp_config.update(default_exp_config)
    final_exp_config.update(exp_config)

    final_env_config.update(default_env_config)
    final_env_config.update(env_config)

    if agent_name == "QRDQN":
        final_agent_config.update(default_qrdqn_agent_config)
    elif agent_name == "PPO":
        final_agent_config.update(default_ppo_agent_config)
    elif agent_name == "TQC":
        final_agent_config.update(default_tqc_agent_config)
    final_agent_config.update(agent_config)

    meta_config: dict = {}
    meta_config.update(
        {
            "exp_config": final_exp_config,
            "env_config": final_env_config,
            "agent_config": final_agent_config,
        }
    )

    return meta_config


def save_config(path, config):
    filename = os.path.join(path, "config.pkl")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)


def load_config(path):
    filename = os.path.join(path, "config.pkl")
    with open(filename, "rb") as f:
        return pickle.load(f)

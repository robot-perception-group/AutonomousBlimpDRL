from stable_baselines3.common.policies import register_policy
from rl.agent.qrdqn_policy import MyMultiInputPolicy

register_policy("MyMultiInputPolicy", MyMultiInputPolicy)

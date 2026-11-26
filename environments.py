
from __future__ import annotations

import os
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

ENV_ID = "LunarLander-v3"
class EnvBuilder:
    def __init__(self):
        pass
    def make_env(self,space,env_idx,seed,gravity=-10.0,turbulence_power=1.5,enable_wind=False,wind_power=15.0,max_episode_steps=None,log_dir=None) -> gym.Env:
        if space=="continuous":
            continuous = True
        elif space=="discrete":
            continuous = False
        else:
            raise ValueError(f"Unknown space: {space}! Possible are continuous and discrete")
        kwargs = {
            "continuous": continuous,
            "gravity": gravity,
            "enable_wind": enable_wind,
            "wind_power": wind_power,
            "turbulence_power": turbulence_power,
        }
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = max_episode_steps
        env = gym.make(ENV_ID,**kwargs)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"env_{env_idx}")
            return Monitor(env, filename=log_path)
        return env



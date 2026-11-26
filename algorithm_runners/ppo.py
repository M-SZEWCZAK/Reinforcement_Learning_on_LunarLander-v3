from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environments import EnvBuilder


def run_ppo_scenario(space,seed,total_timesteps, n_eval_episodes,base_dir=None):
    root = base_dir if base_dir is not None else Path.cwd()
    train_log_dir = root / "logs" / f"init_ppo_{space}_seed{seed}" / "train"
    model_path = root / "models" / f"ppo_{space}_seed{seed}"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    builder = EnvBuilder()
    train_env = DummyVecEnv(
        [
            lambda: builder.make_env(
                space=space,
                env_idx=0,
                seed=seed,
                log_dir=str(train_log_dir),
            )
        ]
    )
    model = PPO("MlpPolicy", train_env, seed=seed, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(str(model_path))
    train_env.close()
    eval_env = builder.make_env(space=space, env_idx=0, seed=10_000 + seed, log_dir=None)
    returns = []

    try:
        for ep in range(n_eval_episodes):
            obs, info = eval_env.reset(seed=10_000 + seed + ep)
            ep_return = 0.0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_return += float(reward)
                done = terminated or truncated

            returns.append(ep_return)
    finally:
        eval_env.close()

    returns_arr = np.array(returns)
    print(
        f"[PPO | {space} | seed={seed} | steps={total_timesteps}] "
        f"Mean: {returns_arr.mean():.2f} | Std: {returns_arr.std():.2f} | "
        f"Min: {returns_arr.min():.2f} | Max: {returns_arr.max():.2f}"
    )
    return returns_arr



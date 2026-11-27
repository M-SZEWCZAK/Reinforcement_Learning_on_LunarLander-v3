from pathlib import Path
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from environments import EnvBuilder


DEFAULT_DQN_KWARGS = dict(
    learning_rate=6.3e-4,
    batch_size=128,
    buffer_size=50_000,
    learning_starts=0,
    gamma=0.99,
    target_update_interval=250,
    train_freq=4,
    gradient_steps=-1,
    exploration_fraction=0.12,
    exploration_final_eps=0.1,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def run_dqn_scenario(seed, total_timesteps, n_eval_episodes, base_dir=None, space="discrete", verbose=1, **dqn_kwargs):
    if space != "discrete":
        raise ValueError(f"DQN supports only the discrete action space")

    root = base_dir if base_dir is not None else Path.cwd()
    root = Path(root)
    train_log_dir = root / "logs" / f"init_dqn_{space}_seed{seed}" / "train"
    model_path = root / "models" / f"dqn_{space}_seed{seed}"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    hyperparams = {**DEFAULT_DQN_KWARGS, **dqn_kwargs}

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
    model = DQN("MlpPolicy", train_env, seed=seed, verbose=verbose, **hyperparams)
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
                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                ep_return += float(reward)
                done = terminated or truncated

            returns.append(ep_return)
    finally:
        eval_env.close()

    returns_arr = np.array(returns)
    print(
        f"[DQN | {space} | seed={seed} | steps={total_timesteps}] "
        f"Mean: {returns_arr.mean():.2f} | Std: {returns_arr.std():.2f} | "
        f"Min: {returns_arr.min():.2f} | Max: {returns_arr.max():.2f}"
    )
    return returns_arr

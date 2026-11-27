import time
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environments import EnvBuilder


def evaluate_policy_on_env(model, env, seed, n_episodes=20):
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    return np.array(returns)


def run_sac_scenario(
    space="continuous",
    seed=0,
    total_timesteps=50_000,
    n_eval_episodes=20,
    eval_freq=10_000,
    base_dir=None,
    **sac_kwargs,
):
    if space != "continuous":
        raise ValueError("SAC supports only the 'continuous' action space.")

    root = base_dir if base_dir is not None else Path.cwd()
    run_dir = root / "logs" / f"init_sac_{space}_seed{seed}"
    train_log_dir = run_dir / "train"
    eval_log_dir = run_dir / "eval"
    model_path = root / "models" / f"sac_{space}_seed{seed}"
    best_model_dir = root / "models" / f"sac_{space}_seed{seed}_best"
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
    eval_cb_env = DummyVecEnv(
        [
            lambda: Monitor(
                builder.make_env(
                    space=space,
                    env_idx=0,
                    seed=20_000 + seed,
                    log_dir=None,
                )
            )
        ]
    )
    eval_callback = EvalCallback(
        eval_cb_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1,
    )

    t0 = time.perf_counter()
    model = SAC("MlpPolicy", train_env, seed=seed, verbose=1, **sac_kwargs)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    elapsed_sec = time.perf_counter() - t0

    model.save(str(model_path))
    train_env.close()
    eval_cb_env.close()

    eval_env = builder.make_env(
        space=space, env_idx=0, seed=10_000 + seed, log_dir=None
    )
    try:
        returns_arr = evaluate_policy_on_env(
            model, eval_env, seed=10_000 + seed, n_episodes=n_eval_episodes
        )
    finally:
        eval_env.close()

    success_rate = float(np.mean(returns_arr >= 200.0) * 100.0)
    print(
        f"[SAC | {space} | seed={seed} | steps={total_timesteps} | time={elapsed_sec:.1f}s] "
        f"Mean: {returns_arr.mean():.2f} | Std: {returns_arr.std():.2f} | "
        f"Min: {returns_arr.min():.2f} | Max: {returns_arr.max():.2f} | "
        f"Success(>=200): {success_rate:.1f}%"
    )
    return returns_arr


def evaluate_wind_robustness(
    model_path,
    seed=0,
    n_eval_episodes=20,
    wind_power=15.0,
    turbulence_power=1.5,
):
    model = SAC.load(str(model_path))
    builder = EnvBuilder()
    results = {}

    for label, wind_flag in [("no_wind", False), ("with_wind", True)]:
        env = builder.make_env(
            space="continuous",
            env_idx=0,
            seed=30_000 + seed,
            enable_wind=wind_flag,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
            log_dir=None,
        )
        try:
            returns = evaluate_policy_on_env(
                model, env, seed=30_000 + seed, n_episodes=n_eval_episodes
            )
        finally:
            env.close()

        success_rate = float(np.mean(returns >= 200.0) * 100.0)
        results[label] = returns
        print(
            f"[SAC Robustness | {label}] Mean: {returns.mean():.2f} | "
            f"Std: {returns.std():.2f} | Success(>=200): {success_rate:.1f}%"
        )

    return results

#all tests written using gemini default model in free tier

import tempfile
import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.monitor import load_results

from environments import EnvBuilder


def test_spaces_discrete_and_continuous():
    builder = EnvBuilder()

    env_d = builder.make_env(space="discrete", env_idx=0, seed=42)
    obs_d, _ = env_d.reset(seed=42)
    assert obs_d.shape == (8,)
    assert isinstance(env_d.action_space, gym.spaces.Discrete)
    assert env_d.action_space.n == 4
    env_d.close()

    env_c = builder.make_env(space="continuous", env_idx=0, seed=42)
    obs_c, _ = env_c.reset(seed=42)
    assert obs_c.shape == (8,)
    assert isinstance(env_c.action_space, gym.spaces.Box)
    assert env_c.action_space.shape == (2,)
    env_c.close()


def test_deterministic_reset_and_rollout():
    builder = EnvBuilder()

    env1 = builder.make_env(space="discrete", env_idx=0, seed=123)
    env2 = builder.make_env(space="discrete", env_idx=0, seed=123)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    assert np.allclose(obs1, obs2)

    for step in range(20):
        a1 = env1.action_space.sample()
        a2 = env2.action_space.sample()
        assert a1 == a2
        o1, r1, term1, trunc1, _ = env1.step(a1)
        o2, r2, term2, trunc2, _ = env2.step(a2)
        assert np.allclose(o1, o2)
        assert r1 == pytest.approx(r2)
        assert term1 == term2 and trunc1 == trunc2
        if term1 or trunc1:
            break

    env1.close()
    env2.close()


def test_wind_and_monitor_logging():
    builder = EnvBuilder()
    with tempfile.TemporaryDirectory() as tmpdir:
        env = builder.make_env(
            space="continuous",
            env_idx=0,
            seed=7,
            gravity=-10.0,
            enable_wind=True,
            wind_power=15.0,
            turbulence_power=1.5,
            max_episode_steps=50,
            log_dir=tmpdir,
        )
        obs, _ = env.reset(seed=7)
        assert obs.shape == (8,)

        done = False
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.close()
        df = load_results(tmpdir)
        assert len(df) >= 1
        assert {"r", "l", "t"}.issubset(df.columns)


def test_invalid_space_raises():
    builder = EnvBuilder()
    with pytest.raises(ValueError):
        builder.make_env(space="invalid_space", env_idx=0, seed=0)


if __name__ == "__main__":
    test_spaces_discrete_and_continuous()
    test_deterministic_reset_and_rollout()
    test_wind_and_monitor_logging()
    test_invalid_space_raises()
    print("All EnvBuilder tests passed!")
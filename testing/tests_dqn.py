# Smoke tests for the DQN runner. Run from the repo root:
#   python -m pytest testing/tests_dqn.py -v

import numpy as np
import pytest
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import load_results

from algorithm_runners.dqn import DEFAULT_DQN_KWARGS, run_dqn_scenario

TINY_STEPS = 500


def test_runner_trains_saves_and_evaluates(tmp_path):
    returns = run_dqn_scenario(
        seed=0, total_timesteps=TINY_STEPS, n_eval_episodes=2, base_dir=tmp_path, verbose=0
    )

    assert isinstance(returns, np.ndarray)
    assert returns.shape == (2,)
    assert np.all(np.isfinite(returns))

    assert (tmp_path / "models" / "dqn_discrete_seed0.zip").exists()
    df = load_results(str(tmp_path / "logs" / "init_dqn_discrete_seed0" / "train"))
    assert len(df) >= 1
    assert {"r", "l", "t"}.issubset(df.columns)


def test_continuous_space_raises(tmp_path):
    with pytest.raises(ValueError):
        run_dqn_scenario(
            seed=0, total_timesteps=10, n_eval_episodes=1, base_dir=tmp_path, space="continuous", verbose=0
        )


def test_same_seed_is_reproducible(tmp_path):
    kwargs = dict(seed=1, total_timesteps=TINY_STEPS, n_eval_episodes=2, verbose=0)
    r1 = run_dqn_scenario(base_dir=tmp_path / "run1", **kwargs)
    r2 = run_dqn_scenario(base_dir=tmp_path / "run2", **kwargs)
    np.testing.assert_allclose(r1, r2)


def test_kwargs_override_defaults(tmp_path):
    run_dqn_scenario(
        seed=0,
        total_timesteps=TINY_STEPS,
        n_eval_episodes=1,
        base_dir=tmp_path,
        verbose=0,
        buffer_size=1_000,
        target_update_interval=100,
    )
    model = DQN.load(str(tmp_path / "models" / "dqn_discrete_seed0"))

    assert model.buffer_size == 1_000
    assert model.target_update_interval == 100
    # Settings not overridden keep the tuned defaults.
    assert model.batch_size == DEFAULT_DQN_KWARGS["batch_size"]
    assert model.exploration_final_eps == DEFAULT_DQN_KWARGS["exploration_final_eps"]

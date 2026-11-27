import tempfile
from pathlib import Path
import pytest

from algorithm_runners.sac import run_sac_scenario


def test_sac_smoke_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        returns = run_sac_scenario(
            space="continuous",
            seed=0,
            total_timesteps=200,
            n_eval_episodes=2,
            base_dir=base_dir,
            learning_starts=10,
        )
        assert len(returns) == 2
        assert (base_dir / "models" / "sac_continuous_seed0.zip").exists()


def test_sac_rejects_discrete_space():
    with pytest.raises(ValueError):
        run_sac_scenario(space="discrete", seed=0, total_timesteps=100, n_eval_episodes=1)

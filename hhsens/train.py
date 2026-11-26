import gc

import jax
from arlbench.autorl import AutoRLEnv


class PPOObjective:
    def __init__(self, env, n_eval_episodes):
        jax.config.update("jax_enable_x64", env.get("jax_enable_x64", False))
        self.env = env
        self.n_eval_episodes = n_eval_episodes

    def __call__(self, config, n_steps, seed):
        e = self.env
        autorl = AutoRLEnv({
            "seed": seed,  # AutoRLEnv.step(seed=0) would fall back to this, so set it here
            "env_framework": e["framework"],
            "env_name": e["name"],
            "env_kwargs": e.get("env_kwargs", {}),
            "eval_env_kwargs": e.get("env_kwargs", {}),
            "n_envs": e["n_envs"],
            "algorithm": "ppo",
            "nas_config": e["network"],
            "n_total_timesteps": n_steps,
            "n_eval_steps": 1,
            "n_eval_episodes": self.n_eval_episodes,
        })
        autorl.reset()
        _, objectives, _, _, _ = autorl.step({**e["ppo"], **config}, n_total_timesteps=n_steps)
        jax.clear_caches()
        gc.collect()
        return float(objectives["reward_mean"])

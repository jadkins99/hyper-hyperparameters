import argparse
from pathlib import Path

from hhsens.config import load_env, load_yaml, ppo_space
from hhsens.hyperhyper import sample_hyperhyper
from hhsens.train import PPOObjective
from hhsens.tuner import tune

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True)
parser.add_argument("--config-id", type=int, required=True)
parser.add_argument("--smac-seed", type=int, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--budget-multiplier", type=float)
args = parser.parse_args()
if not 1 <= args.config_id <= 64:
    parser.error("--config-id must be between 1 and 64")

hh = sample_hyperhyper()[args.config_id - 1]
protocol = load_yaml("tuning.yaml")
env = load_env(args.env)
incumbent, performance = tune(
    PPOObjective(env, protocol["n_eval_episodes"]),
    ppo_space(env["family"], seed=args.smac_seed),
    hh,
    max_budget=env["steps"],
    smac_seed=args.smac_seed,
    out_dir=args.out,
    divergence_score=env["random_policy_return"],
    divergence_threshold=protocol["divergence_threshold"],
    budget_multiplier=args.budget_multiplier or protocol["budget_multiplier"],
    smac_cfg=protocol["smac"],
    run_info={"env": args.env, "config_id": args.config_id},
)
print(performance, incumbent)

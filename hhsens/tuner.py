import csv
import json
import math

import numpy as np
from smac.runhistory.dataclasses import TrialValue

from .smac_factory import build_smac


def _plain(config):
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in dict(config).items()}


def tune(objective, configspace, hh, max_budget, smac_seed, out_dir, divergence_score,
         divergence_threshold, budget_multiplier, smac_cfg, run_info=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    smac = build_smac(hh, configspace, max_budget, smac_seed, out_dir, smac_cfg)

    seeds = range(hh["num_seeds"])
    names = list(configspace.keys())
    total = budget_multiplier * max_budget
    spent = 0
    trials = []

    with open(out_dir / "trials.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "budget", "cum_steps", "performance", *(f"return_{s}" for s in seeds), *names])
        while spent < total:
            info = smac.ask()
            budget = int(round(info.budget))
            config = _plain(info.config)
            returns = [objective(config, budget, s) for s in seeds]
            scores = [r if math.isfinite(r) and r >= divergence_threshold else divergence_score for r in returns]
            performance = float(np.mean(scores))
            smac.tell(info, TrialValue(cost=-performance))
            spent += budget * len(seeds)
            trials.append((config, budget, performance))
            writer.writerow([len(trials) - 1, budget, spent, performance, *returns, *(config[n] for n in names)])
            f.flush()

    incumbent = _plain(smac.intensifier.get_incumbent())
    budget, performance = max((b, p) for c, b, p in trials if c == incumbent)
    with open(out_dir / "run.json", "w") as f:
        json.dump({
            **(run_info or {}),
            "hyperhyper": hh,
            "smac_seed": smac_seed,
            "spent_steps": spent,
            "n_trials": len(trials),
            "incumbent": incumbent,
            "incumbent_budget": budget,
            "incumbent_performance": performance,
        }, f, indent=2)
    return incumbent, performance

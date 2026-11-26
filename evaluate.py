import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from hhsens.config import load_env, load_yaml

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)
sub.add_parser("run").add_argument("run_dir", type=Path)
aggregate = sub.add_parser("aggregate")
aggregate.add_argument("runs_root", type=Path)
aggregate.add_argument("--out", type=Path, default=Path("results"))
args = parser.parse_args()

if args.command == "run":
    from hhsens.train import PPOObjective

    protocol = load_yaml("tuning.yaml")
    run = json.loads((args.run_dir / "run.json").read_text())
    env = load_env(run["env"])
    objective = PPOObjective(env, protocol["n_eval_episodes"])
    returns = [objective(run["incumbent"], env["steps"], seed) for seed in protocol["test_seeds"]]
    returns = [r if math.isfinite(r) else -math.inf for r in returns]
    j_tilde = sum(returns) / len(returns)
    (args.run_dir / "eval.json").write_text(json.dumps({
        "env": run["env"],
        "config_id": run["config_id"],
        "smac_seed": run["smac_seed"],
        "test_seeds": protocol["test_seeds"],
        "returns": returns,
        "j_tilde": j_tilde,
    }, indent=2))
    print(j_tilde)

else:
    rows = []
    for path in sorted(args.runs_root.glob("**/eval.json")):
        e = json.loads(path.read_text())
        rows.append((e["env"], e["config_id"], e["smac_seed"], e["j_tilde"]))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "j_tilde.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env", "config_id", "smac_seed", "j_tilde"])
        writer.writerows(sorted(rows))
    groups = defaultdict(list)
    for env, config_id, _, j_tilde in rows:
        groups[(env, config_id)].append(j_tilde)
    with open(args.out / "j_hat.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env", "config_id", "n_smac_seeds", "j_hat"])
        for (env, config_id), values in sorted(groups.items()):
            writer.writerow([env, config_id, len(values), sum(values) / len(values)])

from pathlib import Path

import yaml
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace, UniformFloatHyperparameter

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load_yaml(name):
    with open(CONFIGS / name) as f:
        return yaml.safe_load(f)


def load_env(key):
    cfg = load_yaml("environments.yaml")
    env = dict(cfg["environments"][key])
    env["ppo"] = {**cfg["ppo_defaults"], **env.get("ppo", {})}
    env["network"] = {**cfg["network_defaults"], **env.get("network", {})}
    return env


def ppo_space(family, seed=None):
    cs = ConfigurationSpace(seed=seed)
    for name, s in load_yaml("ppo_search_space.yaml")["families"][family].items():
        if s["type"] == "float":
            cs.add(UniformFloatHyperparameter(name, s["lower"], s["upper"], log=s.get("log", False),
                                              default_value=s.get("default")))
        else:
            cs.add(CategoricalHyperparameter(name, s["choices"]))
    return cs

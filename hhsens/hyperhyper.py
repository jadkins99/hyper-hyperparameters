from ConfigSpace import (
    ConfigurationSpace,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


def hyperhyper_space(seed=0):
    cs = ConfigurationSpace(seed=seed)
    cs.add([
        OrdinalHyperparameter("num_seeds", [1, 3, 5, 7, 9]),
        OrdinalHyperparameter("min_budget_denom", [5, 10, 20, 50, 100]),
        UniformIntegerHyperparameter("eta", 2, 6),
        OrdinalHyperparameter("max_depth", [5, 10, 15, 20, 30]),
        UniformFloatHyperparameter("xi", 0.0, 0.1),
        OrdinalHyperparameter("ratio_features", [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]),
        UniformIntegerHyperparameter("min_samples_split", 1, 5),
    ])
    return cs


def sample_hyperhyper(n=64, seed=0):
    draws = hyperhyper_space(seed).sample_configuration(size=n)
    draws = draws if isinstance(draws, list) else [draws]
    return [{k: v.item() if hasattr(v, "item") else v for k, v in dict(d).items()} for d in draws]

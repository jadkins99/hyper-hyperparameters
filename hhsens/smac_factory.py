import math

from smac import Scenario
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as MF


def _unused(config, seed=0, budget=0.0):
    raise RuntimeError("SMAC is driven through ask/tell")


def build_smac(hh, configspace, max_budget, seed, output_dir, smac_cfg):
    scenario = Scenario(
        configspace,
        name="smac",
        output_directory=output_dir,
        deterministic=True,
        n_trials=smac_cfg["n_trials"],
        min_budget=max_budget / hh["min_budget_denom"],
        max_budget=max_budget,
        seed=seed,
        n_workers=1,
    )
    smac = MF(
        scenario,
        _unused,
        model=MF.get_model(
            scenario,
            n_trees=smac_cfg["n_trees"],
            max_depth=hh["max_depth"],
            ratio_features=hh["ratio_features"],
            min_samples_split=hh["min_samples_split"],
        ),
        acquisition_function=MF.get_acquisition_function(scenario, xi=hh["xi"]),
        acquisition_maximizer=MF.get_acquisition_maximizer(
            scenario,
            challengers=smac_cfg["challengers"],
            local_search_iterations=smac_cfg["local_search_iterations"],
        ),
        intensifier=MF.get_intensifier(scenario, eta=hh["eta"]),
        initial_design=MF.get_initial_design(
            scenario,
            n_configs_per_hyperparamter=smac_cfg["initial_design_per_hyperparameter"],  # sic
        ),
        random_design=MF.get_random_design(scenario, probability=smac_cfg["random_design_probability"]),
        overwrite=True,
        logging_level=40,
    )

    meta = smac.meta
    applied = {
        "max_depth": meta["model"]["max_depth"],
        "ratio_features": meta["model"]["ratio_features"],
        "min_samples_split": meta["model"]["min_samples_split"],
        "xi": meta["acquisition_function"]["xi"],
        "eta": meta["intensifier"]["eta"],
        "min_budget_denom": max_budget / smac.scenario.min_budget,
    }
    for key, value in applied.items():
        if not math.isclose(value, hh[key], rel_tol=1e-9):
            raise ValueError(f"SMAC has {key}={value}, requested {hh[key]}")
    return smac

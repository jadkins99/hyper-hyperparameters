# Tuning the Tuner: The Hyper-Hyperparameter Sensitivity of AutoRL

Code for running hyper-hyperparameter sensitivity experiments like those described in Tuning the Tuner: The Hyper-Hyperparameter Sensitivity of AutoRL (NeurIPS 2026).
## Install


```bash
pip install swig setuptools wheel numpy==1.24.4
pip install --no-build-isolation pyrfr==0.9.0
pip install -e .
pip install --no-deps "arlbench @ git+https://github.com/automl/arlbench@a2ad9cbcbecdb93f80f81f464ddb168a11604ff3"
```

For GPUs, install the CUDA build of `jax`/`jaxlib` 0.4.28.

## Run

```bash
python tune.py --env cc_cartpole --config-id 39 --smac-seed 0 --out runs/cc_cartpole/39/0
```

`--config-id` picks one of the 64 hyper-hyperparameter settings drawn by `hhsens/hyperhyper.py`. Other settings are in `configs/`. The run writes `trials.csv` and `run.json` to `--out`.

## Evaluate

```bash
python evaluate.py run runs/cc_cartpole/39/0
python evaluate.py aggregate runs
```


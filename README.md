<br />
<div align="center">

  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776ab.svg?logo=python&logoColor=white" alt="Python 3.10+"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-ee4c2c.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
    <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="License: CC BY 4.0"></a>
  </p>

</div>

## Project Structure 📂

```text
.
├── configs/
│   ├── dgp_config.yaml           # simulated processes and their parameters
│   ├── model_constants.yaml      # search spaces and fixed hyperparameters
│   └── run_config.yaml           # split proportions, series length, seeds
├── data/
│   └── brent.parquet             # cached FRED series for the application
├── src/
│   ├── calibration/              # local PIT recalibration (I-splines + XGBoost)
│   ├── forecast_methods/         # MDN, FlexZBoost, KCDE, Flow, Conformal
│   ├── markov_switching_bubble/  # bubble DGP with exact predictive densities
│   ├── metrics/                  # density, point, quantile and region metrics
│   ├── optuna/                   # search spaces and study runner
│   ├── results/                  # artifact paths, I/O and multi-seed reporting
│   ├── stable_mar/               # alpha-stable MAR simulation and estimation
│   ├── theoretical/              # closed-form densities and quantiles
│   └── utils/                    # device, logging, config constants
├── main_simulations.py           # Monte Carlo study on the simulated processes
├── main_applications.py          # the same pipeline on the real Brent series
├── main_bubble.py                # bubble continuation-vs-collapse analysis
├── LICENSE, requirements.txt, ruff.toml, .pre-commit-config.yaml, .gitignore
└── README.md
```

## Usage 🚀

```bash
git clone https://github.com/JulienPeignon/explosive-bubble-forecasting.git
cd explosive-bubble-forecasting/
pip install -r requirements.txt
```

```bash
# Tune and evaluate the headline MDN (Skew-t components, both weighting
# schemes) on a MAR(0,1) process at horizon 1, with 100 Optuna trials
python main_simulations.py --model mdn --density skewt --weighting both \
    --process mar01 --horizon 1 --n_trials 100

# Reuse the saved configuration and re-score on the test set
python main_simulations.py --model mdn --density skewt --weighting both \
    --process mar01 --horizon 1 --evaluate

# A baseline instead of the MDN: flexzboost | kcde | flow | conformal
python main_simulations.py --model flexzboost --process mar01 --horizon 1 \
    --n_trials 100

# The real-data application on Brent
python main_applications.py --data brent --model mdn --horizon 1 --n_trials 100

# The bubble continuation study
python main_bubble.py --horizon 5
```

`--evaluate` reloads a tuned configuration instead of retuning. Run `python <script>.py --help` for the rest.

## License 🔒

This work is shared under the Creative Commons Attribution 4.0 International License. Refer to `LICENSE` for more details.

# MITEDA

Research code for **Mutual information-assisted estimation of distribution algorithm for groundwater contamination monitoring well optimization**.

**Authors:** Zhengchen Zhou, Xuesong Yan, Wenqi Xie  
**Publication:** Expert Systems with Applications, 333 (2027 issue), 133988  
**Paper:** [10.1016/j.eswa.2026.133988](https://doi.org/10.1016/j.eswa.2026.133988)

Design groundwater monitoring networks by optimizing well positions and binary activation decisions. A maximum mutual-information tree models dependencies among decision variables, Hungarian matching aligns well slots, and rank-and-crowding survival maintains the population.

## Model and configuration

`problem` specifies the search area, observation times, source scenarios, hydrogeological scenarios, well count, spacing and installation costs. `optimizer` sets population size, elite size, generations and independent runs.

The forward model uses `adepy.uniform.point2` for a continuous point source in a uniform infinite aquifer. All coordinates and transport parameters must use consistent units. Forward-model failures raise an error with the source, observation location and time.

Both objectives are minimized: `F[0]` is negative worst-case source distinguishability; `F[1]` is installation cost. Distinguishability is the minimum pairwise distance of source-scenario observation vectors, averaging squared distances over hydrogeological scenarios. Mutual information is used in the EDA search model. Constraints enforce minimum active-well count and spacing. Only feasible nondominated layouts are exported.

## Outputs

Each independent run has a `run-NNN` directory:

- `layouts.json`: active/inactive well-slot decisions.
- `results.npz`: objective matrix `F` and constraint matrix `G`.
- `history.json`: all generations' decisions, objectives and constraints.
- `pareto.png`: source distinguishability versus installation cost.

The root `summary.json` records each run's status. Newly computed results are exported directly. To decode active wells, call `problem.get_well_locations(layout)`.

## Installation

Use Python 3.10–3.12 in a dedicated environment:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install -e ".[dev]"
```

## Run

From the project root:

```bash
python -m miteda --config configs/demo.json
python -m miteda --config configs/experiment.json
```

`demo.json` provides a small execution example; `experiment.json` provides a larger configuration. An explicit output path can be supplied with `--output runs/example`. The destination must not already exist. Without that argument, a new timestamped directory is created under `runs/`.

Every run saves `config.json` and `environment.json` alongside results. Sampling follows the algorithm's natural randomness; no fixed random seed is configured. The demo validates execution, not publication-level accuracy or convergence.

## Project structure

```text
configs/             Small and larger experiment configurations
src/miteda/       Importable Python implementation and command-line entry point
notebooks/demo.ipynb  Lightweight example calling the package
tests/               Numerical and workflow checks
.github/workflows/   Automated tests and demo execution
```

## Tests

```bash
python -m pytest -q
```

## Citation

Please cite the linked paper when using this code. Machine-readable citation metadata is provided in `CITATION.cff`.

## Contact

Zhengchen Zhou (Zhouchen), China University of Geosciences (Wuhan)  
Email: [chenzz_14@outlook.com](mailto:chenzz_14@outlook.com)  
ORCID: [0009-0005-0160-1988](https://orcid.org/0009-0005-0160-1988)

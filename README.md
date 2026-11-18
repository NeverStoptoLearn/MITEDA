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

## Third-party software and acknowledgments

This implementation builds on the following open-source software. We thank their authors and contributors.

| Software | Version | Use in this repository |
| --- | --- | --- |
| [AdePy](https://github.com/cneyens/adepy) | 0.2.0 | The `adepy.uniform.point2` analytical concentration solution used by `transport.py` |
| [pymoo](https://pymoo.org/) | 0.6.1.3 | Problem and population interfaces, constrained rank-and-crowding survival, and nondominated sorting |
| [scikit-learn](https://scikit-learn.org/) | 1.5.2 | Mutual-information estimation and conditional regression models |
| [NumPy](https://numpy.org/) | 1.26.4 | Numerical arrays and sampling |
| [SciPy](https://scipy.org/) | 1.13.1 | Hungarian matching and pairwise distances |
| [Matplotlib](https://matplotlib.org/) | 3.9.4 | Result visualization |

AdePy is distributed under the [MIT License](https://github.com/cneyens/adepy/blob/main/LICENSE); pymoo uses the [Apache License 2.0](https://github.com/anyoptimization/pymoo/blob/0.6.1.3/LICENSE). Dependencies retain their upstream licenses.

The software and framework references below supplement the research-paper citation in the Citation section:

```bibtex
@software{adepy,
  author  = {Neyens, Cas},
  title   = {{AdePy}: Analytical solutions for solute transport in groundwater with Python},
  version = {0.2.0},
  url     = {https://github.com/cneyens/adepy}
}

@article{blank2020pymoo,
  author  = {Blank, Julian and Deb, Kalyanmoy},
  title   = {pymoo: Multi-Objective Optimization in Python},
  journal = {IEEE Access},
  volume  = {8},
  pages   = {89497--89509},
  year    = {2020},
  doi     = {10.1109/ACCESS.2020.2990567}
}
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

If you use this code in your research, please cite:

```bibtex
@article{zhou2027miteda,
  title   = {Mutual information-assisted estimation of distribution algorithm for groundwater contamination monitoring well optimization},
  author  = {Zhou, Zhengchen and Yan, Xuesong and Xie, Wenqi},
  journal = {Expert Systems with Applications},
  volume  = {333},
  pages   = {133988},
  year    = {2027},
  doi     = {10.1016/j.eswa.2026.133988}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Zhengchen Zhou (Zhouchen), China University of Geosciences (Wuhan)  
Email: [chenzz_14@outlook.com](mailto:chenzz_14@outlook.com)  
ORCID: [0009-0005-0160-1988](https://orcid.org/0009-0005-0160-1988)

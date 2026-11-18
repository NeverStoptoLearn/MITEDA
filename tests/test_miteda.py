import json
from pathlib import Path
import numpy as np
import pytest
from miteda.encoding import canonicalize_mixed, encode_mixed_solution, decode_mixed_solution
from miteda.model import EDAParams, MixedWellLayoutModel, estimate_mi
from miteda.cli import make_problem
from miteda.optimizer import mo_eda_optimize_mixed
from miteda.transport import calculate_concentration


def settings():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / 'configs/demo.json').read_text())['problem']


def test_encoding_canonical_permutation_invariance():
    wells = [(20, 30, 1), (50, 40, 0), (10, 20, 1)]
    x = encode_mixed_solution(wells, 3)
    assert decode_mixed_solution(x, 3) == wells
    reverse = encode_mixed_solution(wells[::-1], 3)
    assert canonicalize_mixed(x, 3) == canonicalize_mixed(reverse, 3)


def test_transport_matches_adepy():
    from adepy.uniform import point2
    value = calculate_concentration(1000, 60, 50, 30, 1, 0.25, 30, 6, 80, 100, 20, 20)
    expected = point2(1000, 60, 50, 30, 1, 0.25, 30, 6, 0.8, 20, 20)
    assert value == pytest.approx(float(np.asarray(expected).item()))


def test_forward_errors_are_propagated():
    p = make_problem(settings())
    def fail(**kwargs):
        raise ValueError('bad transport')
    p.forward_model = fail
    x = encode_mixed_solution([(10, 10, 1), (40, 40, 1), (70, 70, 0), (90, 90, 0)], 4)
    with pytest.raises(RuntimeError, match='Forward simulation failed'):
        p._evaluate(x, {})


def test_mi_handles_collapsed_and_singleton_classes():
    assert estimate_mi(np.ones(5), np.arange(5), True, True) == 0
    assert estimate_mi(np.arange(5), np.array([0, 0, 0, 0, 1]), True, False) == 0


def test_optimizer_returns_only_feasible_solutions_and_history():
    p = make_problem(settings())
    pareto, history = mo_eda_optimize_mixed(p, pop_size=8, n_gen=2,
        eda_params=EDAParams(mu_size=8), history=True, verbose=False)
    assert len(history) == 2
    assert len(pareto) > 0
    assert np.isfinite(pareto.get('F')).all()
    assert np.all(pareto.get('G') <= 1e-8)
    for x in pareto.get('X'):
        assert all(0 <= x[f'x{i}'] <= 100 and 0 <= x[f'y{i}'] <= 100 and x[f'b{i}'] in (0, 1) for i in range(p.n_wells))


def test_no_feasible_front_is_not_reported_as_success():
    p = make_problem({**settings(), 'min_distance': 1000})
    pareto = mo_eda_optimize_mixed(p, pop_size=8, n_gen=1, eda_params=EDAParams(mu_size=8), verbose=False)
    assert len(pareto) == 0

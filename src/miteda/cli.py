import argparse
import json
import numpy as np
from .problem import GroundwaterMonitoringMOProblem
from .transport import calculate_concentration
from .model import EDAParams
from .optimizer import mo_eda_optimize_mixed
from .experiment_io import start_run, save_json


def make_problem(config):
    settings = dict(config)
    settings['search_area'] = tuple(tuple(a) for a in settings['search_area'])
    if settings['n_wells'] < settings['min_wells'] or settings['min_wells'] < 1:
        raise ValueError('1 <= min_wells <= n_wells is required')
    if settings['min_distance'] < 0 or settings['cost_per_well'] < 0:
        raise ValueError('Spacing and cost must be nonnegative')
    if any(len(a) != 2 or a[0] >= a[1] or any(int(v) != v for v in a) for a in settings['search_area']):
        raise ValueError('Search bounds must be increasing integer pairs')
    if not settings['observation_times'] or min(settings['observation_times']) <= 0:
        raise ValueError('Positive observation times are required')
    return GroundwaterMonitoringMOProblem(forward_model=calculate_concentration, **settings)


def run(config, output=None):
    problem = make_problem(config['problem'])
    settings = config['optimizer']
    population, generations, repeats = settings['population'], settings['generations'], settings['runs']
    if population < 4 or generations < 1 or repeats < 1 or not 4 <= settings['elite_size'] <= population:
        raise ValueError('Invalid optimizer sizes: population/elite >= 4, generations/runs >= 1')
    path = start_run(config, output)
    summary = []
    for index in range(repeats):
        pareto, history = mo_eda_optimize_mixed(problem, pop_size=population, n_gen=generations,
            eda_params=EDAParams(mu_size=settings['elite_size'], min_std=settings.get('min_std', 0.1)),
            verbose=False, history=True)
        folder = path / f'run-{index:03d}'
        folder.mkdir()
        X = [dict(x) for x in pareto.get('X')]
        F = np.asarray(pareto.get('F'), dtype=float).reshape(-1, 2)
        G = np.asarray(pareto.get('G'), dtype=float).reshape(-1, 2)
        decisions = [{key: int(value) for key, value in x.items()} for x in X]
        save_json(folder / 'layouts.json', decisions)
        np.savez_compressed(folder / 'results.npz', F=F, G=G)
        for generation in history:
            generation['X'] = [{k: int(v) for k, v in x.items()} for x in generation['X']]
            generation['F'] = generation['F'].tolist()
            generation['G'] = generation['G'].tolist()
        save_json(folder / 'history.json', history)
        summary.append({'run': index, 'solutions': len(F), 'status': 'ok' if len(F) else 'no_feasible_solution'})
        from .visualization import plot_front
        plot_front(F, folder / 'pareto.png')
    save_json(path / 'summary.json', summary)
    return path


def main():
    parser = argparse.ArgumentParser(description='MI-tree monitoring-network optimization')
    parser.add_argument('--config', default='configs/demo.json')
    parser.add_argument('--output')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as stream:
        config = json.load(stream)
    print(run(config, args.output))

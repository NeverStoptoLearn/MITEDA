from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Iterable, Optional
import copy
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from .encoding import *
from .problem import GroundwaterMonitoringMOProblem
from .model import EDAParams, MixedWellLayoutModel

def mixed_mutation(X_list: list, problem, prob_mut: float=0.1, rng: np.random.Generator=None) -> list:
    """
    对混合编码解进行变异：
    - 输入 X_list: 每个元素是 x_dict，形如 {"x0":..,"y0":..,"b0":.., ...}
    - 对每个解，以 prob_mut 的概率整体触发变异；
    - 对每口井：
        * 以 0.3 的概率，对 (x,y) 进行高斯扰动（步长为搜索范围的 5%）
        * 以 0.05 的概率，翻转 b (0↔1)
    - 最后可选：对 new_dict 做一次 canonicalize_mixed，保证井顺序规范化（适合 EDA）
    """
    if rng is None:
        raise ValueError('Pass the experiment RNG explicitly')
    X_mutated = []
    x_min, x_max = problem.search_area[0]
    y_min, y_max = problem.search_area[1]
    n_wells = problem.n_wells
    step_x = (x_max - x_min) * 0.05
    step_y = (y_max - y_min) * 0.05
    for x_dict in X_list:
        new_dict = copy.deepcopy(x_dict)
        if rng.random() < prob_mut:
            for i in range(n_wells):
                key_x = f'x{i}'
                key_y = f'y{i}'
                key_b = f'b{i}'
                if rng.random() < 0.3:
                    xi = float(new_dict[key_x])
                    yi = float(new_dict[key_y])
                    xi = np.clip(xi + rng.normal(0, step_x), x_min, x_max)
                    yi = np.clip(yi + rng.normal(0, step_y), y_min, y_max)
                    new_dict[key_x] = int(round(xi))
                    new_dict[key_y] = int(round(yi))
                if rng.random() < 0.05:
                    bi = int(new_dict[key_b])
                    new_dict[key_b] = 1 - bi
        new_dict = canonicalize_mixed(new_dict, n_wells)
        X_mutated.append(new_dict)
    return X_mutated

def evaluate_population_mixed(problem: GroundwaterMonitoringMOProblem, pop: Population) -> Population:
    """
    对混合编码种群进行评估：
    - pop.get("X") 是一个 dtype=object 的数组，每个元素是 x_dict
    - 调用 problem._evaluate(X_dict, out) 填充 F, G
    """
    X_arr = pop.get('X')
    n = len(pop)
    F = np.zeros((n, problem.n_obj), dtype=float)
    G = np.zeros((n, problem.n_constr), dtype=float)
    cano_ids = []
    for i in range(n):
        x_dict = X_arr[i]
        out = {}
        problem._evaluate(x_dict, out)
        F[i, :] = out['F']
        G[i, :] = out['G']
        cano_ids.append(out.get('cano_id', None))
    pop.set('F', F)
    pop.set('G', G)
    pop.set('cano_id', np.array(cano_ids, dtype=object))
    return pop

def mo_eda_optimize_mixed(problem: GroundwaterMonitoringMOProblem, pop_size: int=10, n_gen: int=1000, eda_params: Optional[EDAParams]=None, verbose: bool=True, history: bool=False) -> Population | tuple[Population, list[dict]]:
    """
    基于概率模型的混合编码多目标 EDA：
    - 编码: x_i, y_i, b_i (整数+二进制)，内部用向量 (x,y,b) 统计分布
    - 多目标选择: RankAndCrowdingSurvival (NSGA-II 生存机制)
    返回: 最终非支配解的 Population
    """
    if pop_size < 4 or n_gen < 1:
        raise ValueError('pop_size >= 4 and n_gen >= 1 are required')
    if eda_params is None:
        eda_params = EDAParams(mu_size=min(20, pop_size))
    if not 4 <= eda_params.mu_size <= pop_size or eda_params.min_std <= 0:
        raise ValueError('Invalid elite size or minimum standard deviation')
    rng = np.random.default_rng()
    n_wells = problem.n_wells
    x_min, x_max = problem.search_area[0]
    y_min, y_max = problem.search_area[1]
    X_init = []
    for _ in range(pop_size):
        wells = []
        for i in range(n_wells):
            xi = int(rng.integers(int(x_min), int(x_max) + 1))
            yi = int(rng.integers(int(y_min), int(y_max) + 1))
            bi = int(rng.integers(0, 2))
            wells.append((xi, yi, bi))
        x_dict = encode_mixed_solution(wells, n_wells)
        x_dict = canonicalize_mixed(x_dict, n_wells)
        X_init.append(x_dict)
    X_init = np.asarray(X_init, dtype=object)
    pop = Population.new('X', X_init)
    pop = evaluate_population_mixed(problem, pop)
    model = MixedWellLayoutModel(problem, eda_params)
    model.initialize_from_population(X_init)
    survival = RankAndCrowdingSurvival()
    hist = []
    for gen in range(n_gen):
        elite = survival.do(problem, pop, n_survive=eda_params.mu_size)
        X_elite = elite.get('X')
        if gen > 0:
            model.update_from_elite(X_elite)
        X_off = model.sample_dicts(pop_size, rng=rng)
        X_off = mixed_mutation(X_off, problem, prob_mut=0.1, rng=rng)
        X_off = np.asarray(X_off, dtype=object)
        off = Population.new('X', X_off)
        off = evaluate_population_mixed(problem, off)
        pop = Population.merge(pop, off)
        pop = survival.do(problem, pop, n_survive=pop_size)
        if verbose and (gen % 10 == 0 or gen == n_gen - 1):
            F = pop.get('F')
            f1_min, f1_mean = (F[:, 0].min(), F[:, 0].mean())
            f2_min, f2_mean = (F[:, 1].min(), F[:, 1].mean())
            print(f'[Gen {gen:03d}] F1(min/mean) = {f1_min:.3e}/{f1_mean:.3e}, F2(min/mean) = {f2_min:.3e}/{f2_mean:.3e}')
        if history:
            X_copy = [x.copy() for x in pop.get('X')]
            F_copy = pop.get('F').copy()
            G_copy = pop.get('G').copy()
            hist.append({'gen': gen, 'X': X_copy, 'F': F_copy, 'G': G_copy})
    F = pop.get('F')
    feasible = np.flatnonzero(np.all(pop.get('G') <= 1e-10, axis=1))
    if len(feasible):
        nds = NonDominatedSorting().do(F[feasible], only_non_dominated_front=True)
        pareto_pop = pop[feasible[nds]]
    else:
        pareto_pop = pop[np.array([], dtype=int)]
    if history:
        return (pareto_pop, hist)
    else:
        return pareto_pop

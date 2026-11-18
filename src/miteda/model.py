from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Iterable, Optional
from dataclasses import dataclass
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from .encoding import decode_mixed_solution, encode_mixed_solution, canonicalize_mixed
from .alignment import align_wells_to_template

@dataclass
class EDAParams:
    mu_size: int = 20
    min_std: float = 0.1

def estimate_mi(a, b, continuous_a, continuous_b):
    """Handle collapsed elites and singleton classes without estimator failures."""
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return 0.0
    if not continuous_a and (not continuous_b):
        return float(mutual_info_score(a, b))
    neighbors = min(3, len(a) - 1)
    if continuous_a and continuous_b:
        return float(mutual_info_regression(a[:, None], b, n_neighbors=neighbors)[0])
    c, d = (a, b) if continuous_a else (b, a)
    counts = np.unique(d, return_counts=True)[1]
    if counts.min() < 2:
        return 0.0
    neighbors = min(neighbors, int(counts.min()) - 1)
    return float(mutual_info_classif(c[:, None], d, n_neighbors=neighbors)[0])

class MixedWellLayoutModel:
    """
    符合论文原意的全局 MI-Tree 概率模型：
    - 在统一的 3*n_wells 维度空间内构建最大互信息生成树。
    - 自动识别并处理 连续-连续(C-C), 连续-离散(C-D), 离散-离散(D-D), 离散-连续(D-C) 的条件概率。
    """

    def __init__(self, problem, eda_params):
        self.problem = problem
        self.n_wells = problem.n_wells
        self.d = 3 * self.n_wells
        self.mu_size = eda_params.mu_size
        self.min_std = float(eda_params.min_std)
        self.tree_parent = None
        self.tree_root = None
        self.topo_order = []
        self.models = {}

    def initialize_from_population(self, X_list):
        Theta = self._extract_global_array(X_list)
        self._fit_global_mi_tree(Theta)

    def update_from_elite(self, X_elite):
        template_dict = X_elite[0]
        template_wells = decode_mixed_solution(template_dict, self.n_wells)
        aligned_X_elite = []
        for x_dict in X_elite:
            wells = decode_mixed_solution(x_dict, self.n_wells)
            wells_aligned = align_wells_to_template(wells, template_wells)
            new_dict = encode_mixed_solution(wells_aligned, self.n_wells)
            aligned_X_elite.append(new_dict)
        Theta = self._extract_global_array(aligned_X_elite)
        self._fit_global_mi_tree(Theta)

    def sample_dicts(self, n_samples: int, rng: np.random.Generator=None):
        if rng is None:
            raise ValueError('Pass the experiment RNG explicitly')
        x_min, x_max = self.problem.search_area[0]
        y_min, y_max = self.problem.search_area[1]
        samples = np.zeros((n_samples, self.d), dtype=float)
        for j in self.topo_order:
            model = self.models[j]
            mtype = model['type']
            if mtype == 'root_c':
                samples[:, j] = rng.normal(model['mu'], model['std'], size=n_samples)
            elif mtype == 'root_d':
                samples[:, j] = (rng.random(n_samples) < model['p']).astype(float)
            else:
                i = self.tree_parent[j]
                val_i = samples[:, i]
                if mtype == 'd_d':
                    p_arr = np.where(val_i == 1, model['p1'], model['p0'])
                    samples[:, j] = (rng.random(n_samples) < p_arr).astype(float)
                elif mtype == 'd_c':
                    mu_arr = np.where(val_i == 1, model['mu1'], model['mu0'])
                    std_arr = np.where(val_i == 1, model['std1'], model['std0'])
                    samples[:, j] = rng.normal(mu_arr, std_arr)
                elif mtype == 'c_c':
                    mu_arr = model['w'] * val_i + model['b']
                    samples[:, j] = rng.normal(mu_arr, model['std'], size=n_samples)
                elif mtype == 'c_d':
                    if model['clf'] is not None:
                        p_arr = model['clf'].predict_proba(val_i.reshape(-1, 1))[:, 1]
                    else:
                        p_arr = np.full(n_samples, model['p_fallback'])
                    samples[:, j] = (rng.random(n_samples) < p_arr).astype(float)
        X_dicts = []
        for s in range(n_samples):
            wells = []
            for k in range(self.n_wells):
                xi = int(np.clip(np.rint(samples[s, k]), x_min, x_max))
                yi = int(np.clip(np.rint(samples[s, self.n_wells + k]), y_min, y_max))
                bi = int(samples[s, 2 * self.n_wells + k] > 0.5)
                wells.append((xi, yi, bi))
            x_dict = encode_mixed_solution(wells, self.n_wells)
            x_dict = canonicalize_mixed(x_dict, self.n_wells)
            X_dicts.append(x_dict)
        return X_dicts

    def _extract_global_array(self, X_list):
        """将 x_dict 展开为 shape=(n_ind, 3*n_wells) 的连续矩阵"""
        n_ind = len(X_list)
        Theta = np.zeros((n_ind, self.d), dtype=float)
        for idx, x_dict in enumerate(X_list):
            for i in range(self.n_wells):
                Theta[idx, i] = float(x_dict[f'x{i}'])
                Theta[idx, self.n_wells + i] = float(x_dict[f'y{i}'])
                Theta[idx, 2 * self.n_wells + i] = float(x_dict[f'b{i}'])
        return Theta

    def _is_continuous(self, index):
        """判断索引维度是否为连续变量 (x 或 y)"""
        return index < 2 * self.n_wells

    def _fit_global_mi_tree(self, Theta):
        """基于精英集计算全局互信息并拟合异构条件模型"""
        mi_mat = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(i + 1, self.d):
                v_i = Theta[:, i]
                v_j = Theta[:, j]
                is_c_i = self._is_continuous(i)
                is_c_j = self._is_continuous(j)
                mi = estimate_mi(v_i, v_j, is_c_i, is_c_j)
                mi_mat[i, j] = mi_mat[j, i] = mi
        self.tree_parent = self._build_max_spanning_tree(mi_mat)
        self.tree_root = 0
        children = [[] for _ in range(self.d)]
        for j in range(self.d):
            i = self.tree_parent[j]
            if i >= 0:
                children[i].append(j)
        self.topo_order = []
        queue = [self.tree_root]
        while queue:
            curr = queue.pop(0)
            self.topo_order.append(curr)
            for child in children[curr]:
                queue.append(child)
        for j in range(self.d):
            if j not in self.topo_order:
                self.topo_order.append(j)
        self.models = {}
        for j in range(self.d):
            i = self.tree_parent[j]
            is_c_j = self._is_continuous(j)
            v_j = Theta[:, j]
            if i == -1:
                if is_c_j:
                    self.models[j] = {'type': 'root_c', 'mu': np.mean(v_j), 'std': max(np.std(v_j), self.min_std)}
                else:
                    self.models[j] = {'type': 'root_d', 'p': np.clip(np.mean(v_j), 0.001, 1 - 0.001)}
            else:
                is_c_i = self._is_continuous(i)
                v_i = Theta[:, i]
                if not is_c_i and (not is_c_j):
                    p1 = np.mean(v_j[v_i == 1]) if np.any(v_i == 1) else np.mean(v_j)
                    p0 = np.mean(v_j[v_i == 0]) if np.any(v_i == 0) else np.mean(v_j)
                    self.models[j] = {'type': 'd_d', 'p1': np.clip(p1, 0.001, 1 - 0.001), 'p0': np.clip(p0, 0.001, 1 - 0.001)}
                elif not is_c_i and is_c_j:
                    mu1 = np.mean(v_j[v_i == 1]) if np.any(v_i == 1) else np.mean(v_j)
                    std1 = max(np.std(v_j[v_i == 1]) if np.any(v_i == 1) else np.std(v_j), self.min_std)
                    mu0 = np.mean(v_j[v_i == 0]) if np.any(v_i == 0) else np.mean(v_j)
                    std0 = max(np.std(v_j[v_i == 0]) if np.any(v_i == 0) else np.std(v_j), self.min_std)
                    self.models[j] = {'type': 'd_c', 'mu1': mu1, 'std1': std1, 'mu0': mu0, 'std0': std0}
                elif is_c_i and is_c_j:
                    reg = LinearRegression().fit(v_i.reshape(-1, 1), v_j)
                    preds = reg.predict(v_i.reshape(-1, 1))
                    std_res = max(np.std(v_j - preds), self.min_std)
                    self.models[j] = {'type': 'c_c', 'w': reg.coef_[0], 'b': reg.intercept_, 'std': std_res}
                elif is_c_i and (not is_c_j):
                    if len(np.unique(v_j)) > 1:
                        clf = LogisticRegression(solver='lbfgs').fit(v_i.reshape(-1, 1), v_j)
                        self.models[j] = {'type': 'c_d', 'clf': clf}
                    else:
                        self.models[j] = {'type': 'c_d', 'clf': None, 'p_fallback': np.mean(v_j)}

    def _build_max_spanning_tree(self, mi_mat):
        """基于互信息矩阵构造最大生成树 (Prim 算法)"""
        visited = np.zeros(self.d, dtype=bool)
        parent = np.full(self.d, -1, dtype=int)
        visited[0] = True
        for _ in range(self.d - 1):
            best_i, best_j, best_w = (-1, -1, -1.0)
            for i in range(self.d):
                if not visited[i]:
                    continue
                for j in range(self.d):
                    if visited[j]:
                        continue
                    if mi_mat[i, j] > best_w:
                        best_w, best_i, best_j = (mi_mat[i, j], i, j)
            if best_j == -1:
                break
            parent[best_j] = best_i
            visited[best_j] = True
        return parent

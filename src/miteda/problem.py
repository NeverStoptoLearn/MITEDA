from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Iterable, Optional
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Binary
from .encoding import *

class GroundwaterMonitoringMOProblem(ElementwiseProblem):
    """
    地下水监测井优化 - 多目标 / 混合编码 / 溯源鲁棒性 (max-min)

    编码:
      每口井 (x_i, y_i, b_i)，b_i∈{0,1} 控制是否启用；n_wells 为最大井数上限。
      pymoo 中用 vars={"x0":Integer, "y0":Integer, "b0":Binary, ...} 进行混合编码。

    目标:
      F1: 最小化 -f2_maxmin(I)，其中
          f2_maxmin(I) = min_{p≠q} mean D_{pq}(I)
          mean D_{pq}^2(I) = (1/R) ∑_r ||Y(I; θ_p, K_r) - Y(I; θ_q, K_r)||²
      F2: 最小化 成本（按口计费或自定义 install_cost_fn）

    约束 (G<=0):
      G1: 最小井距约束违背量（仅对已选井），G1 = max(0, min_distance - min_pairwise_distance)
      G2: 最小井数约束违背量，G2 = max(0, min_wells - n_selected)

    关键输入:
      forward_model: 函数 c = forward_model(c0, x, y, t, v, n, al, ah, Q, B, xc, yc)
      observation_times: [t1, t2, ...]
      search_area: ((x_min, x_max), (y_min, y_max))

      source_scenarios: list of θ_p，每个 θ_p 可为:
          - dict: {"xc":..., "yc":..., "Q":..., "v":...} (Q、v 可选)
          - list/tuple: [xc, yc] 或 [xc, yc, Q, v]

      hydro_scenarios: list of K_r（可选），每个 K_r 是 dict:
          - 例如 {"v":..., "n":..., "al":..., "ah":..., "B":..., "c0":...}
          - 任一缺省时用默认值: v 默认用 θ 的 v, n=0.25, al=30, ah=6, B=100, c0=1000
          - 若不提供，则使用单一情景 [None]，不考虑水文不确定性。
    """

    def __init__(self, forward_model, observation_times, search_area, source_scenarios, hydro_scenarios: Optional[List[Dict[str, float]]]=None, min_distance: float=10.0, n_wells: int=10, min_wells: int=3, cost_per_well: float=1.0, install_cost_fn=None, cache_precision: int=1):
        self.forward_model = forward_model
        self.observation_times = list(observation_times)
        self.search_area = search_area
        self.n_wells = int(n_wells)
        self.min_distance = float(min_distance)
        self.min_wells = int(min_wells)
        self.cost_per_well = float(cost_per_well)
        self.install_cost_fn = install_cost_fn
        self.cache_precision = int(cache_precision)
        if source_scenarios is None or len(source_scenarios) < 2:
            raise ValueError('source_scenarios 至少需要包含两个不同的污染源情景。')
        self.source_scenarios = list(source_scenarios)
        if hydro_scenarios is None or len(hydro_scenarios) == 0:
            self.hydro_scenarios = [None]
        else:
            self.hydro_scenarios = list(hydro_scenarios)
        x_min, x_max = search_area[0]
        y_min, y_max = search_area[1]
        vars_def = {}
        for i in range(self.n_wells):
            vars_def[f'x{i}'] = Integer(bounds=(x_min, x_max))
            vars_def[f'y{i}'] = Integer(bounds=(y_min, y_max))
            vars_def[f'b{i}'] = Binary()
        super().__init__(vars=vars_def, n_obj=2, n_ieq_constr=2)
        self._cache: Dict[Tuple[str, ...], float] = {}

    def _decode_solution(self, X: Dict[str, float], return_b: bool=False):
        """
        X: pymoo 传入的混合变量字典 { "x0":..,"y0":..,"b0":..,... }
        先进行 canonicalize_mixed，再解码为 wells_all 或选中的 well_locations
        """
        X_canon = canonicalize_mixed(X, self.n_wells)
        wells_all = decode_all_wells(X_canon, self.n_wells)
        return wells_all if return_b else decode_selected_locations(wells_all)

    def get_well_locations(self, X: Dict[str, float]):
        """
        仅获取已选井的位置列表 [(x,y), ...]
        """
        return self._decode_solution(X, return_b=False)

    def _canonicalize_selected_wells(self, well_locations):
        """
        对已选井的位置做一个“顺序无关”的规范化表示：
        - 只看 (x, y)，不关心原来在第几个槽位
        - 按 (x, y) 排序，保证同一组井总是得到同一个 tuple
        - 使用 cache_precision 做四舍五入，避免浮点噪声导致 key 不稳定
        """
        if not well_locations:
            return tuple()
        prec = int(getattr(self, 'cache_precision', 3))
        cano = []
        for x, y in well_locations:
            x_r = round(float(x), prec)
            y_r = round(float(y), prec)
            cano.append((x_r, y_r))
        cano.sort(key=lambda p: (p[0], p[1]))
        return tuple(cano)

    def _evaluate(self, X, out, *args, **kwargs):
        wells_all = self._decode_solution(X, return_b=True)
        well_locations = decode_selected_locations(wells_all)
        distance_violation = self._calculate_distance_violation(well_locations)
        n_selected = sum((b for _, _, b in wells_all))
        wells_min_violation = max(0.0, float(self.min_wells - n_selected))
        out['G'] = np.array([distance_violation, wells_min_violation], dtype=float)
        cache_key = self._get_cache_key(well_locations, precision=self.cache_precision)
        if cache_key in self._cache:
            f2_value = self._cache[cache_key]
        else:
            f2_value = self._calculate_source_robustness_maxmin(well_locations)
            self._cache[cache_key] = f2_value
        total_cost = self._calculate_cost(wells_all)
        out['F'] = np.array([-float(f2_value), float(total_cost)], dtype=float)
        if not np.all(np.isfinite(out['F'])) or not np.all(np.isfinite(out['G'])):
            raise FloatingPointError('Objective or constraint evaluation produced non-finite values')

    def _get_cache_key(self, well_locations, precision=1.0) -> Tuple[str, ...]:
        """
        根据井位置生成一个与顺序无关的缓存键。
        precision: 控制位置量化精度
        """
        parts = []
        for x, y in well_locations:
            xd = round(x / precision) * precision
            yd = round(y / precision) * precision
            parts.append(f'{xd:.3f}_{yd:.3f}')
        parts.sort()
        return tuple(parts)

    def _calculate_distance_violation(self, well_locations) -> float:
        """
        计算最小井距约束违背：
        若最小井距 d_min < self.min_distance, 则 violation = self.min_distance - d_min，否则为 0。
        """
        m = len(well_locations)
        if m < 2:
            return 0.0
        min_d = float('inf')
        for i in range(m):
            xi, yi = well_locations[i]
            for j in range(i + 1, m):
                xj, yj = well_locations[j]
                d = float(np.hypot(xi - xj, yi - yj))
                if d < min_d:
                    min_d = d
        return max(0.0, self.min_distance - min_d)

    def _calculate_cost(self, wells_all) -> float:
        """
        简单成本：每口被选井计一个成本（或调用 install_cost_fn）
        """
        cost = 0.0
        for x, y, b in wells_all:
            if b == 1:
                if self.install_cost_fn is not None:
                    cost += float(self.install_cost_fn(x, y))
                else:
                    cost += self.cost_per_well
        return float(cost)

    def _calculate_source_robustness_maxmin(self, well_locations) -> float:
        """
        计算 max-min 溯源鲁棒性目标：
        f2_maxmin(I) = min_{p≠q} mean D_{pq}(I),
        mean D_{pq}^2(I) = (1/R) ∑_r ||Y(I; θ_p, K_r) - Y(I; θ_q, K_r)||²

        - 若井数为 0 或源情景不足 2 个，则返回 0。
        """
        if len(well_locations) == 0:
            return 0.0
        P = len(self.source_scenarios)
        if P < 2:
            return 0.0
        hydro_list = self.hydro_scenarios if self.hydro_scenarios and len(self.hydro_scenarios) > 0 else [None]
        R = len(hydro_list)
        Y = [[None for _ in range(R)] for _ in range(P)]
        for p, theta in enumerate(self.source_scenarios):
            for r, K in enumerate(hydro_list):
                Y[p][r] = self._simulate_source_hydro(theta, K, well_locations)
        min_dbar = np.inf
        for p in range(P - 1):
            for q in range(p + 1, P):
                d2_sum = 0.0
                for r in range(R):
                    diff = Y[p][r] - Y[q][r]
                    d2 = float(np.sum(diff ** 2))
                    d2_sum += d2
                d2_mean = d2_sum / R
                dbar = float(np.sqrt(max(d2_mean, 0.0)))
                if dbar < min_dbar:
                    min_dbar = dbar
        if not np.isfinite(min_dbar):
            return 0.0
        return min_dbar

    def _simulate_source_hydro(self, theta, hydro: Optional[Dict[str, float]], well_locations) -> np.ndarray:
        """
        给定源情景 theta、水文情景 hydro 和井布局 well_locations，生成拼接观测向量 Y(I; theta, hydro)。
        - theta: dict 或 list/tuple
        - hydro: dict 或 None
        返回: 1D numpy 数组，长度 = len(wells) * len(observation_times)
        """
        if isinstance(theta, dict):
            xc = float(theta['xc'])
            yc = float(theta['yc'])
            Q = float(theta.get('Q', 50.0))
            v_theta = float(theta.get('v', 1.0))
        else:
            arr = np.asarray(theta, dtype=float)
            if arr.size >= 4:
                xc, yc, Q, v_theta = arr[:4]
            elif arr.size == 2:
                xc, yc = arr
                Q = 50.0
                v_theta = 1.0
            else:
                raise ValueError(f'theta 维度不支持: {arr}')
        if hydro is None:
            v = v_theta
            n = 0.25
            al = 30.0
            ah = 6.0
            B = 100.0
            c0 = 1000.0
        else:
            v = float(hydro.get('v', v_theta))
            n = float(hydro.get('n', 0.25))
            al = float(hydro.get('al', 30.0))
            ah = float(hydro.get('ah', 6.0))
            B = float(hydro.get('B', 100.0))
            c0 = float(hydro.get('c0', 1000.0))
        preds = []
        for xw, yw in well_locations:
            for t in self.observation_times:
                try:
                    c = self.forward_model(c0=c0, x=xw, y=yw, t=t, v=v, n=n, al=al, ah=ah, Q=Q, B=B, xc=xc, yc=yc)
                    value = np.asarray(c, dtype=float)
                    if value.size != 1 or not np.isfinite(value).all():
                        raise ValueError('Forward model must return one finite concentration')
                    preds.append(float(value.item()))
                except Exception as exc:
                    raise RuntimeError(f'Forward simulation failed at well {(xw, yw)}, time {t}, source {theta}') from exc
        return np.asarray(preds, dtype=float)

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Iterable, Optional
def decode_mixed_solution(x_dict, n_wells):
    """
    将混合变量解 x_dict 解码为井列表:
    wells = [(x0, y0, b0), (x1, y1, b1), ...]
    """
    wells = []
    for i in range(n_wells):
        xi = int(x_dict[f"x{i}"])
        yi = int(x_dict[f"y{i}"])
        bi = int(bool(x_dict[f"b{i}"]))  # 确保是 0/1
        wells.append((xi, yi, bi))
    return wells

def encode_mixed_solution(wells, n_wells):
    """
    将 wells = [(x0, y0, b0), ...] 编码回字典形式 x_dict
    """
    x_dict = {}
    for i in range(n_wells):
        xi, yi, bi = wells[i]
        x_dict[f"x{i}"] = int(xi)
        x_dict[f"y{i}"] = int(yi)
        x_dict[f"b{i}"] = int(bi)
    return x_dict

def canonicalize_mixed(x_dict, n_wells):
    """
    混合变量规范化表示：
    - 先把所有井解码成 (x, y, b)
    - 把已选井(b=1)和未选井(b=0)分开
    - 对“已选井”按 (x, y) 排序，放在前面
    - 对“未选井”按 (x, y) 排序，放在后面
    - 再编码回 x_dict
    """
    wells = decode_mixed_solution(x_dict, n_wells)

    active = [(x, y, 1) for (x, y, b) in wells if b == 1]
    inactive = [(x, y, 0) for (x, y, b) in wells if b == 0]

    active_sorted = sorted(active, key=lambda w: (w[0], w[1]))
    inactive_sorted = sorted(inactive, key=lambda w: (w[0], w[1]))

    wells_sorted = active_sorted + inactive_sorted

    # 防止奇怪情况
    if len(wells_sorted) < n_wells:
        wells_sorted += [(0, 0, 0)] * (n_wells - len(wells_sorted))
    wells_sorted = wells_sorted[:n_wells]

    return encode_mixed_solution(wells_sorted, n_wells)

def decode_all_wells(X: Dict[str, float], n_wells: int) -> List[Tuple[float, float, int]]:
    """
    将决策变量字典 X 解码为 [(x, y, b), ...]，长度为 n_wells
    约定键名：x0,y0,b0, x1,y1,b1, ..., x{n-1},y{n-1},b{n-1}
    """
    wells = []
    for i in range(n_wells):
        xi = float(X[f"x{i}"])
        yi = float(X[f"y{i}"])
        bi = int(X[f"b{i}"])
        wells.append((xi, yi, bi))
    return wells

def decode_selected_locations(wells_all: Iterable[Tuple[float, float, int]]) -> List[Tuple[float, float]]:
    """
    仅保留被选择的井位置 (b==1)
    """
    return [(x, y) for (x, y, b) in wells_all if b == 1]

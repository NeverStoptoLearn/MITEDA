import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def align_wells_to_template(wells, template_wells):
    """
    使用匈牙利算法将当前的 wells 列表与 template_wells 进行最佳匹配对齐。

    参数:
        wells: 当前解的井列表 [(x,y,b), ...]
        template_wells: 参考模板的井列表 [(x,y,b), ...] (通常是上一代的均值或精英解)

    返回:
        aligned_wells: 重新排列顺序后的 wells
    """
    # 提取坐标用于计算距离矩阵 (只考虑激活的井，或者全部井)
    # 这里简单起见，我们对所有井槽位进行匹配
    # 注意：为了计算方便，我们把 (x, y, b) 视为 3D 向量，或者仅用 (x, y) 匹配

    # 提取位置坐标 (n_wells, 2)
    pos_current = np.array([[w[0], w[1]] for w in wells])
    pos_template = np.array([[w[0], w[1]] for w in template_wells])

    # 计算距离矩阵
    dists = cdist(pos_current, pos_template, metric='euclidean')

    # 匈牙利算法找到最小总距离的行-列索引匹配
    row_ind, col_ind = linear_sum_assignment(dists)

    # row_ind 是当前 wells 的索引，col_ind 是目标模板的索引
    # 我们希望将 wells[row] 移动到 col 的位置，使其与模板对齐
    # 即：新的第 k 个井，应该是原来的第 row_ind[k] 个井，其中 col_ind[row_ind[k]] == k

    # 创建一个新的列表用于存放对齐后的井
    aligned_wells = [None] * len(wells)

    for r, c in zip(row_ind, col_ind):
        # 意思是：当前的第 r 个井，最佳匹配是模板的第 c 个位置
        # 所以我们把当前的第 r 个井放到新的第 c 个槽位
        aligned_wells[c] = wells[r]

    return aligned_wells

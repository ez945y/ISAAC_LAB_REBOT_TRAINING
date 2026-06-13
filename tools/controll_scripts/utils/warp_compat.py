# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX view 回傳值的相容性工具。

Isaac Sim 5.1 / Kit 110 起，`root_physx_view.get_*()`（如 ``get_dof_limits``、
``get_jacobians``、``get_generalized_mass_matrices``、
``get_gravity_compensation_forces``）回傳的是 ``warp.array``，不支援 PyTorch
風格的進階索引。舊版本回傳的本來就是 ``torch.Tensor``。

``physx_to_torch`` 把兩者統一成 ``torch.Tensor``（共享記憶體、保留 device），
讓呼叫端可以照常用 ``[:, ids, ...]`` 索引。
"""

import torch


def physx_to_torch(value):
    """將 PhysX view 的回傳值轉成 torch tensor（已是 tensor 則原樣回傳）。"""
    if isinstance(value, torch.Tensor):
        return value
    import warp as wp
    return wp.to_torch(value)

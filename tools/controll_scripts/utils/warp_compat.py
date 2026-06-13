# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Warp / PhysX 回傳值的相容性工具。

Isaac Sim 5.1 / Kit 110 起，Isaac Lab 改用 warp 後端，下列來源回傳的不再是
``torch.Tensor``，而是 ``warp.array`` 或 lazy 的 ``ProxyArray``（dtype 可能是
``float``，也可能是 ``vec3`` / ``quatf`` 等 warp 結構型別），無法直接餵進
PyTorch 的數學函式（如 ``math_utils.subtract_frame_transforms``）：

* ``root_physx_view.get_*()`` —— ``get_dof_limits``、``get_jacobians``、
  ``get_generalized_mass_matrices``、``get_gravity_compensation_forces``。
* ``robot.data.*`` 的姿態 / 速度 —— ``body_pos_w``、``body_quat_w``、
  ``root_pos_w``、``root_quat_w``、``joint_pos``、``joint_vel`` 等。

``physx_to_torch`` 把這些統一轉成 ``torch.Tensor``（共享記憶體、保留 device），
``vec3`` / ``quatf`` 等結構型別會展平成最後一維（3 / 4）。已經是 tensor 時直接
原樣回傳，因此對舊版本是無作用的安全包裝。
"""

import torch


def physx_to_torch(value):
    """將 warp / PhysX 的回傳值轉成 torch tensor（已是 tensor 則原樣回傳）。"""
    if isinstance(value, torch.Tensor):
        return value

    import warp as wp

    # warp.array 與 ProxyArray 都能由 wp.to_torch 處理（含 vec3/quatf 結構型別）。
    return wp.to_torch(value)

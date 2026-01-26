# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import apply_delta_pose, compute_pose_error

if TYPE_CHECKING:
    from .differential_ik_cfg import DifferentialIKControllerCfg


class DifferentialIKController:
    r"""Differential inverse kinematics (IK) controller.

    This controller is based on the concept of differential inverse kinematics [1, 2] which is a method for computing
    the change in joint positions that yields the desired change in pose.

    .. math::

        \Delta \mathbf{q} &= \mathbf{J}^{\dagger} \Delta \mathbf{x} \\
        \mathbf{q}_{\text{desired}} &= \mathbf{q}_{\text{current}} + \Delta \mathbf{q}

    where :math:`\mathbf{J}^{\dagger}` is the pseudo-inverse of the Jacobian matrix :math:`\mathbf{J}`,
    :math:`\Delta \mathbf{x}` is the desired change in pose, and :math:`\mathbf{q}_{\text{current}}`
    is the current joint positions.

    To deal with singularity in Jacobian, the following methods are supported for computing inverse of the Jacobian:

    - "pinv": Moore-Penrose pseudo-inverse
    - "svd": Adaptive singular-value decomposition (SVD)
    - "trans": Transpose of matrix
    - "dls": Damped version of Moore-Penrose pseudo-inverse (also called Levenberg-Marquardt)


    .. caution::
        The controller does not assume anything about the frames of the current and desired end-effector pose,
        or the joint-space velocities. It is up to the user to ensure that these quantities are given
        in the correct format.

    Reference:

    1. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    2. `Introduction to Inverse Kinematics <https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf>`_
       by Samuel R. Buss (University of California, San Diego)

    """

    def __init__(self, cfg: DifferentialIKControllerCfg, num_envs: int, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            num_envs: The number of environments.
            device: The device to use for computations.
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        # create buffers
        self.ee_pos_des = torch.zeros(self.num_envs, 3, device=self._device)
        self.ee_quat_des = torch.zeros(self.num_envs, 4, device=self._device)
        # -- input command
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self._device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        if self.cfg.command_type == "position":
            return 3  # (x, y, z)
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        else:
            return 7  # (x, y, z, qw, qx, qy, qz)

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the internals.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        pass

    def set_command(
        self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    ):
        """Set target end-effector pose command.

        Based on the configured command type and relative mode, the method computes the desired end-effector pose.
        It is up to the user to ensure that the command is given in the correct frame. The method only
        applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

        Args:
            command: The input command in shape (N, 3) or (N, 6) or (N, 7).
            ee_pos: The current end-effector position in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current end-effector orientation (w, x, y, z) in shape (N, 4).
                This is only needed if the command type is ``position_*`` or ``pose_rel``.

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
        """
        # store command
        self._command[:] = command
        # compute the desired end-effector pose
        if self.cfg.command_type == "position":
            # we need end-effector orientation even though we are in position mode
            # this is only needed for display purposes
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                self.ee_pos_des[:] = ee_pos + self._command
                self.ee_quat_des[:] = ee_quat
            else:
                self.ee_pos_des[:] = self._command
                self.ee_quat_des[:] = ee_quat
        else:
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError(
                        "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                    )
                # print("command: ", self._command[0].cpu().numpy())
                self.ee_pos_des, self.ee_quat_des = apply_delta_pose(ee_pos, ee_quat, self._command)
            else:
                self.ee_pos_des = self._command[:, 0:3]
                self.ee_quat_des = self._command[:, 3:7]

    def compute(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Computes the target joint positions that will yield the desired end effector pose.

        Args:
            ee_pos: The current end-effector position in shape (N, 3).
            ee_quat: The current end-effector orientation in shape (N, 4).
            jacobian: The geometric jacobian matrix in shape (N, 6, num_joints).
            joint_pos: The current joint positions in shape (N, num_joints).

        Returns:
            The target joint positions commands in shape (N, num_joints).
        """
        # compute the delta in joint-space
        if "position" in self.cfg.command_type:
            position_error = self.ee_pos_des - ee_pos
            jacobian_pos = jacobian[:, 0:3]
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=position_error, jacobian=jacobian_pos)
        else:
            # print(f"[DiffIK] ee_pos_des: {self.ee_pos_des[0].cpu().numpy()}")
            # print(f"[DiffIK] ee_pos: {ee_pos[0].cpu().numpy()}")


            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)
        # print(f"[DiffIK] delta_joint_pos: {delta_joint_pos[0].cpu().numpy()}")
        return joint_pos + delta_joint_pos

    """
    Helper functions.
    """

    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Computes the change in joint position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired
        end-effector position.

        Args:
            delta_pose: The desired delta pose in shape (N, 3) or (N, 6).
            jacobian: The geometric jacobian matrix in shape (N, 3, num_joints) or (N, 6, num_joints).

        Returns:
            The desired delta in joint space. Shape is (N, num-jointsß).
        """
        if self.cfg.ik_params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{self.cfg.ik_method}' is not defined!")
        # compute the delta in joint-space
        if self.cfg.ik_method == "pinv":  # Jacobian pseudo-inverse
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "svd":  # adaptive SVD
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            min_singular_value = self.cfg.ik_params["min_singular_value"]
            # computation
            # U: 6xd, S: dxd, V: d x num-joint
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "trans":  # Jacobian transpose
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_joint_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "dls":  # damped least squares
            # parameters
            lambda_val = self.cfg.ik_params["lambda_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
            delta_joint_pos = (
                jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "adaptive":  # Adaptive DLS for 5-DOF SO-ARM
            # ============================================================
            # Adaptive Damped Least Squares (ADLS) for 5-DOF Manipulator
            # ============================================================
            # 這個方法專門為 5-DOF SO-ARM 設計:
            # 1. 位置優先：位置誤差權重高於姿態誤差
            # 2. 自適應阻尼：根據操作性指標（manipulability）自動調整阻尼
            # 3. 加權最小二乘：使用加權矩陣處理欠驅動情況
            
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            lambda_base = self.cfg.ik_params["lambda_base"]
            lambda_max = self.cfg.ik_params["lambda_max"]
            min_singular_value = self.cfg.ik_params["min_singular_value"]
            position_weight = self.cfg.ik_params["position_weight"]
            orientation_weight = self.cfg.ik_params["orientation_weight"]
            
            # Get dimensions
            batch_size = jacobian.shape[0]
            task_dim = jacobian.shape[1]  # 3 for position, 6 for pose
            num_joints = jacobian.shape[2]  # Should be 5 for SO-ARM
            
            # Build weight matrix for task space (position vs orientation)
            if task_dim == 6:
                # Pose control: weight position more than orientation for 5-DOF arm
                weights = torch.tensor(
                    [position_weight] * 3 + [orientation_weight] * 3,
                    device=self._device, dtype=jacobian.dtype
                )
                W = torch.diag(weights).unsqueeze(0).expand(batch_size, -1, -1)
                
                # Apply weights to Jacobian and error
                jacobian_w = W @ jacobian  # (N, 6, num_joints)
                delta_pose_w = (W @ delta_pose.unsqueeze(-1)).squeeze(-1)  # (N, 6)
            else:
                # Position only control
                jacobian_w = jacobian
                delta_pose_w = delta_pose
            
            # Compute manipulability measure using SVD for adaptive damping
            # Manipulability = sqrt(det(J @ J^T)) ≈ product of singular values
            U, S, Vh = torch.linalg.svd(jacobian_w, full_matrices=False)
            
            # Compute minimum singular value as condition measure
            min_sv = S.min(dim=1, keepdim=True)[0]  # (N, 1)
            
            # Adaptive damping: increase damping when approaching singularity
            # lambda = lambda_base + (lambda_max - lambda_base) * exp(-min_sv / threshold)
            damping_factor = torch.exp(-min_sv / min_singular_value)
            lambda_adaptive = lambda_base + (lambda_max - lambda_base) * damping_factor  # (N, 1)
            
            # Damped pseudo-inverse using SVD: J^+ = V @ S_damped^+ @ U^T
            # S_damped^+ = S / (S^2 + lambda^2)
            S_sq = S ** 2
            lambda_sq = lambda_adaptive ** 2  # (N, 1)
            S_damped_inv = S / (S_sq + lambda_sq)  # (N, min(task_dim, num_joints))
            
            # Reconstruct damped pseudo-inverse
            # J^+ = Vh^T @ diag(S_damped_inv) @ U^T
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)  # (N, num_joints, k) where k=min(task_dim, num_joints)
                @ torch.diag_embed(S_damped_inv)     # (N, k, k)
                @ torch.transpose(U, dim0=1, dim1=2)[:, :S.shape[1], :]  # (N, k, task_dim)
            )
            
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose_w.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "dls_5dof":  # DLS for 5-DOF arms (ignore yaw)
            # ============================================================
            # Damped Least Squares for 5-DOF Manipulator (Ignoring Yaw)
            # ============================================================
            # 這個方法專門為缺少 yaw 自由度的 5-DOF 機械臂設計：
            # 1. 忽略 yaw 誤差：將軸角度誤差中的 z 分量設為零
            # 2. 移除 Jacobian 的 yaw 行：減少任務空間維度從 6 到 5
            # 3. 使用加權 DLS：位置誤差權重高於姿態誤差
            #
            # delta_pose 輸入格式 (6D pose 模式):
            #   [Δx, Δy, Δz, ωx, ωy, ωz] 
            #   其中 (ωx, ωy, ωz) 是軸角度誤差:
            #     - ωx: roll  (繞 x 軸旋轉)
            #     - ωy: pitch (繞 y 軸旋轉)  
            #     - ωz: yaw   (繞 z 軸旋轉) <- 這個會被忽略
            
            # parameters
            lambda_val = self.cfg.ik_params["lambda_val"]
            ignore_yaw = self.cfg.ik_params.get("ignore_yaw", True)
            position_weight = self.cfg.ik_params.get("position_weight", 1.0)
            orientation_weight = self.cfg.ik_params.get("orientation_weight", 0.5)
            
            # Get dimensions
            batch_size = jacobian.shape[0]
            task_dim = jacobian.shape[1]  # 3 for position, 6 for pose
            
            if task_dim == 6 and ignore_yaw:
                # ============================================================
                # 處理 6D pose 控制，忽略 yaw
                # ============================================================
                # delta_pose: [Δx, Δy, Δz, ωx, ωy, ωz]
                #              0    1    2   3    4    5
                # 
                # 我們要移除 ωz (index 5)，保留 [Δx, Δy, Δz, ωx, ωy]
                
                # 建立權重向量: [pos_w, pos_w, pos_w, ori_w, ori_w]
                # 注意：移除 yaw 後只有 5 個維度
                weights = torch.tensor(
                    [position_weight, position_weight, position_weight,
                     orientation_weight, orientation_weight],
                    device=self._device, dtype=jacobian.dtype
                )
                
                # 移除 delta_pose 中的 yaw 分量 (index 5)
                # 保留 indices: [0, 1, 2, 3, 4]
                delta_pose_5d = torch.cat([
                    delta_pose[:, 0:3],  # position error: Δx, Δy, Δz
                    delta_pose[:, 3:5]   # orientation error: ωx, ωy (roll, pitch only)
                ], dim=1)  # Shape: (N, 5)
                
                # 移除 Jacobian 中的 yaw 行 (index 5)
                # 原始 Jacobian: (N, 6, num_joints) -> (N, 5, num_joints)
                # Rows: [vx, vy, vz, ωx, ωy, ωz] -> [vx, vy, vz, ωx, ωy]
                jacobian_5d = torch.cat([
                    jacobian[:, 0:3, :],  # linear velocity rows
                    jacobian[:, 3:5, :]   # angular velocity rows (roll, pitch only)
                ], dim=1)  # Shape: (N, 5, num_joints)
                
                # 應用權重
                W = torch.diag(weights).unsqueeze(0).expand(batch_size, -1, -1)  # (N, 5, 5)
                jacobian_w = W @ jacobian_5d  # (N, 5, num_joints)
                delta_pose_w = weights * delta_pose_5d  # (N, 5)
            else:
                # Position only control (3D)
                jacobian_w = jacobian
                delta_pose_w = delta_pose
            
            # Damped Least Squares 計算
            # J^+ = J^T @ (J @ J^T + λ²I)^(-1)
            jacobian_T = torch.transpose(jacobian_w, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian_w.shape[1], device=self._device)
            
            delta_joint_pos = (
                jacobian_T @ torch.inverse(jacobian_w @ jacobian_T + lambda_matrix) @ delta_pose_w.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.cfg.ik_method}")
        
        return delta_joint_pos

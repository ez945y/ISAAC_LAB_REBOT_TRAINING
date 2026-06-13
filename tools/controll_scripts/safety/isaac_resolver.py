# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab kinematics bridge for DAM EE-space SafetyGuard actions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..utils import physx_to_torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..configs.base import BaseRobotConfig
    from ..controllers.base import BaseController


_SO101_ISAAC_TO_URDF_JOINTS = {
    "shoulder_pan": "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist_Pitch",
    "wrist_roll": "Wrist_Roll",
}
_SO101_EE_FRAME_ALIASES = {
    "gripper_link": "gripper",
}


def default_so101_urdf_path() -> str:
    """Return the bundled SO-ARM-101 URDF used for Pinocchio FK."""
    safety_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(safety_dir)
    return os.path.join(package_dir, "so_arm_101", "description", "so101_new_calib.urdf")


def dam_xyzw_to_isaac_wxyz(pose: np.ndarray) -> np.ndarray:
    """Convert DAM pose [x,y,z,qx,qy,qz,qw] to Isaac pose [x,y,z,qw,qx,qy,qz]."""
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape[0] != 7:
        raise ValueError("Expected 7 pose values")
    return np.array([pose[0], pose[1], pose[2], pose[6], pose[3], pose[4], pose[5]])


def isaac_wxyz_to_dam_xyzw(pose: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert Isaac pose [x,y,z,qw,qx,qy,qz] to DAM pose [x,y,z,qx,qy,qz,qw]."""
    if isinstance(pose, torch.Tensor):
        pose_np = pose.detach().cpu().numpy().reshape(-1)
    else:
        pose_np = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose_np.shape[0] != 7:
        raise ValueError("Expected 7 pose values")
    return np.array([pose_np[0], pose_np[1], pose_np[2], pose_np[4], pose_np[5], pose_np[6], pose_np[3]])


class _PinocchioForwardKinematics:
    """Small Pinocchio FK helper for validated DAM joint targets."""

    def __init__(
        self,
        urdf_path: str,
        controlled_joint_names: list[str],
        ee_frame_name: str,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise ImportError(
                "EE-space DAM filtering requires pinocchio for FK. Install pinocchio "
                "or use joint-space DAM filtering."
            ) from exc

        self._pin = pin
        full_model = pin.buildModelFromUrdf(urdf_path)
        all_joint_names = [full_model.names[i] for i in range(1, full_model.njoints)]
        missing_joints = [name for name in controlled_joint_names if name not in all_joint_names]
        if missing_joints:
            raise ValueError(
                "Controlled joints are missing from FK URDF: "
                f"{missing_joints}. Available joints: {all_joint_names}"
            )
        joints_to_lock = [
            full_model.getJointId(name)
            for name in all_joint_names
            if name not in controlled_joint_names
        ]
        q_reference = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, joints_to_lock, q_reference)
        self.data = self.model.createData()
        if not self.model.existFrame(ee_frame_name):
            available_frames = [frame.name for frame in self.model.frames]
            raise ValueError(
                f"EE frame {ee_frame_name!r} is missing from FK model. "
                f"Available frames: {available_frames}"
            )
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)

    def compute(self, joint_positions: np.ndarray) -> np.ndarray:
        q = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.model.nq:
            raise ValueError(f"FK expected {self.model.nq} joint positions, got {q.shape[0]}")
        self._pin.forwardKinematics(self.model, self.data, q)
        self._pin.updateFramePlacements(self.model, self.data)

        o_m_f = self.data.oMf[self.ee_frame_id]
        quat = self._pin.Quaternion(o_m_f.rotation)
        return np.array(
            [
                o_m_f.translation[0],
                o_m_f.translation[1],
                o_m_f.translation[2],
                quat.x,
                quat.y,
                quat.z,
                quat.w,
            ],
            dtype=np.float64,
        )


class IsaacControllerKinematicsResolver:
    """DAM SafetyKinematicsResolver backed by an Isaac Lab controller.

    The inverse path uses the live Isaac articulation state and the existing
    controller IK object. The forward path uses Pinocchio on the validated DAM
    joint target, which lets ``DAMSafetyWrapper.filter_ee()`` return the exact
    safe joint target Isaac should apply.
    """

    def __init__(
        self,
        robot: "Articulation",
        controller: "BaseController",
        robot_config: "BaseRobotConfig",
        device: str,
        *,
        urdf_path: str | None = None,
        ee_frame_name: str | None = None,
    ) -> None:
        self._robot = robot
        self._controller = controller
        self._robot_config = robot_config
        self._device = device
        self._n_arm = len(robot_config.arm_joint_names)
        self.last_ik_joint_positions: np.ndarray | None = None
        self.last_safe_joint_positions: np.ndarray | None = None
        self._pending_gripper_target: float | None = None

        if urdf_path is None and isinstance(robot_config.name, str) and "SO-ARM-101" in robot_config.name:
            urdf_path = default_so101_urdf_path()
        if urdf_path is None:
            raise ValueError("EE-space DAM filtering requires a URDF path for forward kinematics")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"FK URDF not found: {urdf_path}")
        controlled_joint_names = list(robot_config.arm_joint_names)
        resolved_ee_frame_name = ee_frame_name or robot_config.ee_body_name
        if os.path.abspath(urdf_path) == os.path.abspath(default_so101_urdf_path()):
            controlled_joint_names = [
                _SO101_ISAAC_TO_URDF_JOINTS.get(name, name)
                for name in controlled_joint_names
            ]
            resolved_ee_frame_name = _SO101_EE_FRAME_ALIASES.get(
                resolved_ee_frame_name,
                resolved_ee_frame_name,
            )

        self._fk = _PinocchioForwardKinematics(
            urdf_path=urdf_path,
            controlled_joint_names=controlled_joint_names,
            ee_frame_name=resolved_ee_frame_name,
        )

    def inverse_kinematics(
        self,
        target_ee_pose: np.ndarray,
        current_joint_positions: np.ndarray,
    ) -> np.ndarray:
        """Return a full arm+gripper joint proposal for a DAM EE pose."""
        import isaaclab.utils.math as math_utils

        current = np.asarray(current_joint_positions, dtype=np.float64).reshape(-1)
        if current.shape[0] < self._n_arm:
            raise ValueError(f"Expected at least {self._n_arm} current joints, got {current.shape[0]}")

        target_pose_isaac = dam_xyzw_to_isaac_wxyz(target_ee_pose)
        target_pose = torch.as_tensor(
            target_pose_isaac,
            dtype=self._robot.data.joint_pos.dtype,
            device=self._device,
        ).unsqueeze(0)

        ee_pos_w = self._robot.data.body_pos_w[:, self._controller._ee_body_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self._controller._ee_body_idx]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )

        weighted_target = target_pose.clone()
        orientation_weight = getattr(self._controller, "_orientation_weight", 1.0)
        if orientation_weight < 1.0 and hasattr(self._controller, "_slerp_quat"):
            weighted_target[:, 3:7] = self._controller._slerp_quat(
                ee_quat_b,
                target_pose[:, 3:7],
                orientation_weight,
            )

        self._controller._ik_controller.set_command(weighted_target)

        jacobian_w = physx_to_torch(self._robot.root_physx_view.get_jacobians())[
            :, self._controller._jacobi_body_idx, :, self._controller._jacobi_joint_ids
        ]
        base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot, jacobian_w[:, 3:, :])

        joint_pos = torch.as_tensor(
            current[: self._n_arm],
            dtype=self._robot.data.joint_pos.dtype,
            device=self._device,
        ).unsqueeze(0)
        joint_pos_des = self._controller._ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            jacobian_b,
            joint_pos,
        )
        arm_target = joint_pos_des[0].detach().cpu().numpy().astype(np.float64)
        if current.shape[0] > self._n_arm:
            gripper_target = (
                current[self._n_arm]
                if self._pending_gripper_target is None
                else self._pending_gripper_target
            )
            full_target = np.concatenate([arm_target, np.array([gripper_target], dtype=np.float64)])
        else:
            full_target = arm_target
        self.last_ik_joint_positions = full_target.copy()
        return full_target

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Return DAM pose [x,y,z,qx,qy,qz,qw] for a validated joint target."""
        joints = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if joints.shape[0] < self._n_arm:
            raise ValueError(
                f"FK expected at least {self._n_arm} joints, got {joints.shape[0]}"
            )
        pose = self._fk.compute(joints[: self._n_arm])
        self.last_safe_joint_positions = joints.copy()
        return pose

    @property
    def current_ee_pose_dam(self) -> np.ndarray:
        """Current live EE pose converted to DAM [x,y,z,qx,qy,qz,qw] convention."""
        return isaac_wxyz_to_dam_xyzw(self._controller.current_ee_pose[0])

    def set_gripper_target(self, gripper_target: float | None) -> None:
        """Set the gripper joint target to append to the next IK proposal."""
        self._pending_gripper_target = gripper_target

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class SO101CubeMoveIKAbsMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper for SO-ARM-101 Cube Move (pick right → place left).
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose."""
        if env_ids is None:
            env_ids = slice(None)

        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        noise: float | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target pose to action [pos(3), quat(4), gripper(1)]."""
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        (gripper_action,) = gripper_action_dict.values()

        pose_action = torch.cat([target_pos, PoseUtils.quat_from_matrix(target_rot)], dim=0)

        noise_scale = None
        if action_noise_dict is not None:
            noise_scale = list(action_noise_dict.values())[0]
        elif noise is not None:
            noise_scale = noise

        if noise_scale is not None and noise_scale > 0:
            noise_vec = noise_scale * torch.randn_like(pose_action)
            pose_action = pose_action + noise_vec

        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert action to target pose."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        target_pos = action[:, :3]
        target_quat = action[:, 3:7]
        target_rot = PoseUtils.matrix_from_quat(target_quat)
        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions (last dimension)."""
        print(actions[:, -1:])
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals — only 'grasp' for move task."""
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["grasp"] = subtask_terms["grasp_1"][env_ids]
        return signals

# Copyright (c) 2024, Robot Control Library
# SPDX-License-Identifier: BSD-3-Clause

"""鍵盤輸入設備"""

from typing import Tuple

import torch
import carb.input

# Kit 110 下 `omni.appwindow` 是 namespace 子模組，不會被自動載入；
# Se3Keyboard 內部會呼叫 omni.appwindow.get_default_app_window()，
# 先顯式 import 才不會出現 "module 'omni' has no attribute 'appwindow'"。
import omni.appwindow  # noqa: F401

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from .base import BaseInputDevice


class KeyboardInputDevice(BaseInputDevice):
    """鍵盤輸入設備
    
    將鍵盤增量輸入轉換為絕對目標姿態。
    支持連續夾爪控制和場景重置。
    
    控制方式:
    - WASD: XY 平面移動
    - Q/E: Z 軸升降
    - Z/X: Roll 旋轉
    - T/G: Pitch 旋轉
    - C/V: Yaw 旋轉
    - B: 打開夾爪
    - N: 關閉夾爪
    - L: 重置
    """
    
    def __init__(
        self,
        initial_pose: torch.Tensor,
        device: str,
        pos_sensitivity: float = 0.01,  # 高靈敏度
        rot_sensitivity: float = 0.01,
        gripper_speed: float = 0.02,  # 增加夾爪速度
    ):
        """初始化鍵盤輸入設備
        
        Args:
            initial_pose: 初始目標姿態 [pos(3) + quat(4)]
            device: 計算設備
            pos_sensitivity: 位置增量靈敏度
            rot_sensitivity: 旋轉增量靈敏度
            gripper_speed: 夾爪開合速度
        """
        self._device = device
        self._pos_sensitivity = pos_sensitivity
        self._rot_sensitivity = rot_sensitivity
        self._gripper_speed = gripper_speed
        
        # 目標狀態
        self._target_pose = initial_pose.clone()
        self._initial_pose = initial_pose.clone()
        self._gripper_pos = 0.0  # 初始關閉
        
        # 控制標誌
        self._gripper_opening = False
        self._gripper_closing = False
        self._reset_requested = False
        
        # 創建 Se3Keyboard
        self._keyboard = Se3Keyboard(
            cfg=Se3KeyboardCfg(pos_sensitivity=1.0, rot_sensitivity=1.0, gripper_term=True)
        )
        
        # 設定自定義事件處理器
        self._original_handler = self._keyboard._on_keyboard_event
        self._keyboard._on_keyboard_event = self._custom_keyboard_handler
        
        self._print_controls()
    
    def _print_controls(self) -> None:
        """打印控制說明"""
        print("\n" + "=" * 50)
        print("鍵盤控制")
        print("=" * 50)
        print("  位置控制:")
        print("    WASD - XY 平面移動")
        print("    Q/E  - Z 軸升降")
        print("  旋轉控制:")
        print("    Z/X  - Roll (繞X軸)")
        print("    T/G  - Pitch (繞Y軸)")
        print("    C/V  - Yaw (繞Z軸)")
        print("  夾爪控制:")
        print("    B    - 打開夾爪")
        print("    N    - 關閉夾爪")
        print("  其他:")
        print("    L    - 重置")
        print("=" * 50)
    
    def _custom_keyboard_handler(self, event, *args, **kwargs):
        """自定義鍵盤事件處理器"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "B":
                self._gripper_opening = True
            elif event.input.name == "N":
                self._gripper_closing = True
            elif event.input.name == "L":
                self._reset_requested = True
        
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name == "B":
                self._gripper_opening = False
            elif event.input.name == "N":
                self._gripper_closing = False
        
        return self._original_handler(event, *args, **kwargs)
    
    def update(self) -> Tuple[torch.Tensor, float, bool]:
        """更新輸入狀態
        
        Returns:
            Tuple[torch.Tensor, float, bool]:
                - target_pose: 絕對目標姿態
                - gripper_pos: 夾爪位置 [0, 1]
                - reset_requested: 是否請求重置
        """
        # 檢查重置請求
        reset = self._reset_requested
        if reset:
            self._target_pose = self._initial_pose.clone()
            self._gripper_pos = 0.0  # 重置為關閉
            self._reset_requested = False
            return self._target_pose, self._gripper_pos, True
        
        # 處理連續夾爪控制
        if self._gripper_opening:
            self._gripper_pos = min(self._gripper_pos + self._gripper_speed, 1.0)
        if self._gripper_closing:
            self._gripper_pos = max(self._gripper_pos - self._gripper_speed, 0.0)
        
        # 獲取鍵盤輸入（Se3Keyboard 已經乘過靈敏度了）
        command = self._keyboard.advance().to(self._device)
        # Se3Keyboard 輸出: [x, y, z, rx, ry, rz, gripper]
        # 我們再乘一個額外的縮放因子
        delta_pos = command[:3] * self._pos_sensitivity
        delta_rot = command[3:6] * self._rot_sensitivity
        
        # 調試：顯示按鍵輸入
        if torch.any(delta_pos != 0):
            print(f"[DEBUG] delta_pos: x={delta_pos[0]:.5f}, y={delta_pos[1]:.5f}, z={delta_pos[2]:.5f}")
        
        # 更新目標位置
        self._target_pose[:, 0:3] += delta_pos.unsqueeze(0)
        
        # 更新目標旋轉
        if torch.any(delta_rot != 0):
            self._apply_rotation_delta(delta_rot)
        
        return self._target_pose.clone(), self._gripper_pos, False
    
    def _apply_rotation_delta(self, delta_rot: torch.Tensor) -> None:
        """應用旋轉增量"""
        from scipy.spatial.transform import Rotation as R
        
        current_quat = self._target_pose[0, 3:7].cpu().numpy()
        # Isaac 格式 (w, x, y, z) -> scipy 格式 (x, y, z, w)
        current_rot = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
        
        delta_rotation = R.from_rotvec(delta_rot.cpu().numpy())
        new_rot = delta_rotation * current_rot
        
        quat_xyzw = new_rot.as_quat()  # scipy 返回 (x, y, z, w)
        # 轉回 Isaac 格式 (w, x, y, z)
        self._target_pose[:, 3:7] = torch.tensor(
            [[quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]],
            device=self._device
        )
    
    @property
    def target_pose(self) -> torch.Tensor:
        """當前目標姿態"""
        return self._target_pose.clone()
    
    @property
    def gripper_pos(self) -> float:
        """當前夾爪位置"""
        return self._gripper_pos
    
    def reset_target(self, pose: torch.Tensor) -> None:
        """重置目標姿態
        
        Args:
            pose: 新的目標姿態
        """
        self._target_pose = pose.clone()
        self._initial_pose = pose.clone()
    
    def sync_to_actual(self, actual_pose: torch.Tensor) -> None:
        """同步目標到實際姿態（當目標不可達時）
        
        Args:
            actual_pose: 實際 EE 姿態
        """
        self._target_pose = actual_pose.clone()

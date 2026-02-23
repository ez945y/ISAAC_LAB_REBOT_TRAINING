import torch
import numpy as np

# 5 個手臂關節 + 1 個夾爪 (與 Isaac Sim USD 中的關節名稱對應)
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"

# 每個關節的物理限制 (radians)，順序對應 [5 arm joints + gripper]
# 來源：真實機器人關節限制表
JOINT_LIMITS = {
    #                   (lower,  upper)
    "shoulder_pan":    (-1.8243,   1.8243),
    "shoulder_lift":   (-1.7691,   1.7691),
    "elbow_flex":      (-1.6026,   1.6026),
    "wrist_flex":      (-1.8067,   1.8067),
    "wrist_roll":      (-3.0741,   3.0741),
    "gripper":         (0.0,   1.7453),
}

# 預計算各關節的 lower / upper 向量，順序: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
_JOINT_ORDER = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
JOINT_LOWER = torch.tensor([JOINT_LIMITS[j][0] for j in _JOINT_ORDER])  # [6]
JOINT_UPPER = torch.tensor([JOINT_LIMITS[j][1] for j in _JOINT_ORDER])  # [6]

def normalize_joints(values):
    """
    Isaac Sim 弧度 → LeRobot 正規化格式
    支援 torch.Tensor 或 np.ndarray
    輸入: [..., 6] radians
    輸出: [..., 6] LeRobot 格式 (-100~100 for arm, 0~100 for gripper)
    """
    # 自動判斷輸入類型，並轉成 torch 或 numpy
    is_torch = isinstance(values, torch.Tensor)
    device = values.device if is_torch else None

    # 轉成 numpy 方便統一處理（如果原本是 torch，先 cpu().numpy()）
    if is_torch:
        values_np = values.cpu().numpy()
    else:
        values_np = values  # 已經是 numpy

    # 轉成 float32 避免精度問題
    values_np = values_np.astype(np.float32)

    # lower / upper 轉 numpy
    lower_np = JOINT_LOWER.cpu().numpy().astype(np.float32)
    upper_np = JOINT_UPPER.cpu().numpy().astype(np.float32)

    result = np.zeros_like(values_np)

    # Arm joints: radians → -100 ~ 100
    arm_range = upper_np[:5] - lower_np[:5]
    result[..., :5] = ((values_np[..., :5] - lower_np[:5]) / arm_range) * 200.0 - 100.0

    # Gripper: radians → 0 ~ 100
    gripper_range = upper_np[5:6] - lower_np[5:6]
    result[..., 5:6] = ((values_np[..., 5:6] - lower_np[5:6]) / gripper_range) * 100.0

    if is_torch:
        return torch.from_numpy(result).to(device=device, dtype=torch.float32)
    else:
        return result


def denormalize_joints(norm_values: torch.Tensor) -> torch.Tensor:
    """
    LeRobot 正規化 → Isaac Sim 弧度（你原本的函式，微調後更清晰）
    """
    lower = JOINT_LOWER.to(norm_values.device)
    upper = JOINT_UPPER.to(norm_values.device)
    result = torch.zeros_like(norm_values)

    # Arm: -100~100 → radians
    arm_norm_01 = (norm_values[..., :5] + 100.0) / 200.0
    result[..., :5] = lower[:5] + arm_norm_01 * (upper[:5] - lower[:5])

    # Gripper: 0~100 → radians
    gripper_norm_01 = norm_values[..., 5:6] / 100.0
    result[..., 5:6] = lower[5:6] + gripper_norm_01 * (upper[5:6] - lower[5:6])

    return result
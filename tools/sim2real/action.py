import torch
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

def normalize_joints(rad_values: torch.Tensor) -> torch.Tensor:
    """
    Isaac Sim 弧度 → LeRobot 正規化格式
    輸入: [..., 6] radians
    輸出: [..., 6] LeRobot 格式 (-100~100 for arm, 0~100 for gripper)
    """
    lower = JOINT_LOWER.to(rad_values.device)
    upper = JOINT_UPPER.to(rad_values.device)

    result = torch.zeros_like(rad_values)

    # Arm joints: radians → -100 ~ 100
    arm_range = upper[:5] - lower[:5]
    result[..., :5] = ((rad_values[..., :5] - lower[:5]) / arm_range) * 200.0 - 100.0

    # Gripper: radians → 0 ~ 100
    gripper_range = upper[5:6] - lower[5:6]
    result[..., 5:6] = ((rad_values[..., 5:6] - lower[5:6]) / gripper_range) * 100.0

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
import torch
import numpy as np
# 確保導入你的 IK Solver
from ik_solver import SO101OfficialIKSolver

def verify_relative_ik():
    device = "cpu" # 測試建議先用 CPU 排除驅動問題
    ik_bridge = SO101OfficialIKSolver("so101_new_calib.urdf", device=device)
    
    # 1. 定義測試姿勢 (單位：度)
    # 我們測試：當機器人在 [0,0,0,0,0] 時，給予不同位移
    current_angles = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    
    print(f"當前關節角度 (Seed): {current_angles.tolist()}")
    print("-" * 70)
    print(f"{'位移測試 (Delta Pose)':<25} | {'計算出的新角度 (Deg)':<35} | {'角度變化量'}")
    print("-" * 70)

    # 測試案例 (相對位移: dx, dy, dz, dr, dp, dyw)
    test_deltas = {
        "原地不動 (Zero Delta)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "向前移 1cm (+1cm X)":  [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        "向上移 1cm (+1cm Z)":  [0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
        "向左移 1cm (+1cm Y)":  [0.0, 0.01, 0.0, 0.0, 0.0, 0.0],
    }

    for name, delta_list in test_deltas.items():
        delta_tensor = torch.tensor(delta_list, device=device)
        
        # 執行 solve (注意：你的 solve 內部會把 current_angles 從 Deg 轉 Rad)
        try:
            target_angles = ik_bridge.solve(current_angles, delta_tensor)
            
            target_list = [round(a.item(), 3) for a in target_angles]
            diff_list = [round((target_angles[i] - current_angles[i]).item(), 3) for i in range(5)]
            
            print(f"{name:<25} | {str(target_list):<35} | {str(diff_list)}")
        except Exception as e:
            print(f"{name:<25} | 錯誤: {e}")

    # --- 額外驗證 FK 是否直覺 ---
    print("\n[FK 方向性驗證]")
    # 測試：如果肩膀抬高 10 度，Z 應該要增加還是減少？
    plus_lift = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0], device=device)
    pos_zero = ik_bridge.forward_kinematics(current_angles)
    pos_lift = ik_bridge.forward_kinematics(plus_lift)
    
    z_diff = pos_lift[2] - pos_zero[2]
    print(f"當 Lift 增加 10 度時，Z 座標變化了: {z_diff.item():.4f} 米")
    if z_diff > 0:
        print("-> 結論：在模型中，Lift 增加 = 手臂向上抬")
    else:
        print("-> 結論：在模型中，Lift 增加 = 手臂向下壓")
import math
import time

def test_circular_trajectory():
    device = "cpu" # 測試建議先用 CPU 排除驅動問題
    ik_bridge = SO101OfficialIKSolver("so101_new_calib.urdf", device=device)
    print("\n" + "="*50)
    print("開始圓形路徑測試 (X-Z 平面)")
    print("="*50)
    
    device = "cpu"
    current_angles = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    
    center_x, center_z = 0.0, 0.0  # 這是相對於當前位置的偏移
    radius = 0.025  # 2.5 cm
    steps = 20      # 採樣點數量
    
    for i in range(steps + 1):
        # 計算圓周座標 (極座標轉直角座標)
        theta = 2.0 * math.pi * i / steps
        dx = radius * math.cos(theta) - radius # 從 0 開始，所以減去半徑偏移
        dz = radius * math.sin(theta)
        
        # 構造 6 軸 Delta Pose (dx, dy, dz, roll, pitch, yaw)
        delta_pose = torch.tensor([dx, 0.0, dz, 0.0, 0.0, 0.0], device=device)
        
        # 求解
        target_angles = ik_bridge.solve(current_angles, delta_pose)
        
        # 打印進度 (只看前三個關鍵關節)
        ang_list = [round(a.item(), 2) for a in target_angles[:3]]
        print(f"Step {i:02d} | Target Delta X:{dx:+.3f} Z:{dz:+.3f} | Joint Angles: {ang_list}")
        
        # (選擇性) 如果要在實體機測試，這裡可以下發指令給馬達
        # time.sleep(0.1)

# 在 main 呼叫
if __name__ == "__main__":
    verify_relative_ik()
    test_circular_trajectory()


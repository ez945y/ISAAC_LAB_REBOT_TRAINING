import torch
import numpy as np
import pinocchio as pin

class SO101OfficialIKSolver:
    def __init__(self, urdf_path, device="cuda"):
        self.device = device
        # 1. 讀取並構建 Reduced Model (保持 5 軸)
        full_model = pin.buildModelFromUrdf(urdf_path)
        controlled_joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
        
        all_joint_names = [full_model.names[i] for i in range(1, full_model.njoints)]
        joints_to_lock = [full_model.getJointId(name) for name in all_joint_names if name not in controlled_joints]
        
        q_reference = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, joints_to_lock, q_reference)
        
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("gripper_link")
        
        from differential_ik import DifferentialIKController
        from differential_ik_cfg import DifferentialIKControllerCfg
        self.cfg = DifferentialIKControllerCfg(ik_method="dls_5dof", use_relative_mode=True)
        self.controller = DifferentialIKController(self.cfg, num_envs=1, device=device)
        
    def solve(self, current_joints_tensor, action_ee_tensor):
        """
        Args:
            current_joints_tensor: (5,) Tensor, 單位: 弧度 (Isaac Lab 預設)
            action_ee_tensor: (6,) Tensor, Pose delta
        """
        # 確保在 CPU 上轉換，Pinocchio 只吃 numpy
        q_curr_np = current_joints_tensor.detach().cpu().numpy().astype(np.float32)
        
        # ❗ 修正點：機器人 API 使用角度 (Degrees)，Pinocchio 內部需要弧度 (Radians)
        q_rad = np.deg2rad(q_curr_np) 
        
        # Pinocchio 運動學計算
        pin.forwardKinematics(self.model, self.data, q_rad)
        pin.updateFramePlacements(self.model, self.data)
        
        # 獲取位姿並轉回 Tensor 給官方控制器
        oMf = self.data.oMf[self.ee_frame_id]
        ee_pos = torch.from_numpy(oMf.translation).to(self.device).float().unsqueeze(0)
        q_pin = pin.Quaternion(oMf.rotation)
        ee_quat = torch.tensor([q_pin.w, q_pin.x, q_pin.y, q_pin.z], device=self.device).float().unsqueeze(0)
        
        # 獲取 6x5 雅可比
        jac_np = pin.computeFrameJacobian(self.model, self.data, q_rad, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        jacobian = torch.from_numpy(jac_np).to(self.device).float().unsqueeze(0)
        
        # --- 調用官方控制器 (全 Tensor 運作) ---
        q_curr_torch = torch.from_numpy(q_rad).to(self.device).float().unsqueeze(0)
        
        # ❗ 這裡直接使用傳入的 action_ee_tensor
        self.controller.set_command(action_ee_tensor.unsqueeze(0), ee_pos, ee_quat)
        target_q_rad = self.controller.compute(ee_pos, ee_quat, jacobian, q_curr_torch)
        
        # 返回 Tensor 給 main.py 
        # ❗ 修正點：將解算結果轉回角度 (Degrees) 給機器人馬達
        return torch.rad2deg(target_q_rad.squeeze())
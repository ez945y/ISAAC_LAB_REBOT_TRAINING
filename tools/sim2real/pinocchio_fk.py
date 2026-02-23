import numpy as np
import pinocchio as pin

class PinocchioFK:
    """Pinocchio-based forward kinematics for SO-ARM-101 (5-DOF reduced model)."""

    def __init__(self, urdf_path: str):
        full_model = pin.buildModelFromUrdf(urdf_path)
        controlled_joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']

        all_joint_names = [full_model.names[i] for i in range(1, full_model.njoints)]
        joints_to_lock = [
            full_model.getJointId(name)
            for name in all_joint_names
            if name not in controlled_joints
        ]

        q_reference = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, joints_to_lock, q_reference)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("gripper_link")

    def compute(self, q_rad_np: np.ndarray):
        """
        Args:
            q_rad_np: (5,) numpy array, arm joint positions in radians

        Returns:
            ee_pos: (3,) numpy array
            ee_quat_wxyz: (4,) numpy array [w, x, y, z]
            jacobian: (6, 5) numpy array
        """
        pin.forwardKinematics(self.model, self.data, q_rad_np)
        pin.updateFramePlacements(self.model, self.data)

        oMf = self.data.oMf[self.ee_frame_id]
        ee_pos = oMf.translation.copy()
        q_pin = pin.Quaternion(oMf.rotation)
        ee_quat_wxyz = np.array([q_pin.w, q_pin.x, q_pin.y, q_pin.z])

        jac = pin.computeFrameJacobian(
            self.model, self.data, q_rad_np,
            self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        return ee_pos, ee_quat_wxyz, jac
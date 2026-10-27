"""One sensor adapter shared by recurrent training and evaluation."""
from __future__ import annotations

import numpy as np
import torch

try:
    from ..student_vision import preprocess_student_rgb
except ImportError:  # python config_run/main_separate.py
    from student_vision import preprocess_student_rgb

IMU_BODY_NAME = "imu_in_pelvis"


def world_to_body(vector, quaternion_wxyz):
    """Inverse quaternion rotation for MetaSim's WXYZ body orientations."""
    quat = torch.nn.functional.normalize(quaternion_wxyz, dim=-1)
    xyz = quat[..., 1:]
    cross = torch.linalg.cross(xyz, vector, dim=-1)
    return vector - 2 * quat[..., :1] * cross + 2 * torch.linalg.cross(xyz, cross, dim=-1)


class StudentInputs:
    def __init__(self, base, env, device, camera_name="camera0", joint_names=None,
                 image_size=(128, 128)):
        self.handler = base.env.handler
        self.robot_name = env.robot_name
        self.device = torch.device(device)
        self.camera_name = camera_name
        self.image_size = tuple(image_size)
        self.joint_names = list(joint_names if joint_names is not None else env.robot_joint_names)
        robot = self.handler.get_states().robots[self.robot_name]
        actual_names = list(robot.joint_names)
        if len(set(self.joint_names)) != len(self.joint_names) or set(self.joint_names) != set(actual_names):
            raise ValueError("Recurrent student joint names must cover every simulator joint exactly once")
        self.joint_indices = torch.tensor([actual_names.index(name) for name in self.joint_names],
                                          device=robot.joint_pos.device)
        if IMU_BODY_NAME not in robot.body_names:
            raise ValueError(f"Recurrent student requires IMU body {IMU_BODY_NAME!r} in robot.body_names")
        self.imu_index = list(robot.body_names).index(IMU_BODY_NAME)

    def read(self, previous_actions):
        states = self.handler.get_states()
        robot = states.robots[self.robot_name]
        if getattr(self.handler.scenario, "defer_camera_rendering", False):
            cameras = self.handler.get_camera_states()
        else:
            cameras = states.cameras
        if self.camera_name not in cameras:
            raise ValueError(f"Recurrent student camera {self.camera_name!r} is unavailable")
        rgb = cameras[self.camera_name].rgb
        if rgb.ndim != 4 or rgb.shape[-1] < 3 or min(rgb.shape[1:3]) < 32:
            raise ValueError(f"Expected NHWC RGB image >= 32x32, got {tuple(rgb.shape)}")
        images = preprocess_student_rgb(rgb, device=self.device, output_size=self.image_size)
        imu_body = robot.body_state[:, self.imu_index]
        orientation = imu_body[:, 3:7]
        gravity = torch.zeros_like(imu_body[:, :3])
        gravity[:, 2] = -1
        imu = torch.cat((world_to_body(imu_body[:, 10:13], orientation),
                         world_to_body(gravity, orientation)), dim=-1)
        return (
            images,
            robot.joint_pos.index_select(1, self.joint_indices).to(self.device, dtype=torch.float32),
            robot.joint_vel.index_select(1, self.joint_indices).to(self.device, dtype=torch.float32),
            imu.to(self.device, dtype=torch.float32),
            previous_actions,
        )


class RolloutState:
    """Per-environment recurrent state, including the selected/clipped command."""
    def __init__(self, student, num_envs):
        self.hidden = student.initial_hidden(num_envs)
        self.previous_actions = self.hidden.new_zeros(num_envs, student.num_actions)
        self.reset_mask = torch.ones(num_envs, dtype=torch.bool, device=self.hidden.device)

    def after_step(self, hidden, executed_actions, dones):
        mask = torch.as_tensor(np.asarray(dones), device=self.hidden.device, dtype=torch.bool)
        self.hidden = hidden.detach().masked_fill(mask.view(1, -1, 1), 0)
        self.previous_actions = torch.as_tensor(
            executed_actions, device=self.hidden.device, dtype=torch.float32
        ).clone().masked_fill(mask[:, None], 0)
        self.reset_mask = mask

"""Focused tests for the six-policy ChairMan DAgger integration."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
import yaml

from config_run.separate_dagger import _new_student, _student_checkpoints, _student_inputs
from config_run.dagger_vp.dagger_trainer import DAggerBuffer, train_dagger_step


class _CameraHandler:
    def __init__(self, rgb):
        self._states = SimpleNamespace(
            cameras={"camera0": SimpleNamespace(rgb=rgb)}
        )

    def get_states(self):
        return self._states


class ChairmanSeparateDaggerTest(unittest.TestCase):
    def test_student_inputs_include_all_joints_and_exclude_task_features(self):
        rgb = torch.arange(2 * 4 * 5 * 4, dtype=torch.uint8).reshape(2, 4, 5, 4)
        metasim_env = SimpleNamespace(
            env=SimpleNamespace(handler=_CameraHandler(rgb))
        )
        observation = np.arange(2 * 50, dtype=np.float32).reshape(2, 50)

        images_uint8, images, joints = _student_inputs(
            metasim_env, observation, "cpu", num_joints=43, camera_name="camera0"
        )

        self.assertEqual(images_uint8.shape, (2, 3, 4, 5))
        self.assertTrue(torch.equal(
            images_uint8, rgb[..., :3].permute(0, 3, 1, 2)
        ))
        self.assertTrue(torch.allclose(images, images_uint8.float() / 255.0))
        self.assertEqual(joints.shape, (2, 43))
        self.assertTrue(torch.equal(joints, torch.from_numpy(observation[:, :43])))

    def test_student_and_buffer_train_with_43_joints_and_34_actions(self):
        env = SimpleNamespace(
            action_space=SimpleNamespace(shape=(34,)),
            robot_joint_names=tuple(f"joint_{i}" for i in range(43)),
        )
        student, num_actions, num_joints = _new_student(env, "cpu")
        self.assertEqual((num_actions, num_joints), (34, 43))
        buffer = DAggerBuffer(
            max_samples=2, img_shape=(3, 128, 128),
            num_joints=num_joints, num_actions=num_actions, device="cpu",
        )
        buffer.add_batch(
            torch.zeros(2, 3, 128, 128, dtype=torch.uint8),
            torch.randn(2, 43), torch.randn(2, 34),
        )
        optimizer = torch.optim.Adam(student.parameters())
        loss = train_dagger_step(student, optimizer, buffer, batch_size=2)
        self.assertTrue(np.isfinite(loss))
        self.assertIsNotNone(student.joint_head[0].weight.grad)

    def test_student_checkpoints_are_numeric_and_ignore_training_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "student_model_step_100.pth",
                "student_model_step_20.pth",
                "student_model_step_20.pth.training.pth",
                "student_model_final.pth",
                "unrelated.pth",
            ):
                (root / name).touch()

            found = _student_checkpoints(root)

        self.assertEqual([path.name for _, path in found], [
            "student_model_step_20.pth",
            "student_model_step_100.pth",
            "student_model_final.pth",
        ])

    def test_configs_define_six_policy_teacher_and_camera(self):
        root = Path("config_run/configs/chairman_separate")
        expected_modes = {
            "train_dagger.yaml": "train_dagger",
            "eval_dagger.yaml": "eval_dagger",
            "eval_dagger_video.yaml": "eval_dagger_video",
        }
        expected_policies = {
            "waist", "right_arm", "left_arm", "right_fingers",
            "left_fingers", "direction",
        }

        for filename, mode in expected_modes.items():
            with self.subTest(filename=filename):
                config = yaml.safe_load(
                    (root / filename).read_text(encoding="utf-8")
                )
                self.assertEqual(config["task"], "chairmanseparate")
                self.assertEqual(config["train_or_eval"], mode)
                self.assertEqual(set(config["policies"]), expected_policies)
                self.assertEqual(
                    config["cameras"]["camera1"]["params"]["name"],
                    "camera0",
                )


if __name__ == "__main__":
    unittest.main()


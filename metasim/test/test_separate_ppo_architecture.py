import unittest
from pathlib import Path

import yaml

from config_run.separate_ppo_trainer import _policy_kwargs


CONFIG_DIR = Path("config_run/configs/chairman_separate")


class SeparatePPOArchitectureTest(unittest.TestCase):
    def test_each_policy_can_override_actor_and_critic_layers(self):
        config = yaml.safe_load((CONFIG_DIR / "train_ppo.yaml").read_text())
        expected = {
            "waist": {"pi": [256, 128, 64], "vf": [256, 128, 64]},
            "right_arm": {"pi": [128, 64, 64], "vf": [128, 128, 64]},
            "left_arm": {"pi": [128, 64, 64], "vf": [128, 128, 64]},
            "right_fingers": {"pi": [64, 64], "vf": [128, 64]},
            "left_fingers": {"pi": [64, 64], "vf": [128, 64]},
            "direction": {"pi": [128, 128, 64], "vf": [256, 128, 64]},
        }

        for policy, architecture in expected.items():
            self.assertEqual(
                _policy_kwargs(config, policy)["net_arch"], architecture
            )

    def test_train_and_resume_declare_the_same_architectures(self):
        train = yaml.safe_load((CONFIG_DIR / "train_ppo.yaml").read_text())
        resume = yaml.safe_load(
            (CONFIG_DIR / "load_and_train_ppo.yaml").read_text()
        )

        for policy in train["policies"]:
            self.assertEqual(
                _policy_kwargs(train, policy)["net_arch"],
                _policy_kwargs(resume, policy)["net_arch"],
            )

    def test_invalid_layer_sizes_are_rejected(self):
        config = {
            "net_arch_pivf": True,
            "policies": {
                "waist": {
                    "net_arch_pi": [128, 0],
                    "net_arch_vf": [128, 64],
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "positive integers"):
            _policy_kwargs(config, "waist")


if __name__ == "__main__":
    unittest.main()

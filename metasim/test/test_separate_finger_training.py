import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from config_run.separate_finger_training import (
    FingerPoseSupervision, constrain_action_std, reset_finger_actor,
)
from metasim.utils.chairman_grasp import STAGE2_FINGER_JOINT_TARGETS_BY_SIDE


def _env():
    # Deliberately use actuator order rather than target dictionary order.
    names = list(STAGE2_FINGER_JOINT_TARGETS_BY_SIDE["left"])
    names[3:7] = names[5:7] + names[3:5]
    action_space = spaces.Box(-1.75, 1.75, shape=(7,), dtype=np.float32)
    return SimpleNamespace(
        torch_device="cpu", action_space=action_space, action_names=names,
        sim_joint_names=names, policy_action_indices={"left_fingers": tuple(range(7))},
        policy_observation_indices={"left_fingers": tuple(range(14))},
        policy_uses_action_history={"left_fingers": True},
        observation_groups={"joint_positions": tuple(range(7)), "stage": tuple(range(7, 14))},
    )


class SeparateFingerTrainingTest(unittest.TestCase):
    def test_targets_preserve_actuator_order_and_stage_layout(self):
        env = _env()
        objective = FingerPoseSupervision(env, "left_fingers")
        obs = torch.zeros(7, 28)
        obs[:, 7:14] = torch.eye(7)
        target = objective.targets(obs)
        closed = torch.tensor([
            STAGE2_FINGER_JOINT_TARGETS_BY_SIDE["left"][n] for n in env.action_names
        ])
        self.assertTrue(torch.equal(target[2], closed))
        self.assertTrue(torch.equal(target[3], closed))
        self.assertTrue(torch.equal(target[[0, 1, 4, 5, 6]], torch.zeros(5, 7)))

    def test_out_of_bounds_actor_still_receives_corrective_gradient(self):
        objective = FingerPoseSupervision(_env(), "left_fingers")
        mean = torch.nn.Parameter(torch.full((7,), 35.0))
        policy = SimpleNamespace(get_distribution=lambda obs: SimpleNamespace(
            distribution=SimpleNamespace(mean=mean.expand(len(obs), -1))))
        obs = torch.zeros(8, 28)
        obs[:, 9] = 1.0  # stage 2
        loss = objective.loss(policy, obs, augmented=False)
        loss.backward()
        self.assertTrue(torch.all(mean.grad > 0))
        self.assertTrue(torch.isfinite(mean.grad).all())

    def test_resume_resets_std_and_clears_only_its_optimizer_momentum(self):
        log_std = torch.nn.Parameter(torch.full((7,), math.log(64.0)))
        actor_weight = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([log_std, actor_weight])
        (log_std.sum() + actor_weight.sum()).backward()
        optimizer.step()
        model = SimpleNamespace(policy=SimpleNamespace(log_std=log_std, optimizer=optimizer))
        spec = {"action_std_min": 0.05, "action_std_max": 0.25, "log_std_init": -1.6}
        constrain_action_std(model, spec, reset=True)
        self.assertTrue(torch.allclose(log_std.exp(), torch.full((7,), math.exp(-1.6))))
        self.assertNotIn(log_std, optimizer.state)
        self.assertIn(actor_weight, optimizer.state)
        with torch.no_grad():
            log_std.fill_(10.0)
        constrain_action_std(model, spec)
        self.assertTrue(torch.all(log_std.exp() <= 0.250001))

    def test_fresh_actor_learns_open_and_close_from_different_histories(self):
        torch.manual_seed(7)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            env = _env()
            objective = FingerPoseSupervision(env, "left_fingers", stage_only=True)
            policy = ActorCriticPolicy(
                spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32),
                env.action_space, lambda _: 0.003,
                net_arch={"pi": [32, 32], "vf": [32]},
            )
            obs = torch.zeros(128, 28)
            objective.constrain_actor(policy)
            for _ in range(400):
                loss = objective.loss(policy, obs)
                policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                policy.optimizer.step()
                objective.constrain_actor(policy)
            heldout = objective.augment(torch.zeros(1024, 28))
            with torch.no_grad():
                predicted = policy.get_distribution(heldout).distribution.mean
            error = (predicted - objective.targets(heldout)).abs()
            self.assertLess(float(error.mean()), 0.03)
            self.assertLess(float(torch.quantile(error, 0.99)), 0.10)
        finally:
            torch.set_num_threads(old_threads)

    def test_reset_changes_only_actor_not_critic(self):
        env = _env()
        policy = ActorCriticPolicy(
            spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32),
            env.action_space, lambda _: 0.003,
            net_arch={"pi": [32, 32], "vf": [32]},
        )
        before = {name: p.detach().clone() for name, p in policy.named_parameters()}
        reset_finger_actor(SimpleNamespace(policy=policy))
        changed = []
        for name, parameter in policy.named_parameters():
            if not torch.equal(parameter, before[name]):
                changed.append(name)
                self.assertTrue(
                    name.startswith("mlp_extractor.policy_net.")
                    or name.startswith("action_net."), name)
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()

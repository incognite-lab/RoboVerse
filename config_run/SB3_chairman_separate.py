"""Six-policy ChairMan wrapper with the pretrained leg policy unchanged."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from metasim.utils.chair_navigation import CHAIR_FINAL_DISTANCE, chair_back_direction_xy
from metasim.utils.humanoid_robot_util import neck_height_tensor
from metasim.utils.chairman_grasp import STAGE2_FINGER_JOINT_TARGETS
try:
    from .SB3_chairman_multi_env import StableBaseline3VecEnv as _MultiEnv
except ImportError:
    from SB3_chairman_multi_env import StableBaseline3VecEnv as _MultiEnv

POLICY_NAMES = ("waist", "right_arm", "left_arm", "right_fingers", "left_fingers", "direction")
ACTION_HISTORY_LENGTH = 2
DEFAULT_OBSERVATIONS = {
    "waist": ("joint_positions", "robot_body_states", "pelvis_velocity", "chair_geometry", "stage"),
    "right_arm": ("joint_positions", "right_hand_task", "pelvis_velocity", "chair_geometry", "stage"),
    "left_arm": ("joint_positions", "left_hand_task", "pelvis_velocity", "chair_geometry", "stage"),
    "right_fingers": ("joint_positions", "right_hand_task", "fingertip_forces", "stage"),
    "left_fingers": ("joint_positions", "left_hand_task", "fingertip_forces", "stage"),
    "direction": ("joint_positions", "pelvis_velocity", "chair_geometry", "previous_walk_command", "stage"),
}

class PolicySpaceEnv(gym.Env):
    """Space-only object used to construct an SB3 policy."""
    def __init__(self, parent, name):
        super().__init__()
        self.num_envs = parent.num_envs
        self.observation_space = parent.policy_observation_spaces[name]
        self.action_space = parent.policy_action_spaces[name]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError(
            "PolicySpaceEnv is space-only; SeparatePPOTrainer owns stepping"
        )

class StableBaseline3VecEnv(_MultiEnv):
    """Partition one physical action into six independently trained policies."""
    POLICY_NAMES = POLICY_NAMES

    def _collect_step_diagnostics(self, infos, dones, timeout):
        task = self.env.env.handler.task
        if not getattr(task, "collect_stage_diagnostics", False):
            return
        events = task.completed_stage_events
        ids = torch.nonzero(dones | (events >= 0), as_tuple=True)[0]
        if not ids.numel():
            return
        masks = {**task.failure_masks, "global_timeout": timeout}
        reason_flags = torch.stack([mask.to(ids.device)[ids] for mask in masks.values()], dim=1).cpu().tolist()
        diagnostics = self.stage_endpoint_geometry(ids)
        for (index, geometry), flags in zip(diagnostics.items(), reason_flags):
            infos[index]["stage_endpoint_geometry"] = geometry
            infos[index]["failure_reasons"] = tuple(reason for reason, flagged in zip(masks, flags) if flagged)

    def stage_endpoint_geometry(self, env_ids):
        """Measure stage endpoints from the current cached simulator state."""
        handler = self.env.env.handler
        states = handler.get_states()
        robot = states.robots[self.robot_name]
        chair = states.objects["chair"]
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=robot.body_state.device)
        pelvis = robot.body_state[ids, robot.body_names.index("pelvis")]
        base = chair.body_state[ids, chair.body_names.index("base_link")]
        offset = pelvis[:, :2] - base[:, :2]
        back = chair_back_direction_xy(base[:, 3:7])
        lateral = torch.stack((back[:, 1], -back[:, 0]), dim=1)
        left = robot.body_state[ids, robot.body_names.index("left_endeffector"), :3]
        right = robot.body_state[ids, robot.body_names.index("endeffector"), :3]
        left_target = chair.body_state[ids, chair.body_names.index("target_hand_left"), :3]
        right_target = chair.body_state[ids, chair.body_names.index("target_hand_right"), :3]
        joints = list(robot.joint_names)
        finger_ids = [joints.index(name) for name in STAGE2_FINGER_JOINT_TARGETS]
        fingers = robot.joint_pos[ids][:, finger_ids]
        arm_ids = [i for i, name in enumerate(joints) if "shoulder_" in name or "elbow_joint" in name]
        def norm(value):
            return torch.linalg.vector_norm(value, dim=1)

        finger_targets = fingers.new_tensor(list(STAGE2_FINGER_JOINT_TARGETS.values()))
        values = {
            "pelvis_chair_xy_m": norm(offset),
            "pelvis_chair_lateral_m": (offset * lateral).sum(dim=1),
            "pelvis_chair_back_m": (offset * back).sum(dim=1),
            "neck_height_m": neck_height_tensor(states, self.robot_name)[ids],
            "walk_goal_error_m": norm(offset - CHAIR_FINAL_DISTANCE * back),
            "left_hand_target_error_m": norm(left - left_target),
            "right_hand_target_error_m": norm(right - right_target),
            "finger_grasp_error_rad": (fingers - finger_targets).abs().amax(dim=1),
            "chair_start_error_m": norm(base[:, :3] - base.new_tensor([0.75, 0.0, 0.1])),
            "chair_pull_goal_error_m": norm(base[:, :3] - base.new_tensor([-0.25, 0.0, 0.1])),
            "chair_speed_m_s": norm(base[:, 7:10]),
            "pelvis_speed_xy_m_s": norm(pelvis[:, 7:9]),
            "finger_open_error_rad": fingers.abs().amax(dim=1),
            "arm_rest_error_rad": robot.joint_pos[ids][:, arm_ids].abs().amax(dim=1),
        }
        measurements = torch.stack(tuple(values.values()), dim=1).cpu().tolist()
        return {index: dict(zip(values, row)) for index, row in zip(ids.tolist(), measurements)}

    def __init__(self, env):
        super().__init__(env)
        configured = getattr(env.scenario.task, "separate_policy_specs", {}) or {}
        unknown = sorted(set(configured) - set(POLICY_NAMES))
        if unknown:
            raise ValueError(f"Unknown separate policy names: {unknown}")
        self.policy_specs = {name: dict(configured.get(name, {})) for name in POLICY_NAMES}
        self.observation_groups = self._build_observation_groups()
        self.action_groups = self._build_action_groups()
        self.policy_action_indices = self._resolve_action_indices()
        self.policy_observation_indices = self._resolve_observation_indices()
        self.policy_action_spaces, self.policy_observation_spaces = {}, {}
        for name in POLICY_NAMES:
            ids = list(self.policy_action_indices[name])
            self.policy_action_spaces[name] = spaces.Box(
                low=self.action_space.low[ids].copy(), high=self.action_space.high[ids].copy(), dtype=np.float32)
            history_dim = (ACTION_HISTORY_LENGTH * len(ids)
                           if self.policy_uses_action_history[name] else 0)
            self.policy_observation_spaces[name] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(len(self.policy_observation_indices[name]) + history_dim,),
                dtype=np.float32)
        self.policy_action_history = {
            name: torch.zeros((self.num_envs, ACTION_HISTORY_LENGTH,
                              len(self.policy_action_indices[name])),
                              dtype=torch.float32, device=self.torch_device)
            for name in POLICY_NAMES
        }

    def _build_action_groups(self):
        names = tuple(self.action_names)
        return {
            "waist": tuple(i for i, n in enumerate(names) if n.startswith("waist_")),
            "right_arm": tuple(i for i, n in enumerate(names) if n.startswith("right_") and "_hand_" not in n),
            "left_arm": tuple(i for i, n in enumerate(names) if n.startswith("left_") and "_hand_" not in n),
            "right_fingers": tuple(i for i, n in enumerate(names) if n.startswith("right_hand_")),
            "left_fingers": tuple(i for i, n in enumerate(names) if n.startswith("left_hand_")),
            "direction": tuple(range(len(names) - 3, len(names))),
        }

    def _build_observation_groups(self):
        groups, cursor = OrderedDict(), 0
        def add(name, size):
            nonlocal cursor
            groups[name] = tuple(range(cursor, cursor + size)); cursor += size
        add("joint_positions", len(self.robot_joint_names))
        add("robot_body_states", len(self.main_robot_link_names) * 7)
        for name, size in (
            ("pelvis_velocity", 6), ("chair_position", 3), ("chair_velocity", 3),
            ("chair_vector_world", 3), ("chair_vector_body", 2), ("chair_distance", 1),
            ("final_distance", 1), ("staging_vector_body", 2), ("final_vector_body", 2),
            ("chair_back_body", 2), ("hand_target_body", 6),
            ("hand_orientation_error", 6), ("hand_velocity_body", 6),
            ("fingertip_forces", 6), ("previous_walk_command", 3),
            ("stage", self.num_stages), ("arm_errors", 2)):
            add(name, size)
        if cursor != self.observation_space.shape[0]:
            raise RuntimeError(f"Observation layout covers {cursor}, expected {self.observation_space.shape[0]}")
        groups["chair_geometry"] = sum((groups[k] for k in (
            "chair_position", "chair_velocity", "chair_vector_world", "chair_vector_body",
            "chair_distance", "final_distance", "staging_vector_body", "final_vector_body",
            "chair_back_body")), ())
        groups["left_hand_task"] = (groups["hand_target_body"][:3]
            + groups["hand_orientation_error"][:3] + groups["hand_velocity_body"][:3]
            + groups["arm_errors"][:1])
        groups["right_hand_task"] = (groups["hand_target_body"][3:]
            + groups["hand_orientation_error"][3:] + groups["hand_velocity_body"][3:]
            + groups["arm_errors"][1:])
        groups["all"] = tuple(range(cursor))
        return groups

    def _resolve_action_indices(self):
        result, owner = {}, {}
        for policy in POLICY_NAMES:
            spec, explicit = self.policy_specs[policy], self.policy_specs[policy].get("action_names")
            if explicit is not None:
                missing = [n for n in explicit if n not in self.action_names]
                if missing: raise ValueError(f"{policy}.action_names has unknown values: {missing}")
                indices = tuple(self.action_names.index(n) for n in explicit)
            else:
                requested = spec.get("action_groups", (policy,))
                unknown = [g for g in requested if g not in self.action_groups]
                if unknown: raise ValueError(f"{policy}.action_groups has unknown groups: {unknown}")
                indices = tuple(i for group in requested for i in self.action_groups[group])
            if not indices: raise ValueError(f"Policy {policy} has an empty action space")
            for index in indices:
                if index in owner:
                    raise ValueError(f"Action {self.action_names[index]} belongs to {owner[index]} and {policy}")
                owner[index] = policy
            result[policy] = indices
        missing = [self.action_names[i] for i in range(len(self.action_names)) if i not in owner]
        if missing: raise ValueError(f"No separate policy controls actions: {missing}")
        return result

    def _resolve_observation_indices(self):
        result = {}
        self.policy_uses_action_history = {}
        for policy in POLICY_NAMES:
            requested = self.policy_specs[policy].get("observation_groups", DEFAULT_OBSERVATIONS[policy])
            self.policy_uses_action_history[policy] = "previous_policy_actions" in requested
            physical_groups = [g for g in requested if g != "previous_policy_actions"]
            unknown = [g for g in physical_groups if g not in self.observation_groups]
            if unknown: raise ValueError(f"{policy}.observation_groups has unknown groups: {unknown}")
            indices = tuple(dict.fromkeys(i for group in physical_groups for i in self.observation_groups[group]))
            if not indices and not self.policy_uses_action_history[policy]:
                raise ValueError(f"Policy {policy} has an empty observation space")
            result[policy] = indices
        return result

    def policy_env(self, name):
        return PolicySpaceEnv(self, name)

    def policy_observations_torch(self, full_observation):
        observations = {}
        for name, ids in self.policy_observation_indices.items():
            selected = full_observation.index_select(
                1, torch.as_tensor(ids, dtype=torch.long,
                                   device=full_observation.device))
            if self.policy_uses_action_history[name]:
                history = self.policy_action_history[name].reshape(self.num_envs, -1)
                selected = torch.cat((selected, history), dim=1)
            observations[name] = selected
        return observations

    def _update_action_history(self, policy_actions, done=None):
        for name in POLICY_NAMES:
            action = torch.as_tensor(
                policy_actions[name], dtype=torch.float32, device=self.torch_device)
            ids = torch.as_tensor(self.policy_action_indices[name], dtype=torch.long,
                                  device=self.torch_device)
            controlled = torch.maximum(torch.minimum(
                action, self._action_high_torch[ids]), self._action_low_torch[ids])
            history = self.policy_action_history[name]
            history[:, 1].copy_(history[:, 0])
            history[:, 0].copy_(controlled.detach())
            if done is not None and done.any():
                history[done] = 0.0

    def compose_actions_torch(self, actions):
        missing = set(POLICY_NAMES) - set(actions)
        if missing: raise ValueError(f"Missing actions for policies: {sorted(missing)}")
        full = torch.zeros((self.num_envs, self.action_space.shape[0]),
                           dtype=torch.float32, device=self.torch_device)
        for name in POLICY_NAMES:
            local = torch.as_tensor(actions[name], dtype=torch.float32, device=self.torch_device)
            expected = (self.num_envs, len(self.policy_action_indices[name]))
            if tuple(local.shape) != expected:
                raise ValueError(f"{name} action shape must be {expected}, got {tuple(local.shape)}")
            ids = torch.as_tensor(self.policy_action_indices[name], dtype=torch.long, device=self.torch_device)
            low, high = self._action_low_torch[ids], self._action_high_torch[ids]
            full.index_copy_(1, ids, torch.maximum(torch.minimum(local, high), low))
        return full

    def separate_rewards_torch(self, metadata, policy_actions):
        raw, done, rewards = metadata.get("raw_reward_terms", {}), metadata.get("physical_done"), {}
        policy_reward_terms = {}
        for name in POLICY_NAMES:
            action = policy_actions[name]
            ids = torch.as_tensor(
                self.policy_action_indices[name], dtype=torch.long,
                device=self.torch_device)
            controlled_action = torch.maximum(torch.minimum(
                action, self._action_high_torch[ids]), self._action_low_torch[ids])
            reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.torch_device)
            policy_reward_terms[name] = {}
            for term, weight in self.policy_specs[name].get("reward_terms", {}).items():
                if term == "PolicyDeltaAction":
                    scale = float(self.policy_specs[name].get("delta_action_scale", 0.35))
                    value = torch.clamp((controlled_action - self.policy_action_history[name][:, 0]).abs().mean(1) / scale, 0, 1)
                elif term == "TaskReward":
                    value = metadata["task_reward"]
                else:
                    if term not in raw:
                        raise KeyError(f"Reward {term!r} for {name} unavailable; choose from {sorted(raw)}")
                    value = raw[term].to(self.torch_device).float()
                contribution = float(weight) * value
                reward += contribution
                policy_reward_terms[name][term] = contribution.detach()
            rewards[name] = reward
        metadata["policy_reward_terms"] = policy_reward_terms
        self._update_action_history(policy_actions, done)
        return rewards

    def torch_reset(self):
        observation = super().torch_reset()
        for history in self.policy_action_history.values(): history.zero_()
        return observation

    def reset(self):
        observation = super().reset()
        for history in self.policy_action_history.values():
            history.zero_()
        return observation

    def step_async(self, actions):
        actions_t = torch.as_tensor(
            actions, dtype=torch.float32, device=self.torch_device)
        local = {}
        for name, indices in self.policy_action_indices.items():
            ids = torch.as_tensor(
                indices, dtype=torch.long, device=self.torch_device)
            local[name] = actions_t.index_select(1, ids)
        self._update_action_history(local)
        super().step_async(actions)

    def step_wait(self):
        observation, rewards, dones, infos = super().step_wait()
        done_t = torch.as_tensor(
            dones, dtype=torch.bool, device=self.torch_device)
        if done_t.any():
            for history in self.policy_action_history.values():
                history[done_t] = 0.0
        return observation, rewards, dones, infos

    def torch_step(self, actions):
        observation, reward, done, metadata = super().torch_step(actions)
        metadata["task_reward"] = reward
        metadata["raw_reward_terms"] = getattr(self.env.env.handler.task, "last_raw_reward_terms", {})
        return observation, reward, done, metadata

ChairmanSeparateVecEnv = StableBaseline3VecEnv

"""Train finger actors against the known stage-dependent joint pose.

PPO still trains the actor and critic on physical rollouts. This auxiliary
objective provides an unclipped actor gradient, including at action limits,
and teaches open-to-grasp transitions independently of snapshot finger poses.
"""
from __future__ import annotations

import math
import torch

from metasim.utils.chairman_grasp import STAGE2_FINGER_JOINT_TARGETS_BY_SIDE


def reset_finger_actor(model):
    """Discard the failed actor only; preserve the critic and other policies."""
    actor = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net
    actor.apply(lambda layer: model.policy.init_weights(layer, gain=math.sqrt(2)))
    head.apply(lambda layer: model.policy.init_weights(layer, gain=0.01))
    for parameter in list(actor.parameters()) + list(head.parameters()):
        model.policy.optimizer.state.pop(parameter, None)


def constrain_action_std(model, spec, *, reset=False):
    """Project Gaussian std into configured bounds; clear stale Adam momentum."""
    low = float(spec.get("action_std_min", 0.05))
    high = spec.get("action_std_max")
    if high is None:
        return
    high = float(high)
    if not math.isfinite(low) or not math.isfinite(high) or not 0 < low <= high:
        raise ValueError("action_std bounds must satisfy 0 < min <= max")
    parameter = model.policy.log_std
    with torch.no_grad():
        old = parameter.clone()
        if reset:
            initial = float(spec.get("log_std_init", -1.6))
            if not math.isfinite(initial):
                raise ValueError("log_std_init must be finite")
            parameter.fill_(initial)
        parameter.clamp_(math.log(low), math.log(high))
    if reset or not torch.equal(old, parameter):
        model.policy.optimizer.state.pop(parameter, None)


class FingerPoseSupervision:
    def __init__(self, env, name, *, stage_only=False):
        self.stage_only = bool(stage_only)
        side = name.removesuffix("_fingers")
        if side not in STAGE2_FINGER_JOINT_TARGETS_BY_SIDE:
            raise ValueError("Finger pose supervision only supports left/right_fingers")
        device = env.torch_device
        action_ids = env.policy_action_indices[name]
        names = [env.action_names[i] for i in action_ids]
        targets = STAGE2_FINGER_JOINT_TARGETS_BY_SIDE[side]
        if set(names) != set(targets):
            raise ValueError(f"{name}: pose supervision requires all seven finger joints")
        local = {physical: i for i, physical in enumerate(env.policy_observation_indices[name])}
        try:
            stage_indices = [local[i] for i in env.observation_groups["stage"]]
            joint_indices = [local[env.observation_groups["joint_positions"][
                env.sim_joint_names.index(joint)]] for joint in names]
        except KeyError as exc:
            raise ValueError(f"{name}: pose supervision requires stage and joint_positions observations") from exc
        self.stage_indices = torch.tensor(stage_indices, device=device)
        self.joint_indices = torch.tensor(joint_indices, device=device)
        self.num_stages = len(stage_indices)
        # The observation also includes the terminal stage 6; train labels
        # only for actionable stages 0..5 while preserving the saved layout.
        if self.num_stages < 6:
            raise ValueError("Finger pose supervision needs stages 0 through 5")
        self.closed = torch.tensor([targets[n] for n in names], device=device)
        self.low = torch.tensor(env.action_space.low[list(action_ids)], device=device)
        self.high = torch.tensor(env.action_space.high[list(action_ids)], device=device)
        self.scale = (self.high - self.low).clamp_min(0.1)
        self.history_dim = 2 * len(names) if env.policy_uses_action_history[name] else 0

    def constrain_actor(self, policy):
        """For fixed poses, make the actor independent of nuisance observations.

        The critic retains the complete observation. Zero weights are stored
        in the standard SB3 checkpoint, so evaluation needs no special routing.
        Disable stage_only if fingers later need contact-dependent target poses.
        """
        if not self.stage_only:
            return
        layer = next(module for module in policy.mlp_extractor.policy_net.modules()
                     if isinstance(module, torch.nn.Linear))
        mask = torch.ones(layer.in_features, dtype=torch.bool, device=layer.weight.device)
        mask[self.stage_indices] = False
        with torch.no_grad():
            layer.weight[:, mask] = 0.0

    def targets(self, observations):
        stage = observations.index_select(1, self.stage_indices).argmax(1)
        close = (stage == 2) | (stage == 3)
        return close[:, None].float() * self.closed[None, :]

    def augment(self, observations):
        """Balance stages and vary current joint positions / both previous actions.

        The desired finger pose only depends on stage. Counterfactual finger
        states are valid labels for this objective even before stage 2 is reached.
        They are never used as physical PPO transitions or critic targets.
        """
        result = observations.detach().clone()
        count = len(result)
        # Remove dependence on the stage-0 hand/leg/contact context during
        # warmup. These synthetic rows are actor labels only, never PPO data.
        diverse = torch.rand(count, device=result.device) < 0.5
        result[diverse] = 1.5 * torch.randn_like(result[diverse])
        stages = torch.randint(6, (count,), device=result.device)
        result[:, self.stage_indices] = torch.nn.functional.one_hot(
            stages, self.num_stages).float()
        positions = self.low + torch.rand(
            (count, len(self.low)), device=result.device) * (self.high - self.low)
        # Half of examples explicitly start with fully open fingers.
        positions[torch.rand(count, device=result.device) < 0.5] = 0.0
        result[:, self.joint_indices] = positions
        if self.history_dim:
            history = self.low.repeat(2) + torch.rand(
                (count, self.history_dim), device=result.device
            ) * (self.high - self.low).repeat(2)
            history[torch.rand(count, device=result.device) < 0.5] = 0.0
            result[:, -self.history_dim:] = history
        return result

    def loss(self, policy, observations, *, augmented=True):
        inputs = self.augment(observations) if augmented else observations
        # Supervise the raw mean, not its clipped value: the latter has zero
        # gradient when an old actor outputs e.g. 35 rad outside joint limits.
        mean = policy.get_distribution(inputs).distribution.mean
        target = self.targets(inputs)
        return ((mean - target) / self.scale).square().mean()

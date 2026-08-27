"""Inference wrapper for Unitree's pretrained G1 walking policy.

The bundled ``motion.pt`` is the 12-DoF recurrent policy from
``unitreerobotics/unitree_rl_gym``.  The policy produces normalized actions,
not joint positions.  :class:`G1MotionPolicy` builds the exact 47-element
observation used during training and converts the network output to leg joint
position targets in radians.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]
JointState = Union[ArrayLike, Mapping[str, float]]


class G1MotionPolicy:
    """Stateful controller around Unitree's recurrent 12-DoF G1 policy.

    One instance can control either one robot or a fixed-size batch of robots.
    The batch size is inferred from ``joint_positions`` on the first call.  If
    it changes later, all recurrent state is reset.

    Inputs and outputs use the joint order in :attr:`JOINT_NAMES`.  Angular
    velocity must be in the pelvis/body frame unless
    ``angular_velocity_frame="world"`` is selected.  Quaternions use WXYZ
    order, which is also the order exposed by MetaSim robot body states.
    """

    JOINT_NAMES = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )

    NUM_ACTIONS = 12
    NUM_OBSERVATIONS = 47
    CONTROL_DT = 0.02
    GAIT_PERIOD = 0.8

    DEFAULT_ANGLES = np.array(
        [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
        dtype=np.float32,
    )
    JOINT_LOWER_LIMITS = np.array(
        [
            -2.5307,
            -0.5236,
            -2.7576,
            -0.087267,
            -0.87267,
            -0.2618,
            -2.5307,
            -2.9671,
            -2.7576,
            -0.087267,
            -0.87267,
            -0.2618,
        ],
        dtype=np.float32,
    )
    JOINT_UPPER_LIMITS = np.array(
        [
            2.8798,
            2.9671,
            2.7576,
            2.8798,
            0.5236,
            0.2618,
            2.8798,
            0.5236,
            2.7576,
            2.8798,
            0.5236,
            0.2618,
        ],
        dtype=np.float32,
    )

    ANGULAR_VELOCITY_SCALE = 0.25
    JOINT_POSITION_SCALE = 1.0
    JOINT_VELOCITY_SCALE = 0.05
    COMMAND_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
    MAX_COMMAND = np.array([0.8, 0.5, 1.57], dtype=np.float32)
    ACTION_SCALE = 0.25

    def __init__(
        self,
        policy_path: Optional[Union[str, Path]] = None,
        *,
        device: Union[str, torch.device] = "cpu",
        control_dt: float = CONTROL_DT,
        clip_commands: bool = True,
        clip_joint_targets: bool = True,
        validate_tensors: bool = False,
    ) -> None:
        """Load the policy and initialize its recurrent controller state."""
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")

        self.policy_path = (
            Path(policy_path).expanduser()
            if policy_path is not None
            else Path(__file__).with_name("motion.pt")
        )
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"G1 motion policy was not found: {self.policy_path}")

        self.device = torch.device(device)
        self.control_dt = float(control_dt)
        self.clip_commands = clip_commands
        self.clip_joint_targets = clip_joint_targets
        # Full finite-value reductions synchronize CUDA. Keep them available
        # for controller debugging, but disabled on the training hot path.
        self.validate_tensors = validate_tensors

        self._policy = torch.jit.load(str(self.policy_path), map_location=self.device)
        self._policy.eval()
        self._batch_size = 1
        self._previous_action = np.zeros((1, self.NUM_ACTIONS), dtype=np.float32)
        self._elapsed_time = np.zeros(1, dtype=np.float32)
        self.last_observation = np.zeros((1, self.NUM_OBSERVATIONS), dtype=np.float32)
        self.last_action = np.zeros((1, self.NUM_ACTIONS), dtype=np.float32)
        self._previous_action_torch = torch.zeros(
            (1, self.NUM_ACTIONS), dtype=torch.float32, device=self.device
        )
        self._elapsed_time_torch = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.last_observation_torch = torch.zeros(
            (1, self.NUM_OBSERVATIONS), dtype=torch.float32, device=self.device
        )
        self.last_action_torch = torch.zeros(
            (1, self.NUM_ACTIONS), dtype=torch.float32, device=self.device
        )
        self._resize_policy_memory(1)
        self._validate_loaded_policy()

    @property
    def batch_size(self) -> int:
        """Current number of independently tracked robot environments."""
        return self._batch_size

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Reset LSTM memory, previous action, and gait phase.

        Args:
            env_ids: Indices to reset.  ``None`` resets the complete batch.
        """
        if env_ids is None:
            self._policy.reset_memory()
            self._previous_action.fill(0.0)
            self._elapsed_time.fill(0.0)
            self.last_observation.fill(0.0)
            self.last_action.fill(0.0)
            self._previous_action_torch.zero_()
            self._elapsed_time_torch.zero_()
            self.last_observation_torch.zero_()
            self.last_action_torch.zero_()
            return

        indices = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return
        if np.any(indices < 0) or np.any(indices >= self._batch_size):
            raise IndexError(f"env_ids must be in [0, {self._batch_size - 1}]")

        torch_indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self._policy.hidden_state.index_fill_(1, torch_indices, 0.0)
            self._policy.cell_state.index_fill_(1, torch_indices, 0.0)
        self._previous_action[indices] = 0.0
        self._elapsed_time[indices] = 0.0
        self.last_observation[indices] = 0.0
        self.last_action[indices] = 0.0
        self._previous_action_torch.index_fill_(0, torch_indices, 0.0)
        self._elapsed_time_torch.index_fill_(0, torch_indices, 0.0)
        self.last_observation_torch.index_fill_(0, torch_indices, 0.0)
        self.last_action_torch.index_fill_(0, torch_indices, 0.0)

    def predict_joint_positions(
        self,
        *,
        joint_positions: JointState,
        joint_velocities: JointState,
        angular_velocity: ArrayLike,
        command: ArrayLike,
        base_quaternion_wxyz: Optional[ArrayLike] = None,
        projected_gravity: Optional[ArrayLike] = None,
        angular_velocity_frame: str = "body",
        command_is_normalized: bool = False,
        time_seconds: Optional[ArrayLike] = None,
        previous_action: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Run one policy step and return target leg angles in radians.

        ``command`` normally contains physical ``[vx, vy, yaw_rate]`` values in
        m/s, m/s, and rad/s.  With ``command_is_normalized=True`` it instead
        contains joystick values in ``[-1, 1]`` and is multiplied by the
        original policy command limits.

        Supply exactly one of ``base_quaternion_wxyz`` and
        ``projected_gravity``.  The latter is useful if the simulator already
        computes gravity expressed in the pelvis frame.
        """
        q, squeeze_output = self._joint_batch(joint_positions, "joint_positions")
        self._ensure_batch_size(q.shape[0])
        dq, _ = self._joint_batch(joint_velocities, "joint_velocities")
        if dq.shape[0] != self._batch_size:
            raise ValueError("joint_velocities batch size does not match joint_positions")

        phase_time = self._prepare_phase_time(time_seconds)
        observation = self.build_observation(
            joint_positions=q,
            joint_velocities=dq,
            angular_velocity=angular_velocity,
            command=command,
            base_quaternion_wxyz=base_quaternion_wxyz,
            projected_gravity=projected_gravity,
            angular_velocity_frame=angular_velocity_frame,
            command_is_normalized=command_is_normalized,
            time_seconds=phase_time,
            previous_action=previous_action,
        )

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            action_tensor = self._policy(observation_tensor)
        action = action_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if action.shape != (self._batch_size, self.NUM_ACTIONS):
            raise RuntimeError(
                "Unexpected motion policy output shape: "
                f"expected {(self._batch_size, self.NUM_ACTIONS)}, got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("Motion policy produced NaN or infinite actions")

        targets = self.DEFAULT_ANGLES[None, :] + self.ACTION_SCALE * action
        if self.clip_joint_targets:
            targets = np.clip(targets, self.JOINT_LOWER_LIMITS, self.JOINT_UPPER_LIMITS)

        self._previous_action = action.copy()
        self.last_action = action.copy()
        self.last_observation = observation.copy()
        targets = targets.astype(np.float32, copy=False)
        return targets[0] if squeeze_output else targets

    __call__ = predict_joint_positions

    def predict_joint_positions_torch(
        self,
        *,
        joint_positions: ArrayLike,
        joint_velocities: ArrayLike,
        angular_velocity: ArrayLike,
        command: ArrayLike,
        base_quaternion_wxyz: Optional[ArrayLike] = None,
        projected_gravity: Optional[ArrayLike] = None,
        angular_velocity_frame: str = "body",
        command_is_normalized: bool = False,
        time_seconds: Optional[ArrayLike] = None,
        previous_action: Optional[ArrayLike] = None,
    ) -> torch.Tensor:
        """Torch-native batched inference without a CPU/NumPy round trip."""
        q, squeeze_output = self._torch_matrix(
            joint_positions, self.NUM_ACTIONS, "joint_positions"
        )
        self._ensure_batch_size(q.shape[0])
        dq, _ = self._torch_matrix(
            joint_velocities, self.NUM_ACTIONS, "joint_velocities"
        )
        if dq.shape[0] != self._batch_size:
            raise ValueError("joint_velocities batch size does not match joint_positions")

        observation = self.build_observation_torch(
            joint_positions=q,
            joint_velocities=dq,
            angular_velocity=angular_velocity,
            command=command,
            base_quaternion_wxyz=base_quaternion_wxyz,
            projected_gravity=projected_gravity,
            angular_velocity_frame=angular_velocity_frame,
            command_is_normalized=command_is_normalized,
            time_seconds=time_seconds,
            previous_action=previous_action,
        )
        with torch.inference_mode():
            action = self._policy(observation)
        if tuple(action.shape) != (self._batch_size, self.NUM_ACTIONS):
            raise RuntimeError(
                "Unexpected motion policy output shape: "
                f"expected {(self._batch_size, self.NUM_ACTIONS)}, got {tuple(action.shape)}"
            )
        if self.validate_tensors and not bool(torch.isfinite(action).all()):
            raise RuntimeError("Motion policy produced NaN or infinite actions")

        defaults = torch.as_tensor(
            self.DEFAULT_ANGLES, dtype=torch.float32, device=self.device
        )
        targets = defaults.unsqueeze(0) + self.ACTION_SCALE * action
        if self.clip_joint_targets:
            lower = torch.as_tensor(
                self.JOINT_LOWER_LIMITS, dtype=torch.float32, device=self.device
            )
            upper = torch.as_tensor(
                self.JOINT_UPPER_LIMITS, dtype=torch.float32, device=self.device
            )
            targets = torch.maximum(torch.minimum(targets, upper), lower)

        self._previous_action_torch.copy_(action)
        self.last_action_torch.copy_(action)
        self.last_observation_torch.copy_(observation)
        return targets[0] if squeeze_output else targets

    def build_observation_torch(
        self,
        *,
        joint_positions: ArrayLike,
        joint_velocities: ArrayLike,
        angular_velocity: ArrayLike,
        command: ArrayLike,
        base_quaternion_wxyz: Optional[ArrayLike] = None,
        projected_gravity: Optional[ArrayLike] = None,
        angular_velocity_frame: str = "body",
        command_is_normalized: bool = False,
        time_seconds: Optional[ArrayLike] = None,
        previous_action: Optional[ArrayLike] = None,
    ) -> torch.Tensor:
        """Build the 47-element walking observation directly on ``device``."""
        q, _ = self._torch_matrix(
            joint_positions, self.NUM_ACTIONS, "joint_positions"
        )
        self._ensure_batch_size(q.shape[0])
        dq, _ = self._torch_matrix(
            joint_velocities, self.NUM_ACTIONS, "joint_velocities"
        )
        omega = self._torch_vector(angular_velocity, 3, "angular_velocity")
        cmd = self._torch_vector(command, 3, "command")
        max_command = torch.as_tensor(
            self.MAX_COMMAND, dtype=torch.float32, device=self.device
        )
        if command_is_normalized:
            if self.clip_commands:
                cmd = cmd.clamp(-1.0, 1.0)
            cmd = cmd * max_command
        elif self.clip_commands:
            cmd = torch.maximum(torch.minimum(cmd, max_command), -max_command)

        if (base_quaternion_wxyz is None) == (projected_gravity is None):
            raise ValueError(
                "Supply exactly one of base_quaternion_wxyz or projected_gravity"
            )
        quaternion = None
        if base_quaternion_wxyz is not None:
            quaternion = self._torch_vector(
                base_quaternion_wxyz, 4, "base_quaternion_wxyz"
            )
            norms = torch.linalg.vector_norm(quaternion, dim=1, keepdim=True)
            if self.validate_tensors and not bool((norms >= 1e-8).all()):
                raise ValueError("base_quaternion_wxyz must not contain a zero quaternion")
            quaternion = quaternion / norms
            w, x, y, z = quaternion.unbind(dim=1)
            gravity = torch.stack(
                (
                    2.0 * (-z * x + w * y),
                    -2.0 * (z * y + w * x),
                    1.0 - 2.0 * (w * w + z * z),
                ),
                dim=1,
            )
        else:
            gravity = self._torch_vector(projected_gravity, 3, "projected_gravity")

        if angular_velocity_frame == "world":
            if quaternion is None:
                raise ValueError(
                    "base_quaternion_wxyz is required when angular_velocity_frame='world'"
                )
            w = quaternion[:, :1]
            inverse_xyz = -quaternion[:, 1:]
            cross_1 = 2.0 * torch.linalg.cross(inverse_xyz, omega, dim=1)
            omega = omega + w * cross_1 + torch.linalg.cross(
                inverse_xyz, cross_1, dim=1
            )
        elif angular_velocity_frame != "body":
            raise ValueError("angular_velocity_frame must be 'body' or 'world'")

        previous = (
            self._previous_action_torch
            if previous_action is None
            else self._torch_vector(previous_action, self.NUM_ACTIONS, "previous_action")
        )
        if time_seconds is None:
            self._elapsed_time_torch.add_(self.control_dt)
            phase_time = self._elapsed_time_torch
        else:
            phase_time = self._torch_scalar(time_seconds, "time_seconds")
            if self.validate_tensors and not bool((phase_time >= 0.0).all()):
                raise ValueError("time_seconds must be non-negative")
            self._elapsed_time_torch.copy_(phase_time)
        phase = torch.remainder(phase_time, self.GAIT_PERIOD) / self.GAIT_PERIOD
        defaults = torch.as_tensor(
            self.DEFAULT_ANGLES, dtype=torch.float32, device=self.device
        )
        command_scale = torch.as_tensor(
            self.COMMAND_SCALE, dtype=torch.float32, device=self.device
        )
        observation = torch.cat(
            (
                omega * self.ANGULAR_VELOCITY_SCALE,
                gravity,
                cmd * command_scale,
                (q - defaults) * self.JOINT_POSITION_SCALE,
                dq * self.JOINT_VELOCITY_SCALE,
                previous,
                torch.sin(2.0 * torch.pi * phase).unsqueeze(1),
                torch.cos(2.0 * torch.pi * phase).unsqueeze(1),
            ),
            dim=1,
        )
        if tuple(observation.shape) != (self._batch_size, self.NUM_OBSERVATIONS):
            raise RuntimeError(
                f"Internal error: built observation has shape {tuple(observation.shape)}"
            )
        if self.validate_tensors and not bool(torch.isfinite(observation).all()):
            raise ValueError("Policy inputs contain NaN or infinite values")
        return observation

    def predict_joint_dict(self, **kwargs: object) -> dict[str, float]:
        """Run one single-robot step and return ``joint_name: target_rad``."""
        targets = self.predict_joint_positions(**kwargs)
        if targets.ndim != 1:
            raise ValueError("predict_joint_dict only supports a single robot; use the array API for batches")
        return dict(zip(self.JOINT_NAMES, targets.tolist()))

    def build_observation(
        self,
        *,
        joint_positions: JointState,
        joint_velocities: JointState,
        angular_velocity: ArrayLike,
        command: ArrayLike,
        base_quaternion_wxyz: Optional[ArrayLike] = None,
        projected_gravity: Optional[ArrayLike] = None,
        angular_velocity_frame: str = "body",
        command_is_normalized: bool = False,
        time_seconds: Optional[ArrayLike] = None,
        previous_action: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Build the exact 47-element observation without running the network."""
        q, _ = self._joint_batch(joint_positions, "joint_positions")
        self._ensure_batch_size(q.shape[0])
        dq, _ = self._joint_batch(joint_velocities, "joint_velocities")
        if dq.shape[0] != self._batch_size:
            raise ValueError("joint_velocities batch size does not match joint_positions")

        omega = self._vector_batch(angular_velocity, 3, "angular_velocity")
        cmd = self._vector_batch(command, 3, "command")
        if command_is_normalized:
            if self.clip_commands:
                cmd = np.clip(cmd, -1.0, 1.0)
            cmd = cmd * self.MAX_COMMAND[None, :]
        elif self.clip_commands:
            cmd = np.clip(cmd, -self.MAX_COMMAND, self.MAX_COMMAND)

        if (base_quaternion_wxyz is None) == (projected_gravity is None):
            raise ValueError(
                "Supply exactly one of base_quaternion_wxyz or projected_gravity"
            )
        quaternion = None
        if base_quaternion_wxyz is not None:
            quaternion = self._normalized_quaternion(base_quaternion_wxyz)
            gravity = self.projected_gravity_from_quaternion(quaternion)
        else:
            gravity = self._vector_batch(projected_gravity, 3, "projected_gravity")

        if angular_velocity_frame == "world":
            if quaternion is None:
                raise ValueError(
                    "base_quaternion_wxyz is required when angular_velocity_frame='world'"
                )
            omega = self.world_to_body_vector(omega, quaternion)
        elif angular_velocity_frame != "body":
            raise ValueError("angular_velocity_frame must be 'body' or 'world'")

        if previous_action is None:
            previous = self._previous_action
        else:
            previous = self._vector_batch(previous_action, self.NUM_ACTIONS, "previous_action")

        if time_seconds is None:
            phase_time = self._elapsed_time
        else:
            phase_time = self._scalar_batch(time_seconds, "time_seconds")
        phase = np.remainder(phase_time, self.GAIT_PERIOD) / self.GAIT_PERIOD

        observation = np.concatenate(
            [
                omega * self.ANGULAR_VELOCITY_SCALE,
                gravity,
                cmd * self.COMMAND_SCALE[None, :],
                (q - self.DEFAULT_ANGLES[None, :]) * self.JOINT_POSITION_SCALE,
                dq * self.JOINT_VELOCITY_SCALE,
                previous,
                np.sin(2.0 * np.pi * phase)[:, None],
                np.cos(2.0 * np.pi * phase)[:, None],
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        if observation.shape != (self._batch_size, self.NUM_OBSERVATIONS):
            raise RuntimeError(f"Internal error: built observation has shape {observation.shape}")
        if not np.all(np.isfinite(observation)):
            raise ValueError("Policy inputs contain NaN or infinite values")
        return observation

    @classmethod
    def ordered_joint_values(cls, values: Mapping[str, float]) -> np.ndarray:
        """Convert a joint mapping to the policy's fixed 12-joint order."""
        missing = [name for name in cls.JOINT_NAMES if name not in values]
        if missing:
            raise KeyError(f"Missing G1 leg joints: {', '.join(missing)}")
        return np.asarray([values[name] for name in cls.JOINT_NAMES], dtype=np.float32)

    @staticmethod
    def projected_gravity_from_quaternion(quaternion_wxyz: ArrayLike) -> np.ndarray:
        """Return world gravity expressed in the body frame for WXYZ quaternions."""
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float32)
        squeeze = quaternion.ndim == 1
        quaternion = np.atleast_2d(quaternion)
        if quaternion.shape[1] != 4:
            raise ValueError("quaternion_wxyz must have shape (4,) or (N, 4)")
        norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("quaternion_wxyz must not contain a zero quaternion")
        w, x, y, z = (quaternion / norms).T
        gravity = np.stack(
            [
                2.0 * (-z * x + w * y),
                -2.0 * (z * y + w * x),
                1.0 - 2.0 * (w * w + z * z),
            ],
            axis=1,
        ).astype(np.float32)
        return gravity[0] if squeeze else gravity

    @staticmethod
    def world_to_body_vector(vector: ArrayLike, quaternion_wxyz: ArrayLike) -> np.ndarray:
        """Rotate world-frame vectors into the body frame using WXYZ quaternions."""
        vectors = np.atleast_2d(np.asarray(vector, dtype=np.float32))
        quaternions = np.atleast_2d(np.asarray(quaternion_wxyz, dtype=np.float32))
        if vectors.shape[1] != 3 or quaternions.shape[1] != 4:
            raise ValueError("Expected vector shape (N, 3) and quaternion shape (N, 4)")
        if vectors.shape[0] != quaternions.shape[0]:
            raise ValueError("Vector and quaternion batch sizes do not match")

        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("quaternion_wxyz must not contain a zero quaternion")
        w = quaternions[:, 0:1] / norms
        xyz = quaternions[:, 1:4] / norms
        # Applying the inverse unit quaternion: q* v q.
        inverse_xyz = -xyz
        cross_1 = 2.0 * np.cross(inverse_xyz, vectors)
        return (vectors + w * cross_1 + np.cross(inverse_xyz, cross_1)).astype(np.float32)

    def _prepare_phase_time(self, time_seconds: Optional[ArrayLike]) -> np.ndarray:
        if time_seconds is None:
            self._elapsed_time += self.control_dt
        else:
            supplied_time = self._scalar_batch(time_seconds, "time_seconds")
            if np.any(supplied_time < 0.0):
                raise ValueError("time_seconds must be non-negative")
            self._elapsed_time = supplied_time.copy()
        return self._elapsed_time.copy()

    def _joint_batch(self, value: JointState, name: str) -> tuple[np.ndarray, bool]:
        if isinstance(value, Mapping):
            array = self.ordered_joint_values(value)
        else:
            array = self._to_numpy(value)
        squeeze = array.ndim == 1
        array = np.atleast_2d(array).astype(np.float32, copy=False)
        if array.ndim != 2 or array.shape[1] != self.NUM_ACTIONS:
            raise ValueError(
                f"{name} must have shape ({self.NUM_ACTIONS},) or (N, {self.NUM_ACTIONS}); "
                f"got {array.shape}"
            )
        return array, squeeze

    def _vector_batch(self, value: ArrayLike, width: int, name: str) -> np.ndarray:
        array = self._to_numpy(value)
        array = np.atleast_2d(array).astype(np.float32, copy=False)
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"{name} must have shape ({width},) or (N, {width}); got {array.shape}")
        if array.shape[0] == 1 and self._batch_size > 1:
            array = np.repeat(array, self._batch_size, axis=0)
        if array.shape[0] != self._batch_size:
            raise ValueError(f"{name} batch size does not match joint_positions")
        return array

    def _scalar_batch(self, value: ArrayLike, name: str) -> np.ndarray:
        array = self._to_numpy(value).reshape(-1).astype(np.float32, copy=False)
        if array.size == 1 and self._batch_size > 1:
            array = np.repeat(array, self._batch_size)
        if array.size != self._batch_size:
            raise ValueError(f"{name} must be a scalar or contain one value per environment")
        return array

    def _normalized_quaternion(self, value: ArrayLike) -> np.ndarray:
        quaternion = self._vector_batch(value, 4, "base_quaternion_wxyz")
        norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("base_quaternion_wxyz must not contain a zero quaternion")
        return (quaternion / norms).astype(np.float32)

    @staticmethod
    def _to_numpy(value: ArrayLike) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _torch_matrix(
        self, value: ArrayLike, width: int, name: str
    ) -> tuple[torch.Tensor, bool]:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        squeeze = tensor.ndim == 1
        if squeeze:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != width:
            raise ValueError(
                f"{name} must have shape ({width},) or (N, {width}); "
                f"got {tuple(tensor.shape)}"
            )
        return tensor, squeeze

    def _torch_vector(self, value: ArrayLike, width: int, name: str) -> torch.Tensor:
        tensor, _ = self._torch_matrix(value, width, name)
        if tensor.shape[0] == 1 and self._batch_size > 1:
            tensor = tensor.expand(self._batch_size, -1)
        if tensor.shape[0] != self._batch_size:
            raise ValueError(f"{name} batch size does not match joint_positions")
        return tensor

    def _torch_scalar(self, value: ArrayLike, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(-1)
        if tensor.numel() == 1 and self._batch_size > 1:
            tensor = tensor.expand(self._batch_size)
        if tensor.numel() != self._batch_size:
            raise ValueError(f"{name} must be a scalar or contain one value per environment")
        return tensor

    def _ensure_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("Policy batch must contain at least one environment")
        if batch_size != self._batch_size:
            self._resize_policy_memory(batch_size)

    def _resize_policy_memory(self, batch_size: int) -> None:
        hidden_size = int(self._policy.hidden_state.shape[-1])
        self._policy.hidden_state = torch.zeros(
            (1, batch_size, hidden_size), dtype=torch.float32, device=self.device
        )
        self._policy.cell_state = torch.zeros(
            (1, batch_size, hidden_size), dtype=torch.float32, device=self.device
        )
        self._batch_size = batch_size
        self._previous_action = np.zeros((batch_size, self.NUM_ACTIONS), dtype=np.float32)
        self._elapsed_time = np.zeros(batch_size, dtype=np.float32)
        self.last_observation = np.zeros((batch_size, self.NUM_OBSERVATIONS), dtype=np.float32)
        self.last_action = np.zeros((batch_size, self.NUM_ACTIONS), dtype=np.float32)
        self._previous_action_torch = torch.zeros(
            (batch_size, self.NUM_ACTIONS), dtype=torch.float32, device=self.device
        )
        self._elapsed_time_torch = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.last_observation_torch = torch.zeros(
            (batch_size, self.NUM_OBSERVATIONS), dtype=torch.float32, device=self.device
        )
        self.last_action_torch = torch.zeros(
            (batch_size, self.NUM_ACTIONS), dtype=torch.float32, device=self.device
        )

    def _validate_loaded_policy(self) -> None:
        with torch.inference_mode():
            output = self._policy(
                torch.zeros((1, self.NUM_OBSERVATIONS), dtype=torch.float32, device=self.device)
            )
        self._policy.reset_memory()
        if tuple(output.shape) != (1, self.NUM_ACTIONS):
            raise ValueError(
                f"{self.policy_path} is not the expected G1 motion policy: "
                f"output shape is {tuple(output.shape)}, expected {(1, self.NUM_ACTIONS)}"
            )

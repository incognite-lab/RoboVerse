from __future__ import annotations

import genesis as gs
import numpy as np
import torch
from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint
from genesis.vis.camera import Camera
from loguru import logger as log

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg, _FileBasedMixin
from metasim.cfg.scenario import ScenarioCfg
from metasim.queries.base import BaseQueryType
from metasim.sim import BaseSimHandler, GymEnvWrapper
from metasim.types import Action, EnvState
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState

# Apply IGL compatibility patch
try:
    import genesis.engine.entities.rigid_entity.rigid_geom as _rigid_geom_module
    import igl as _igl

    _original_compute_sd = _rigid_geom_module.RigidGeom._compute_sd

    def _patched_compute_sd(self, query_points):
        """Patched version that handles different IGL return values"""
        result = _igl.signed_distance(query_points, self._sdf_verts, self._sdf_faces)
        if isinstance(result, tuple):
            return result[0] if len(result) > 0 else None
        return result

    _rigid_geom_module.RigidGeom._compute_sd = _patched_compute_sd
except Exception:
    pass


class GenesisHandler(BaseSimHandler):
    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        super().__init__(scenario, optional_queries)
        self._actions_cache: list[Action] = []
        self.object_inst_dict: dict[str, RigidEntity] = {}
        self.camera_inst_dict: dict[str, Camera] = {}

    def launch(self) -> None:
        gs.init(backend=gs.gpu,logging_level=gs._logging.WARNING)  # TODO: add option for cpu
        # zde změna ve vypisování logů - genesis je moc hlučný (nevypisujíse info logy ani debug)
        self.scene_inst = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 1 / 100,
                substeps=1,
            ),  # TODO: substeps > 1 doesn't work
            vis_options=gs.options.VisOptions(n_rendered_envs=self.scenario.num_envs),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=not self.headless,
        )

        ## Add ground
        try:
            self.scene_inst.add_entity(gs.morphs.Plane())
        except (ValueError, Exception) as e:
            # Fallback if Plane has issues
            log.warning(f"Could not add ground plane: {e}")
            pass
        ## Add robot
        self.robot_inst: RigidEntity = self.scene_inst.add_entity(
            gs.morphs.URDF(
                pos=(0, 0, 2),
                file=self.robot.urdf_path,
                merge_fixed_links=self.robot.collapse_fixed_joints,
            ),
            material=gs.materials.Rigid(gravity_compensation=1 if not self.robot.enabled_gravity else 0),
        )
        self.object_inst_dict[self.robot.name] = self.robot_inst
        log.info(f"robot: {self.robot_inst}")

        ## Add objects
        for obj in self.scenario.objects:
            if isinstance(obj, _FileBasedMixin):
                if isinstance(obj.scale, tuple) or isinstance(obj.scale, list):
                    obj.scale = obj.scale[0]
                    log.warning(
                        f"Genesis does not support different scaling for each axis for {obj.name}, using scale={obj.scale}"
                    )
            if isinstance(obj, PrimitiveCubeCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.Box(size=obj.size), surface=gs.surfaces.Default(color=obj.color)
                )
            elif isinstance(obj, PrimitiveSphereCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.Sphere(radius=obj.radius), surface=gs.surfaces.Default(color=obj.color)
                )
            elif isinstance(obj, RigidObjCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.URDF(file=obj.urdf_path, fixed=obj.fix_base_link, scale=obj.scale),
                )
            elif isinstance(obj, ArticulationObjCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.URDF(file=obj.urdf_path, fixed=obj.fix_base_link, scale=obj.scale),
                )
            else:
                raise NotImplementedError(f"Object type {type(obj)} not supported")
            self.object_inst_dict[obj.name] = obj_inst

        ## Add cameras
        for camera in self.cameras:
            camera_inst = self.scene_inst.add_camera(
                res=(camera.width, camera.height),
                pos=camera.pos,
                lookat=camera.look_at,
                fov=camera.vertical_fov,
            )
            self.camera_inst_dict[camera.name] = camera_inst

        self.scene_inst.build(
            n_envs=self.scenario.num_envs, env_spacing=(self.scenario.env_spacing, self.scenario.env_spacing)
        )

    def _get_states(self, env_ids: list[int] | None = None) -> list[EnvState]:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        object_states = {}
        for obj in self.objects:
            obj_inst = self.object_inst_dict[obj.name]
            if isinstance(obj, ArticulationObjCfg):
                joint_reindex = self.get_joint_reindex(obj.name)
                state = ObjectState(
                    root_state=torch.cat(
                        [
                            obj_inst.get_pos(envs_idx=env_ids),
                            obj_inst.get_quat(envs_idx=env_ids),
                            obj_inst.get_vel(envs_idx=env_ids),
                            obj_inst.get_ang(envs_idx=env_ids),
                        ],
                        dim=-1,
                    ),
                    body_names=self.get_body_names(obj.name),
                    body_state=self.get_body_states(obj.name, envs_idx=env_ids),
                    joint_pos=obj_inst.get_dofs_position(envs_idx=env_ids)[:, joint_reindex],
                    joint_vel=obj_inst.get_dofs_velocity(envs_idx=env_ids)[:, joint_reindex],
                )
            else:
                state = ObjectState(
                    root_state=torch.cat(
                        [
                            obj_inst.get_pos(envs_idx=env_ids),
                            obj_inst.get_quat(envs_idx=env_ids),
                            obj_inst.get_vel(envs_idx=env_ids),
                            obj_inst.get_ang(envs_idx=env_ids),
                        ],
                        dim=-1,
                    ),
                )
            object_states[obj.name] = state

        robot_states = {}
        for obj in [self.robot]:
            obj_inst = self.object_inst_dict[obj.name]
            joint_reindex = self.get_joint_reindex(obj.name)
            state = RobotState(
                root_state=torch.cat(
                    [
                        obj_inst.get_pos(envs_idx=env_ids),
                        obj_inst.get_quat(envs_idx=env_ids),
                        obj_inst.get_vel(envs_idx=env_ids),
                        obj_inst.get_ang(envs_idx=env_ids),
                    ],
                    dim=-1,
                ),
                body_names=self.get_body_names(obj.name),
                body_state=self.get_body_states(obj.name, envs_idx=env_ids),
                joint_pos=obj_inst.get_dofs_position(envs_idx=env_ids)[:, joint_reindex],
                joint_vel=obj_inst.get_dofs_velocity(envs_idx=env_ids)[:, joint_reindex],
                joint_pos_target=None,  # TODO
                joint_vel_target=None,  # TODO
                joint_effort_target=self._get_effort_targets()
                if self._get_control_mode(obj.name) == "effort"
                else None,
            )
            robot_states[obj.name] = state

        camera_states = {}
        for camera in self.cameras:
            camera_inst = self.camera_inst_dict[camera.name]
            rgb, depth, _, _ = camera_inst.render(depth=True)
            state = CameraState(
                rgb=torch.from_numpy(rgb.copy()).unsqueeze(0).repeat_interleave(self.num_envs, dim=0),  # XXX
                depth=torch.from_numpy(depth.copy()).unsqueeze(0).repeat_interleave(self.num_envs, dim=0),  # XXX
            )
            camera_states[camera.name] = state

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, sensors={})


    def get_body_names(self, obj_name: str, sort: bool = False) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            links = self.object_inst_dict[obj_name].links
            body_names = [b.name for b in links]
            if sort:
                body_names.sort()
            return body_names
        else:
            return []

    def get_body_states(self, obj_name: str, envs_idx: list[int] | None = None) -> torch.Tensor | None:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            links = self.object_inst_dict[obj_name].links
            # Collect all states for all links in a batched way
            pos = torch.stack([b.get_pos(envs_idx=envs_idx) for b in links], dim=1)      # [num_envs, n_links, 3]
            quat = torch.stack([b.get_quat(envs_idx=envs_idx) for b in links], dim=1)     # [num_envs, n_links, 4]
            vel = torch.stack([b.get_vel(envs_idx=envs_idx) for b in links], dim=1)       # [num_envs, n_links, 3]
            ang = torch.stack([b.get_ang(envs_idx=envs_idx) for b in links], dim=1)       # [num_envs, n_links, 3]
            # Concatenate along last dimension: [pos, quat, vel, ang] -> [3+4+3+3=13]
            body_states = torch.cat([pos, quat, vel, ang], dim=-1)                        # [num_envs, n_links, 13]
            return body_states
        else:
            return None

    def _set_states(self, states: list[EnvState], env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        states_flat = [state["objects"] | state["robots"] for state in states]
        if len(states) < self.num_envs:
            states = states * self.num_envs
        states_flat = [state["objects"] | state["robots"] for state in states]
        for obj in self.objects + [self.robot]:
            obj_inst = self.object_inst_dict[obj.name]

            # --- Base link position ---
            pos_tensor = torch.stack([states_flat[eid][obj.name]["pos"] for eid in env_ids])  # [N,3]
            pos = pos_tensor.cpu().numpy()
            obj_inst.set_pos(pos, envs_idx=env_ids, relative=False)
            # --- Base link rotation ---
            quat_tensor = torch.stack([states_flat[eid][obj.name]["rot"] for eid in env_ids])  # [N,4]
            quat = quat_tensor.cpu().numpy()
            # Normalize quaternions
            norms = np.linalg.norm(quat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            quat = quat / norms
            obj_inst.set_quat(quat, envs_idx=env_ids, relative=False)

            # --- Joint positions (only if articulation) ---
            if isinstance(obj, ArticulationObjCfg):
                joint_names = self.get_joint_names(obj.name, sort=False)
                if len(joint_names) == 0 or joint_names == ["root_joint"]:
                    print("DEBUG: no joints for", obj.name)
                else:
                    dof_pos = np.array([
                        [states_flat[env_id][obj.name]["dof_pos"][jn] for jn in joint_names]
                        for env_id in env_ids
                    ])
                    base_pos = obj_inst.get_pos(envs_idx=env_ids)   # [N,3]
                    base_quat = obj_inst.get_quat(envs_idx=env_ids) # [N,4]

                    root_state = np.concatenate([base_pos.detach().cpu().numpy(), base_quat.detach().cpu().numpy()], axis=1)  # [N,7]
                    full_qpos = np.concatenate([root_state, dof_pos], axis=1)   # [N, 7 + n_joints]

                    obj_inst.set_qpos(full_qpos, envs_idx=env_ids)

    def set_dof_targets(self, obj_name: str, actions: list[Action]) -> None:
        self._actions_cache = actions

        control_mode = self._get_control_mode(obj_name)
        joint_names = self.get_joint_names(obj_name, sort=False)

        if control_mode == "effort":
            effort = [
                [actions[env_id][self.robot.name]["dof_effort_target"][jn] for jn in joint_names]
                for env_id in range(self.num_envs)
            ]
            if self.object_dict[obj_name].fix_base_link:
                self.robot_inst.control_dofs_force(
                    force=effort,
                    dofs_idx_local=[j.dof_idx_local for j in self.robot_inst.joints if j.dof_idx_local is not None],
                )
            else:
                self.robot_inst.control_dofs_force(
                    force=effort,
                    dofs_idx_local=[
                        j.dof_idx_local
                        for j in self.robot_inst.joints
                        if j.dof_idx_local is not None and j.name != self.robot_inst.base_joint.name
                    ],
                )
        else:
            position = [
                [actions[env_id][self.robot.name]["dof_pos_target"][jn] for jn in joint_names]
                for env_id in range(self.num_envs)
            ]
            dof_idx_local = []
            for j in self.robot_inst.joints:
                try:
                    if j.dofs_idx_local[0] is not None and j.name != self.robot_inst.base_joint.name:
                        dof_idx_local.append(j.dofs_idx_local[0])
                except IndexError:
                    if j.dofs_idx_local[0] is not None and j.name != "root_joint":
                        dof_idx_local.append(j.dofs_idx_local[0])
            self.robot_inst.control_dofs_position(
                    position=position,
                    dofs_idx_local=dof_idx_local,
                )

    def _simulate(self):
        for _ in range(self.scenario.decimation):
            self.scene_inst.step()

    def refresh_render(self):
        """Refresh the render."""
        if not self.headless:
            self.scene_inst.viewer.update()
        self.scene_inst.visualizer.update()

    def close(self):
        pass

    def _get_effort_targets(self) -> torch.Tensor | None:
        """Get the effort targets from cached actions."""
        if not hasattr(self, "_actions_cache") or not self._actions_cache:
            return None

        joint_names = self.get_joint_names(self.robot.name, sort=False)
        effort_targets = []
        for action in self._actions_cache:
            if "dof_effort_target" in action[self.robot.name] and action[self.robot.name]["dof_effort_target"]:
                effort_values = [action[self.robot.name]["dof_effort_target"][jn] for jn in joint_names]
                effort_targets.append(effort_values)

        if effort_targets:
            return torch.tensor(effort_targets, dtype=torch.float32)
        return None

    def _get_control_mode(self, obj_name: str) -> str:
        """Get the control mode for the object."""
        if hasattr(self.object_dict[obj_name], "control_type"):
            control_types = list(set(self.object_dict[obj_name].control_type.values()))
            if len(control_types) > 1:
                raise ValueError(f"Multiple control types not supported: {control_types}")
            return control_types[0] if control_types else "position"
        return "position"

    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joints: list[RigidJoint] = self.object_inst_dict[obj_name].joints

            joint_names = []

            for j in joints:
                try:
                    if j.dofs_idx_local[0] is not None and j.name != self.object_inst_dict[obj_name].base_joint.name:
                    #if (j.dof_idx_local is not None and j.name != self.object_inst_dict[obj_name].base_joint.name)
                        joint_names.append(j.name)
                except IndexError:
                    if j.dofs_idx_local[0] is not None and j.name != "root_joint":
                        joint_names.append(j.name)

            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    @property
    def num_envs(self) -> int:
        return self.scene_inst.n_envs

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return gs.device


GenesisEnv = GymEnvWrapper(GenesisHandler)

from __future__ import annotations

import genesis as gs
import numpy as np
import torch
import random
gs.init(backend=gs.gpu,logging_level=gs._logging.WARNING)  # TODO: add option for cpu

import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint
from genesis.vis.camera import Camera
from loguru import logger as log


from metasim.cfg.sensors import PinholeCameraCfg, NyxGaussianSplatCameraCfg, GyroSensorCfg, CommandCfg
from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg, _FileBasedMixin
from metasim.cfg.scenario import ScenarioCfg
from metasim.queries.base import BaseQueryType
from metasim.sim import BaseSimHandler, GymEnvWrapper
from .gaussian_splat import (
    NyxGaussianSplatRuntime,
    patch_nyx_rigid_solver_compat,
    prepare_urdfs_for_nyx,
)
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
        self.camera_debug_dots: dict[str, tuple[RigidEntity, object, tuple[float, float, float]]] = {}
        self._nyx_splat = NyxGaussianSplatRuntime()
        self.cached_top_offsets: dict[str, torch.Tensor] = {}
        self._cache_joint_names: dict[str, list[str]] = {}

    def launch(self) -> None:
        show_viewer = not self.headless
        print(show_viewer," show_viewer")
        # zde změna ve vypisování logů - genesis je moc hlučný (nevypisujíse info logy ani debug)
        self.scene_inst = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 1 / 100,
                substeps=1,
            ),  # TODO: substeps > 1 doesn't work
            # MetaSim robot configurations already carry this option, but it
            # was previously ignored by the Genesis backend.  Genesis enables
            # per-entity self collision by default, whereas Unitree's G1
            # locomotion setup has it disabled.  This is especially important
            # for the full 43-DoF model: unintended arm/hand/leg contacts can
            # inject forces that the 12-DoF walking policy never observed.
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=bool(self.robot.enabled_self_collisions),
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=self.scenario.num_envs),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=show_viewer,
        )
        #print(not self.headless)
        prepare_urdfs_for_nyx(self.robot, self.scenario.objects, self.cameras)
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
                file=self.robot.urdf_path,
                merge_fixed_links=self.robot.collapse_fixed_joints,
                fixed=self.robot.fix_base_link,
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
                    gs.morphs.URDF(file=obj.urdf_path, fixed=obj.fix_base_link, scale=obj.scale, merge_fixed_links=obj.colapse_fixed_joints, batch_fixed_verts=obj.batch_fixed_verts),

                )
            else:
                raise NotImplementedError(f"Object type {type(obj)} not supported")
            self.object_inst_dict[obj.name] = obj_inst

        # ## Add cameras
        # for camera in self.cameras:
        #     camera_inst = self.scene_inst.add_camera(
        #         res=(camera.width, camera.height),
        #         pos=camera.pos,
        #         lookat=camera.look_at,
        #         fov=camera.vertical_fov,
        #     )
        #     self.camera_inst_dict[camera.name] = camera_inst
        ## Add cameras
        for camera in self.cameras:
            mount_entity = None
            mount_link = None

            attached_obj_name = getattr(camera, "mount_to", None)
            attached_link_name = getattr(camera, "mount_link", None)
            if isinstance(camera, NyxGaussianSplatCameraCfg) and camera.debug_detach_from_mount:
                attached_obj_name = None
                attached_link_name = None

            if attached_obj_name and attached_obj_name in self.object_inst_dict:
                mount_entity = self.object_inst_dict[attached_obj_name]
                # Získáme lokální offsety
                pos = getattr(camera, "mount_pos", (0.05, 0.0, 0.0))
                quat = getattr(camera, "mount_quat", (1.0, 0.0, 0.0, 0.0))

                for link in mount_entity.links:
                    if link.name == attached_link_name:
                        mount_link = link
                        break

            if isinstance(camera, NyxGaussianSplatCameraCfg):
                camera_inst = self._nyx_splat.make_camera(
                    self.scene_inst,
                    camera,
                    mount_entity,
                    mount_link,
                    robot=self.robot,
                    objects=self.scenario.objects,
                )
                if mount_link is not None:
                    self._add_camera_debug_dot(
                        camera.name,
                        mount_link,
                        getattr(camera, "mount_pos", None) or (0.05, 0.0, 0.0),
                    )
                log.info(f"Nyx Gaussian splat camera '{camera.name}' added")
            elif mount_entity and mount_link:
                camera_inst = self.scene_inst.add_camera(
                    res=(camera.width, camera.height),
                    fov=camera.vertical_fov,
                )

                # --- OPRAVA: Převod pos a quat na 4x4 matici (offset_T) ---
                pos_t = torch.tensor(pos, dtype=gs.tc_float, device=gs.device)
                quat_t = torch.tensor(quat, dtype=gs.tc_float, device=gs.device)

                # Vytvoření 4x4 offset matice
                offset_T = gu.trans_quat_to_T(pos_t, quat_t)

                # Připojení kamery pomocí matice
                camera_inst.attach(mount_link, offset_T=offset_T)
                # --- NOVÉ: DEBUGOVACÍ KULIČKA (BEZ ROTACE) ---
                self._add_camera_debug_dot(camera.name, mount_link, pos)

                log.info(f"Camera '{camera.name}' attached to {attached_obj_name}::{attached_link_name}")
            else:
                # Statická kamera
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
        self._apply_robot_actuator_properties()
        log.info(
            "Genesis rigid-body configuration: self collisions={} (from robot config)",
            bool(self.robot.enabled_self_collisions),
        )
        if any(isinstance(camera, NyxGaussianSplatCameraCfg) and camera.render_sim_geometry for camera in self.cameras):
            patch_nyx_rigid_solver_compat(self.scene_inst)
        self._previous_dof_pos_target: dict[str, torch.Tensor] = {}
        self._previous_dof_vel_target: dict[str, torch.Tensor] = {}

        self._build_link_map()

    def _actuated_dof_indices(self, obj_name: str) -> list[int]:
        """Return local DOF indices in the same order as ``get_joint_names``."""
        cache_key = f"_dof_idx_local_{obj_name}"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)

        obj_inst = self.object_inst_dict[obj_name]
        try:
            base_joint_name = obj_inst.base_joint.name
        except (AttributeError, IndexError):
            # A fixed-base Genesis entity may not expose a base joint at all.
            base_joint_name = None
        dof_idx_local = []
        for joint in obj_inst.joints:
            indices = joint.dofs_idx_local
            if (
                len(indices) > 0
                and indices[0] is not None
                and joint.name != base_joint_name
                and joint.name != "root_joint"
            ):
                dof_idx_local.append(indices[0])
        setattr(self, cache_key, dof_idx_local)
        return dof_idx_local

    @staticmethod
    def _first_property_row(values: torch.Tensor) -> torch.Tensor:
        """Normalize a global/per-environment Genesis property to one row."""
        values = values.detach().cpu()
        return values[0] if values.ndim > 1 else values

    def _apply_robot_actuator_properties(self) -> None:
        """Apply configured PD gains and torque limits after Genesis is built.

        Genesis does not import :class:`BaseActuatorCfg` by itself. Position
        targets therefore used Genesis/URDF defaults before this explicit
        setup, even though stiffness and damping were present in the MetaSim
        robot configuration.
        """
        joint_names = self.get_joint_names(self.robot.name, sort=False)
        dof_indices = self._actuated_dof_indices(self.robot.name)
        if len(joint_names) != len(dof_indices):
            raise RuntimeError(
                f"Genesis joint/DOF mapping mismatch for {self.robot.name}: "
                f"{len(joint_names)} joints, {len(dof_indices)} DOFs"
            )

        joint_to_dof = dict(zip(joint_names, dof_indices))
        missing_actuators = [name for name in joint_names if name not in self.robot.actuators]
        if missing_actuators:
            log.warning(
                "No actuator configuration for Genesis joints: {}",
                ", ".join(missing_actuators),
            )

        def configured(property_name: str):
            names, indices, values = [], [], []
            for name in joint_names:
                actuator = self.robot.actuators.get(name)
                value = getattr(actuator, property_name, None) if actuator is not None else None
                if value is not None:
                    names.append(name)
                    indices.append(joint_to_dof[name])
                    values.append(float(value))
            return names, indices, values

        _, kp_indices, kp_values = configured("stiffness")
        _, kd_indices, kd_values = configured("damping")
        if kp_indices:
            self.robot_inst.set_dofs_kp(kp_values, dofs_idx_local=kp_indices)
        if kd_indices:
            self.robot_inst.set_dofs_kv(kd_values, dofs_idx_local=kd_indices)

        torque_names, torque_indices, torque_values = configured("torque_limit")
        configured_torque = set(torque_names)
        for name in joint_names:
            if name in configured_torque:
                continue
            value = getattr(self.robot, "torque_limits", {}).get(name)
            if value is not None:
                torque_indices.append(joint_to_dof[name])
                torque_values.append(float(value))
        if torque_indices:
            self.robot_inst.set_dofs_force_range(
                [-value for value in torque_values],
                torque_values,
                dofs_idx_local=torque_indices,
            )

        # Read the values back from Genesis. This is deliberately logged once
        # at launch so the output shows what the simulator actually received.
        all_kp = self._first_property_row(self.robot_inst.get_dofs_kp(dofs_idx_local=dof_indices))
        all_kd = self._first_property_row(self.robot_inst.get_dofs_kv(dofs_idx_local=dof_indices))
        force_lower, force_upper = self.robot_inst.get_dofs_force_range(dofs_idx_local=dof_indices)
        force_lower = self._first_property_row(force_lower)
        force_upper = self._first_property_row(force_upper)
        self.applied_actuator_properties = {
            name: {
                "dof_index": int(dof_index),
                "kp": float(all_kp[i]),
                "kd": float(all_kd[i]),
                "force_lower": float(force_lower[i]),
                "force_upper": float(force_upper[i]),
            }
            for i, (name, dof_index) in enumerate(zip(joint_names, dof_indices))
        }
        rows = ["joint | dof | Kp | Kd | force range [Nm]"]
        for name in joint_names:
            values = self.applied_actuator_properties[name]
            rows.append(
                f"{name} | {values['dof_index']} | {values['kp']:.3f} | "
                f"{values['kd']:.3f} | [{values['force_lower']:.3f}, {values['force_upper']:.3f}]"
            )
        log.info("Genesis actuator properties applied (printed once):\n{}", "\n".join(rows))

    def _add_camera_debug_dot(self, camera_name: str, mount_link, local_pos) -> None:
        debug_dot = self.scene_inst.add_entity(
            gs.morphs.Sphere(radius=0.03),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 1.0)),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        self.camera_debug_dots[camera_name] = (debug_dot, mount_link, tuple(float(v) for v in local_pos))

    def _update_camera_debug_dot(self, camera_name: str, env_ids: list[int] | None = None) -> None:
        if camera_name not in self.camera_debug_dots:
            return

        debug_dot, link, local_pos = self.camera_debug_dots[camera_name]
        link_pos = link.get_pos(envs_idx=env_ids)
        link_quat = link.get_quat(envs_idx=env_ids)
        link_T = gu.trans_quat_to_T(link_pos, link_quat)
        pos_t = torch.tensor(local_pos, dtype=gs.tc_float, device=gs.device)
        pos_homogeneous = torch.nn.functional.pad(pos_t, (0, 1), value=1.0)
        new_pos = torch.matmul(link_T, pos_homogeneous)[:, :3]
        debug_dot.set_pos(new_pos, envs_idx=env_ids)


    def _build_link_map(self):
        """
        Vytvoří slovník, který mapuje globální index linku (int) na dvojici (název_objektu, název_linku).
        Toto je nutné, protože Genesis vrací kolize jen jako čísla.
        """
        self.global_link_map = {}

        # Projdeme všechny objekty, které jsme si uložili
        for obj_name, entity in self.object_inst_dict.items():
            if isinstance(entity, RigidEntity):
                for link in entity.links:
                    # link.idx je globální index v solveru pro první environment (env 0)
                    if hasattr(link, 'idx'):
                        self.global_link_map[link.idx] = (obj_name, link.name)

        # Zjistíme celkový počet rigidních těles v jednom prostředí pro výpočet offsetů u n_envs > 1
        # (Pokud máte jen 1 env, není to kritické, ale pro batching je to nutné)
        try:
            # Tohle je heuristika, najdeme nejvyšší index a přičteme 1
            max_idx = max(self.global_link_map.keys()) if self.global_link_map else 0
            self.num_bodies_per_env = max_idx + 1
        except Exception:
            self.num_bodies_per_env = 1000 # Fallback

    def get_contact(self, contact_data: dict) -> list[dict]:
        """
        Zpracuje surová data z get_contacts() a vrátí seznam slovníků s informacemi o kolizích.
        Přidává informace o síle kontaktu.
        """
        if contact_data is None:
            return []

        readable_collisions = []

        # Přesun dat na CPU
        link_a = contact_data['link_a'].cpu().numpy()
        link_b = contact_data['link_b'].cpu().numpy()
        valid_mask = contact_data['valid_mask'].cpu().numpy()

        # --- NOVÉ: Získání sil ---
        # Genesis vrací tensor 'force' o rozměrech [n_envs, max_contacts, 3]
        if 'force_b' in contact_data:
            raw_forces = contact_data['force_b'].cpu().numpy()
        else:
            # Fallback pro případ, že solver síly nevrací (např. jen detekce kolizí)
            raw_forces = np.zeros((*link_a.shape, 3))

        # Získáme indexy, kde je kolize platná
        env_indices, contact_indices = np.where(valid_mask)

        # Iterujeme pouze přes aktivní kolize
        for env_id, i in zip(env_indices, contact_indices):

            # Získáme raw indexy
            idx_a_raw = int(link_a[env_id, i])
            idx_b_raw = int(link_b[env_id, i])

            # Přepočet na základní index
            base_idx_a = idx_a_raw % self.num_bodies_per_env
            base_idx_b = idx_b_raw % self.num_bodies_per_env

            def resolve_name(idx):
                if idx == 0:
                    return "World", "Ground"
                return self.global_link_map.get(idx, (f"Unknown_{idx}", "unknown"))

            obj_a, link_name_a = resolve_name(base_idx_a)
            obj_b, link_name_b = resolve_name(base_idx_b)

            # Ignorování self-collision
            if obj_a == obj_b:
                continue

            # --- ZPRACOVÁNÍ SÍLY ---
            # Získáme vektor síly pro tento konkrétní kontakt
            force_vec = raw_forces[env_id, i] # shape (3,)
            force_magnitude = np.linalg.norm(force_vec)

            # Vytvoření slovníku
            collision_entry = {
                "env_id": int(env_id),
                "body_a": obj_a,
                "link_a": link_name_a,
                "body_b": obj_b,
                "link_b": link_name_b,
                "formatted": f"{obj_a}::{link_name_a} <-> {obj_b}::{link_name_b}",
                "force": float(force_magnitude),      # Skalární velikost
                "force_vec": force_vec.tolist()       # Vektor [x, y, z]
            }

            readable_collisions.append(collision_entry)

        return readable_collisions





    def _get_states(self, env_ids: list[int] | None = None) -> list[EnvState]:
        extra = {}

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        object_states = {}
        for obj in self.objects:
            obj_inst = self.object_inst_dict[obj.name]
            joints_names_arr = np.array(self.get_joint_names(obj.name))
            if isinstance(obj, ArticulationObjCfg):
                #joint_reindex = self.get_joint_reindex(obj.name)
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
                    joint_names=joints_names_arr,
                    body_names=self.get_body_names(obj.name),
                    body_state=self.get_body_states(obj.name, envs_idx=env_ids),
                    joint_pos=obj_inst.get_dofs_position(envs_idx=env_ids),#[:, joint_reindex],
                    joint_vel=obj_inst.get_dofs_velocity(envs_idx=env_ids),#[:, joint_reindex],
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
            #print(obj_inst.control_dofs_force())
            object_states[obj.name] = state

        robot_states = {}
        for obj in [self.robot]:

            joints_names_arr = np.array(self.get_joint_names(obj.name))
            #joints_names_reindexed = joints_names_arr[joint_reindex].tolist()
            obj_inst = self.object_inst_dict[obj.name]
            raw_contact = obj_inst.get_contacts()
            #readable_contacts = self.get_contact(raw_contact)
            if self._previous_dof_pos_target is None or obj.name not in self._previous_dof_pos_target:
                self._previous_dof_pos_target[obj.name] = torch.zeros_like(obj_inst.get_dofs_position(envs_idx=env_ids))
            joint_reindex = self.get_joint_reindex(obj.name)
            if obj_inst.base_link.is_fixed:
                joint_pos = obj_inst.get_dofs_position(envs_idx=env_ids)
                joint_vel = obj_inst.get_dofs_velocity(envs_idx=env_ids)
            else:
                joint_pos = obj_inst.get_dofs_position(envs_idx=env_ids)[:, 6:] # pokud není fixní, první 6 DOF jsou pro base link, takže je ignorujeme
                joint_vel = obj_inst.get_dofs_velocity(envs_idx=env_ids)[:, 6:]
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
                joint_names=joints_names_arr,
                body_names=self.get_body_names(obj.name),
                body_state=self.get_body_states(obj.name, envs_idx=env_ids),
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                joint_pos_target=self._previous_dof_pos_target[obj.name],
                joint_effort_target=self._get_effort_targets(),
                contact=raw_contact,
                joint_vel_target=None # TODO




                # if self._get_control_mode(obj.name) == "effort"
                # else None,
            )
            robot_states[obj.name] = state

        # camera_states = {}
        # for camera in self.cameras:
        #     camera_inst = self.camera_inst_dict[camera.name]
        #     rgb, depth, _, _ = camera_inst.render(depth=True)
        #     state = CameraState(
        #         rgb=torch.from_numpy(rgb.copy()).unsqueeze(0).repeat_interleave(self.num_envs, dim=0),  # XXX
        #         depth=torch.from_numpy(depth.copy()).unsqueeze(0).repeat_interleave(self.num_envs, dim=0),  # XXX
        #     )
        #     camera_states[camera.name] = state
        camera_states = {}
        for camera in self.cameras:
            camera_inst = self.camera_inst_dict[camera.name]

            if isinstance(camera, NyxGaussianSplatCameraCfg):
                if camera.render_sim_geometry:
                    patch_nyx_rigid_solver_compat(self.scene_inst, camera_inst)
                else:
                    self._nyx_splat.step_standalone_scene(camera.name, self.object_inst_dict, camera_inst)
                self._update_camera_debug_dot(camera.name, env_ids)
                camera_inst._stale = True
                rgb = camera_inst.read().rgb
                self._nyx_splat.show_debug_frame(camera_inst, camera, rgb)
                if rgb.dim() == 3:
                    rgb = rgb.unsqueeze(0)
                if camera.render_sim_geometry and rgb.shape[0] != self.num_envs:
                    raise RuntimeError(
                        f"Nyx camera '{camera.name}' returned {rgb.shape[0]} env frame(s), "
                        f"but scenario.num_envs={self.num_envs}. "
                        "For parallel visual training, render_sim_geometry=true must return one RGB frame per env."
                    )
                if not camera.render_sim_geometry and self.num_envs > 1 and rgb.shape[0] != self.num_envs:
                    log.warning(
                        f"Nyx standalone camera '{camera.name}' returned {rgb.shape[0]} frame(s) for "
                        f"num_envs={self.num_envs}; standalone mode is intended for single-env debug/eval."
                    )
                state = CameraState(
                    rgb=rgb,
                    depth=None,
                )
                camera_states[camera.name] = state
                continue

            if getattr(camera_inst, "_attached_link", None) is not None:
                camera_inst.move_to_attach()

                # --- NOVÉ: PŘESUN KULIČKY (JEN POZICE) ---
                self._update_camera_debug_dot(camera.name, env_ids)
                # -----------------------------------------

            # Render obrazu
            rgb, depth, _, _ = camera_inst.render(depth=True)

            state = CameraState(
                rgb=torch.from_numpy(rgb.copy()).unsqueeze(0).repeat_interleave(self.num_envs, dim=0),
                depth=torch.from_numpy(depth.copy()).unsqueeze(0).repeat_interleave(self.num_envs, dim=0),
            )
            camera_states[camera.name] = state
        sensors = {}
        for sensor in self.scenario.sensors:
            if isinstance(sensor, GyroSensorCfg):
                gyro_data = sensor.get_data(robot_states,envs_ids=env_ids)  # shape (num_envs, 3)
                gyro_tensor = torch.tensor(gyro_data, dtype=torch.float32).unsqueeze(0)
                sensors[sensor.name] = gyro_tensor
            elif isinstance(sensor, CommandCfg):
                command_data = sensor.get_command()
                #command_tensor = torch.tensor(command_data, dtype=torch.float32).unsqueeze(1)
                sensors[sensor.name] = command_data
            else:
                log.warning(f"Unknown sensor type: {sensor.cfg_type}, skipping...")

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, sensors=sensors, extras=extra)


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
            if isinstance(obj, RigidObjCfg) and obj.fix_base_link and not (len(env_ids) == self.num_envs):
                continue  # Ignorujeme nastavení pozice pro pevné objekty, protože to může způsobit problémy se stabilitou simulace
            # --- Base link position ---
            pos_tensor = torch.stack([states_flat[eid][obj.name]["pos"] for eid in env_ids])  # [N,3]
            pos = pos_tensor.cpu().numpy()
            obj_inst.set_pos(pos, envs_idx=env_ids, relative=False)
            # --- Base link rotation ---
            quat_tensor = torch.stack([states_flat[eid][obj.name]["rot"] for eid in env_ids])  # [N,4]
            #TODO toto se mi nelíbí
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
                    #print("DEBUG: no joints for", obj.name)
                    continue
                else:
                    dof_pos = np.array(
                        [
                            [
                                states_flat[env_id][obj.name]["dof_pos"][joint.name]
                                for joint in obj_inst.joints
                                if joint.name != "root_joint"
                            ]
                            for env_id in env_ids
                        ],
                        dtype=np.float32,
                    )
                    if dof_pos.dtype != np.float32:
                        dof_pos = dof_pos.astype(np.float32)
                    base_pos = obj_inst.get_pos(envs_idx=env_ids)   # [N,3]
                    base_quat = obj_inst.get_quat(envs_idx=env_ids) # [N,4]

                    root_state = np.concatenate([base_pos.detach().cpu().numpy(), base_quat.detach().cpu().numpy()], axis=1)  # [N,7]
                    #full_qpos = np.concatenate([root_state, dof_pos], axis=1)   # [N, 7 + n_joints]

                    expected_qs = getattr(obj_inst, "n_qs", None)
                    if expected_qs == dof_pos.shape[1]:
                        qpos_to_set = dof_pos
                    elif expected_qs == 7 + dof_pos.shape[1]:
                        qpos_to_set = np.concatenate([root_state, dof_pos], axis=1)
                    elif expected_qs == 7:
                        qpos_to_set = root_state
                    else:
                        raise RuntimeError(...)
                    obj_inst.set_qpos(qpos_to_set, envs_idx=env_ids)


    def _set_states_advanced(self, states: list[EnvState], env_ids: list[int] | None = None) -> None:
        """
        Pokročilé nastavení stavu scény. Nastavuje pozice (pos, quat, joints)
        I RYCHLOSTI (linear vel, angular vel, joint vel) pomocí přímého zápisu do DOFs.
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # Replikace stavů pokud je jich méně než prostředí
        if len(states) < self.num_envs:
            states = states * (len(env_ids) // len(states))

        states_flat = [state["objects"] | state["robots"] for state in states]

        for obj in self.objects + [self.robot]:
            obj_inst = self.object_inst_dict[obj.name]
            obj_name = obj.name

            # 1. Příprava dat z EnvState (vše jako numpy arrays [N, ...])
            # -----------------------------------------------------------
            # Root Position [N, 3]
            pos = torch.stack([states_flat[eid][obj_name]["pos"] for eid in env_ids]).cpu().numpy()

            # Root Rotation [N, 4]
            quat_tensor = torch.stack([states_flat[eid][obj_name]["rot"] for eid in env_ids])
            quat = quat_tensor.cpu().numpy()
            # Normalizace quaternionů (kritické pro stabilitu simulace)
            norms = np.linalg.norm(quat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            quat = quat / norms

            # Root Velocity (Linear) [N, 3] - defaultně nuly
            vel = torch.stack([
                states_flat[eid][obj_name].get("vel", torch.zeros(3)) for eid in env_ids
            ]).cpu().numpy()

            # Root Velocity (Angular) [N, 3] - defaultně nuly
            ang = torch.stack([
                states_flat[eid][obj_name].get("ang", torch.zeros(3)) for eid in env_ids
            ]).cpu().numpy()

            # Joint Positions & Velocities
            joint_names = self.get_joint_names(obj_name, sort=False)

            # Filtrujeme jointy, které nejsou 'root_joint' (Genesis internal)
            # Pokud je objekt RigidObj bez kloubů, joint_names bude prázdné
            valid_joints = [j for j in obj_inst.joints if j.name != "root_joint"]

            if valid_joints:
                # [N, n_joints]
                dof_pos_arr = np.array([
                    [states_flat[eid][obj_name]["dof_pos"].get(j.name, 0.0) for j in valid_joints]
                    for eid in env_ids
                ], dtype=np.float32)

                # [N, n_joints]
                dof_vel_arr = np.array([
                    [states_flat[eid][obj_name].get("dof_vel", {}).get(j.name, 0.0) for j in valid_joints]
                    for eid in env_ids
                ], dtype=np.float32)
            else:
                dof_pos_arr = np.empty((len(env_ids), 0), dtype=np.float32)
                dof_vel_arr = np.empty((len(env_ids), 0), dtype=np.float32)

            # 2. Logika sestavení QPOS (Positions)
            # ------------------------------------
            # Genesis vyžaduje set_qpos s vektorem délky `n_qs`.
            # Pro floating base (volný objekt) je n_qs = 7 (3 pos + 4 quat) + n_joints
            # Pro fixed base (připevněný robot) je n_qs = n_joints

            expected_qs = getattr(obj_inst, "n_qs", 0)
            root_pos_state = np.concatenate([pos, quat], axis=1) # [N, 7]

            qpos_to_set = None

            # A) Fixed base (jen klouby)
            if expected_qs == dof_pos_arr.shape[1]:
                qpos_to_set = dof_pos_arr

            # B) Floating base (root + klouby)
            elif expected_qs == 7 + dof_pos_arr.shape[1]:
                qpos_to_set = np.concatenate([root_pos_state, dof_pos_arr], axis=1)

            # C) Pouze Floating base bez dalších kloubů (např. kostka)
            elif expected_qs == 7:
                qpos_to_set = root_pos_state

            if qpos_to_set is not None:
                # Používáme set_qpos, který je v rigid_entity.py
                # Nastavujeme zero_velocity=False, protože rychlost nastavíme explicitně níže
                obj_inst.set_qpos(qpos_to_set, envs_idx=env_ids, zero_velocity=False)

            # 3. Logika sestavení DOFS VELOCITY (Velocities)
            # ----------------------------------------------
            # Genesis vyžaduje set_dofs_velocity s vektorem délky `n_dofs`.
            # Pro floating base je n_dofs = 6 (3 lin + 3 ang) + n_joints
            # Root DOFs jsou v Genesis typicky: [vel_x, vel_y, vel_z, ang_x, ang_y, ang_z]

            expected_dofs = getattr(obj_inst, "n_dofs", 0)
            root_vel_state = np.concatenate([vel, ang], axis=1) # [N, 6]

            qvel_to_set = None

            # A) Fixed base (jen klouby)
            if expected_dofs == dof_vel_arr.shape[1]:
                qvel_to_set = dof_vel_arr

            # B) Floating base (root 6DOF + klouby)
            elif expected_dofs == 6 + dof_vel_arr.shape[1]:
                qvel_to_set = np.concatenate([root_vel_state, dof_vel_arr], axis=1)

            # C) Pouze Floating base (např. kostka, 6DOF)
            elif expected_dofs == 6:
                qvel_to_set = root_vel_state

            if qvel_to_set is not None:
                # Používáme set_dofs_velocity
                # POZOR: RigidEntity nemá set_vel/set_ang, toto je jediný způsob jak nastavit rychlost báze.
                obj_inst.set_dofs_velocity(qvel_to_set, envs_idx=env_ids)


    def set_dof_targets(self, obj_name: str, actions: list[Action] | torch.Tensor | np.ndarray) -> None:
        self._actions_cache = actions
        control_mode = self._get_control_mode(obj_name)

        # --- 1. CACHE INDEXŮ (odstranění for-cyklu z každého kroku simulace) ---
        dof_idx_local = self._actuated_dof_indices(obj_name)

        # --- 2. OPTIMALIZOVANÝ PŘEVOD NA TORCH TENSOR ---
        # A) Přišel rovnou PyTorch Tensor (Nejrychlejší)
        if isinstance(actions, torch.Tensor):
            target_tensor = actions.to(device=self.device, dtype=torch.float32)

        # B) Přišel Numpy Array přímo ze Stable Baselines 3
        elif isinstance(actions, np.ndarray):
            target_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)

        # C) Fallback: Přišel starý formát list[dict]
        else:
            joint_names = self.get_joint_names(obj_name, sort=False)
            target_key = "dof_effort_target" if control_mode == "effort" else "dof_pos_target"
            raw_targets = [
                [actions[env_id][obj_name][target_key][jn] for jn in joint_names]
                for env_id in range(self.num_envs)
            ]
            target_tensor = torch.tensor(raw_targets, dtype=torch.float32, device=self.device)

        # --- 3. PŘÍMÉ VOLÁNÍ GENESIS (bez Python listů) ---
        if control_mode == "effort":
            self.object_inst_dict[obj_name].control_dofs_force(
                force=target_tensor,
                dofs_idx_local=dof_idx_local,
            )
        else:
            self.object_inst_dict[obj_name].control_dofs_position(
                position=target_tensor,
                dofs_idx_local=dof_idx_local,
            )

        # Uložení předchozího stavu přímo na GPU
        self._previous_dof_pos_target[obj_name] = target_tensor

    def refresh_render(self):
        """Refresh the render."""
        if not self.headless:
            self.scene_inst.viewer.update()
        self.scene_inst.visualizer.update()

    def close(self):
        pass


    def _get_effort_targets(self) -> torch.Tensor | None:
        """Get the effort targets from cached actions."""
        # Bezpečný check, zda cache existuje a není None (zabrání ValueError u numpy polí)
        if getattr(self, "_actions_cache", None) is None:
            return None

        # --- NOVÁ RYCHLÁ CESTA PRO TENSORY A NUMPY POLE ---
        if isinstance(self._actions_cache, (np.ndarray, torch.Tensor)):
            # Pokud robot používá "effort" control mode, samotné akce jsou effort targets
            if self._get_control_mode(self.robot.name) == "effort":
                if isinstance(self._actions_cache, np.ndarray):
                    return torch.tensor(self._actions_cache, dtype=torch.float32, device=self.device)
                return self._actions_cache.clone().to(dtype=torch.float32, device=self.device)
            else:
                # Pokud používá "position" control mode, tak effort targety z akcí nevyčítáme
                return None

        # --- STARÁ CESTA PRO LIST[DICT] ---
        if isinstance(self._actions_cache, list):
            if len(self._actions_cache) == 0:
                return None

            joint_names = self.get_joint_names(self.robot.name, sort=False)
            effort_targets = []
            for action in self._actions_cache:
                # Ochrana proti nekompletním dictům
                if isinstance(action, dict) and self.robot.name in action:
                    if "dof_effort_target" in action[self.robot.name] and action[self.robot.name]["dof_effort_target"]:
                        effort_values = [action[self.robot.name]["dof_effort_target"][jn] for jn in joint_names]
                        effort_targets.append(effort_values)

            if effort_targets:
                return torch.tensor(effort_targets, dtype=torch.float32, device=self.device)

        return None
    def _get_control_mode(self, obj_name: str) -> str:
        """Get the control mode for the object."""
        if hasattr(self.object_dict[obj_name], "control_type"):
            control_types = list(set(self.object_dict[obj_name].control_type.values()))
            if len(control_types) > 1:
                raise ValueError(f"Multiple control types not supported: {control_types}")
            return control_types[0] if control_types else "position"
        return "position"

    def get_joint_names(self, obj_name: str, sort: bool = False) -> list[str]:
        cache_key = f"{obj_name}_{sort}"
        if cache_key in getattr(self, "_cached_joint_names", {}):
            return self._cached_joint_names[cache_key]
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joints: list[RigidJoint] = self.object_inst_dict[obj_name].joints
            try:
                base_joint_name = self.object_inst_dict[obj_name].base_joint.name
            except (AttributeError, IndexError):
                base_joint_name = None

            joint_names = []

            for j in joints:
                indices = j.dofs_idx_local
                if (
                    len(indices) > 0
                    and indices[0] is not None
                    and j.name != base_joint_name
                    and j.name != "root_joint"
                ):
                    joint_names.append(j.name)

            if sort:
                joint_names.sort()
            # Uložit do cache!
            if not hasattr(self, "_cached_joint_names"):
                self._cached_joint_names = {}
            self._cached_joint_names[cache_key] = joint_names
            return joint_names
        else:
            return []
    def _simulate(self):

        for _ in range(self.scenario.decimation):
            self.scene_inst.step()

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

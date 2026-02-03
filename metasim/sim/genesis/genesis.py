from __future__ import annotations

import genesis as gs
import numpy as np
import torch
import random

from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint
from genesis.vis.camera import Camera
from loguru import logger as log


from metasim.cfg.sensors import PinholeCameraCfg, GyroSensorCfg, CommandCfg
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
        show_viewer = not self.headless
        print(show_viewer," show_viewer")
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
            show_viewer=show_viewer,
        )
        #print(not self.headless)
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
                merge_fixed_links=self.robot.collapse_fixed_joints
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
                    gs.morphs.URDF(file=obj.urdf_path, fixed=obj.fix_base_link, scale=obj.scale, merge_fixed_links=obj.colapse_fixed_joints),
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
        self._previous_dof_pos_target: dict[str, torch.Tensor] = {}
        self._previous_dof_vel_target: dict[str, torch.Tensor] = {}

        self._build_link_map()
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
    # def get_contact(self, contact_data: dict) -> list[dict]:
    #     """
    #     Zpracuje surová data z get_contacts() a vrátí seznam slovníků s informacemi o kolizích.
    #     Index 0 je explicitně mapován jako Podlaha (Ground).
    #     """
    #     if contact_data is None:
    #         return []

    #     readable_collisions = []

    #     # Přesun dat na CPU (tato operace je nutná, ale děláme ji hromadně)
    #     link_a = contact_data['link_a'].cpu().numpy()
    #     link_b = contact_data['link_b'].cpu().numpy()
    #     valid_mask = contact_data['valid_mask'].cpu().numpy()

    #     # --- OPTIMALIZACE: Získáme souřadnice jen tam, kde je kolize skutečná ---
    #     # np.where vrátí indexy (env_ids, contact_indices), kde je maska True
    #     # Díky tomu neprocházíme prázdná místa v poli.
    #     env_indices, contact_indices = np.where(valid_mask)

    #     # Iterujeme pouze přes aktivní kolize
    #     for env_id, i in zip(env_indices, contact_indices):

    #         # Získáme raw indexy
    #         idx_a_raw = int(link_a[env_id, i])
    #         idx_b_raw = int(link_b[env_id, i])

    #         # Přepočet na základní index (pro multi-env)
    #         base_idx_a = idx_a_raw % self.num_bodies_per_env
    #         base_idx_b = idx_b_raw % self.num_bodies_per_env

    #         # Pomocná funkce pro získání jména (řeší index 0 jako podlahu)
    #         def resolve_name(idx):
    #             if idx == 0:
    #                 return "World", "Ground"
    #             return self.global_link_map.get(idx, (f"Unknown_{idx}", "unknown"))

    #         obj_a, link_name_a = resolve_name(base_idx_a)
    #         obj_b, link_name_b = resolve_name(base_idx_b)

    #         # Ignorování "self-collision" (robot sám se sebou), pokud chcete
    #         if obj_a == obj_b:
    #             continue

    #         # Vytvoření slovníku
    #         collision_entry = {
    #             "env_id": int(env_id),
    #             "body_a": obj_a,
    #             "link_a": link_name_a,
    #             "body_b": obj_b,
    #             "link_b": link_name_b,
    #             "formatted": f"{obj_a}::{link_name_a} <-> {obj_b}::{link_name_b}"
    #         }
    #         readable_collisions.append(collision_entry)

    #     return readable_collisions

    def _get_states(self, env_ids: list[int] | None = None) -> list[EnvState]:


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
            readable_contacts = self.get_contact(raw_contact)
            if self._previous_dof_pos_target is None or obj.name not in self._previous_dof_pos_target:
                self._previous_dof_pos_target[obj.name] = torch.zeros_like(obj_inst.get_dofs_position(envs_idx=env_ids))
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
                joint_names=joints_names_arr,
                body_names=self.get_body_names(obj.name),
                body_state=self.get_body_states(obj.name, envs_idx=env_ids),
                joint_pos=obj_inst.get_dofs_position(envs_idx=env_ids)[:, 6:],
                joint_vel=obj_inst.get_dofs_velocity(envs_idx=env_ids)[:, 6:],
                joint_pos_target=self._previous_dof_pos_target[obj.name],
                joint_effort_target=self._get_effort_targets(),
                contact=readable_contacts,
                joint_vel_target=None # TODO




                # if self._get_control_mode(obj.name) == "effort"
                # else None,
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
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, sensors=sensors)


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
                if j.name == "door_hinge":
                    print("bagr")
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
        self._previous_dof_pos_target[obj_name] = torch.tensor(position, dtype=torch.float32)


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

    def get_joint_names(self, obj_name: str, sort: bool = False) -> list[str]:
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

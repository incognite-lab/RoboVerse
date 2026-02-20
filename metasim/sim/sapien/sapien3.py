"""Implemention of Sapien Handler.

This file contains the implementation of Sapien2Handler, which is a subclass of BaseSimHandler.
Sapien2Handler is used to handle the simulation environment using Sapien.
Currently using Sapien 2.2
"""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import sapien
import sapien.core as sapien_core
import torch
from loguru import logger as log
from packaging.version import parse as parse_version
from sapien.utils import Viewer

from metasim.cfg.objects import (
    ArticulationObjCfg,
    NonConvexRigidObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.cfg.robots import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.queries.base import BaseQueryType
from metasim.sim import BaseSimHandler, EnvWrapper, GymEnvWrapper
from metasim.types import Action, EnvState
from metasim.utils.math import quat_from_euler_np
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState
from metasim.cfg.sensors import PinholeCameraCfg, GyroSensorCfg, CommandCfg



class Sapien3Handler(BaseSimHandler):
    """Sapien3 Handler class."""

    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        assert parse_version(sapien.__version__) >= parse_version("3.0.0a0"), "Sapien3 is required"
        assert parse_version(sapien.__version__) < parse_version("4.0.0"), "Sapien3 is required"
        log.warning("Sapien3 is still under development, some metasim apis yet don't have sapien3 support")
        super().__init__(scenario, optional_queries)
        self.headless = scenario.headless
        self._actions_cache: list[Action] = []

    def _build_sapien(self):
        self.engine = sapien_core.Engine()  # Create a physical simulation engine
        self.renderer = sapien_core.SapienRenderer()  # Create a renderer

        scene_config = sapien_core.SceneConfig()
        # scene_config.default_dynamic_friction = self.physical_params.dynamic_friction
        # scene_config.default_static_friction = self.physical_params.static_friction
        # scene_config.contact_offset = self.physical_params.contact_offset
        # scene_config.default_restitution = self.physical_params.restitution
        # scene_config.enable_pcm = True
        # scene_config.solver_iterations = self.sim_params.num_position_iterations
        # scene_config.solver_velocity_iterations = self.sim_params.num_velocity_iterations
        scene_config.gravity = [0, 0, -9.81]
        # scene_config.bounce_threshold = self.sim_params.bounce_threshold

        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(self.scenario.sim_params.dt if self.scenario.sim_params.dt is not None else 1 / 100)
        ground_material = self.renderer.create_material()
        ground_material.base_color = np.array([202, 164, 114, 256]) / 256
        ground_material.specular = 0.5
        self.scene.add_ground(altitude=0, render_material=ground_material)

        self.loader = self.scene.create_urdf_loader()

        # Add agents
        self.object_ids: dict[str, sapien_core.Entity] = {}
        self.link_ids: dict[str, list[sapien.physx.PhysxArticulationLinkComponent]] = {}
        self._previous_dof_pos_target: dict[str, np.ndarray] = {}
        self._previous_dof_vel_target: dict[str, np.ndarray] = {}
        self._previous_dof_torque_target: dict[str, np.ndarray] = {}
        self.object_joint_order = {}
        self.camera_ids = {}

        for camera in self.cameras:
            # Create a camera entity in the scene
            camera_id = self.scene.add_camera(
                name=camera.name,
                width=camera.width,
                height=camera.height,
                fovy=np.deg2rad(camera.vertical_fov),
                near=camera.clipping_range[0],
                far=camera.clipping_range[1],
            )
            pos = np.array(camera.pos)
            look_at = np.array(camera.look_at)
            direction_vector = look_at - pos
            yaw = math.atan2(direction_vector[1], direction_vector[0])
            pitch = math.atan2(direction_vector[2], math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2))
            roll = 0
            camera_id.set_pose(sapien_core.Pose(p=pos, q=quat_from_euler_np(roll, -pitch, yaw)))
            self.camera_ids[camera.name] = camera_id

            # near, far = 0.1, 100
            # width, height = 640, 480
            # camera_id = self.scene.add_camera(
            #     name="camera",
            #     width=width,
            #     height=height,
            #     fovy=np.deg2rad(35),
            #     near=near,
            #     far=far,
            # )
            # camera_id.set_pose(sapien.Pose(p=[2, 0, 0], q=[0, 0, -1, 0]))
            # self.camera_ids[camera.name] = camera_id

        for object in [*self.objects, self.robot]:
            if isinstance(object, (ArticulationObjCfg, BaseRobotCfg)):
                self.loader.fix_root_link = object.fix_base_link
                self.loader.scale = object.scale[0]
                file_path = object.urdf_path
                curr_id = self.loader.load(file_path)
                curr_id.set_root_pose(sapien_core.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))

                self.object_ids[object.name] = curr_id

                active_joints = curr_id.get_active_joints()
                # num_joints = len(active_joints)
                cur_joint_names = []
                for id, joint in enumerate(active_joints):
                    joint_name = joint.get_name()
                    cur_joint_names.append(joint_name)
                self.object_joint_order[object.name] = cur_joint_names

                ### TODO
                # Change dof properties
                ###

                if isinstance(object, BaseRobotCfg):
                    active_joints = curr_id.get_active_joints()
                    for id, joint in enumerate(active_joints):
                        stiffness = object.actuators[joint.get_name()].stiffness
                        damping = object.actuators[joint.get_name()].damping
                        if stiffness is not None and damping is not None:
                            joint.set_drive_property(stiffness, damping)
                else:
                    active_joints = curr_id.get_active_joints()
                    for id, joint in enumerate(active_joints):
                        joint.set_drive_property(0, 0)

                # if agent.dof.init:
                #     robot.set_qpos(agent.dof.init)

                # if agent.dof.target:
                #     robot.set_drive_target(agent.dof.target)

            elif isinstance(object, PrimitiveCubeCfg):
                actor_builder = self.scene.create_actor_builder()
                # material = get_material(self.scene, agent.rigid_shape_property)
                actor_builder.add_box_collision(
                    half_size=object.half_size,
                    density=object.density,
                    # material=material,
                )
                actor_builder.add_box_visual(
                    half_size=object.half_size,
                    # color=object.color if object.color else [1.0, 1.0, 0.0],
                    material=sapien_core.render.RenderMaterial(
                        base_color=object.color[:3] + [1] if object.color else [1.0, 1.0, 0.0, 1.0]
                    ),
                )
                box = actor_builder.build(name="box")  # Add a box
                box.set_pose(sapien_core.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
                # box.set_damping(agent.rigid_shape_property.linear_damping, agent.rigid_shape_property.angular_damping)
                # if agent.vel:
                #     box.set_velocity(agent.vel)
                # if agent.ang_vel:
                #     box.set_angular_velocity(agent.ang_vel)
                # box.set_damping(agent.rigid_shape_property.linear_damping, agent.rigid_shape_property.angular_damping)
                # if agent.fix_base_link:
                #     box.lock_motion()
                # agent.instance = box
                self.object_ids[object.name] = box
                self.object_joint_order[object.name] = []

            elif isinstance(object, PrimitiveSphereCfg):
                actor_builder = self.scene.create_actor_builder()
                # material = get_material(self.scene, agent.rigid_shape_property)
                actor_builder.add_sphere_collision(radius=object.radius, density=object.density)
                actor_builder.add_sphere_visual(
                    radius=object.radius,
                    material=sapien_core.render.RenderMaterial(
                        base_color=object.color[:3] + [1] if object.color else [1.0, 1.0, 0.0, 1.0]
                    ),
                )
                sphere = actor_builder.build(name="sphere")  # Add a sphere
                sphere.set_pose(sapien_core.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
                # sphere.set_damping(
                #     agent.rigid_shape_property.linear_damping, agent.rigid_shape_property.angular_damping
                # )
                # if agent.vel:
                #     sphere.set_velocity(agent.vel)
                # if agent.ang_vel:
                #     sphere.set_angular_velocity(agent.ang_vel)
                # if agent.fix_base_link:
                #     sphere.lock_motion()
                # agent.instance = sphere
                self.object_ids[object.name] = sphere
                self.object_joint_order[object.name] = []

            elif isinstance(object, NonConvexRigidObjCfg):
                builder = self.scene.create_actor_builder()
                scene_pose = sapien_core.Pose(p=np.array(object.mesh_pose[:3]), q=np.array(object.mesh_pose[3:]))
                builder.add_nonconvex_collision_from_file(object.usd_path, scene_pose)
                builder.add_visual_from_file(object.usd_path, scene_pose)
                curr_id = builder.build_static(name=object.name)

                self.object_ids[object.name] = curr_id
                self.object_joint_order[object.name] = []

            elif isinstance(object, RigidObjCfg):
                self.loader.fix_root_link = object.fix_base_link
                self.loader.scale = object.scale[0]
                file_path = object.urdf_path
                curr_id: sapien_core.Entity
                try:
                    curr_id = self.loader.load(file_path)
                except Exception as e:
                    log.warning(f"Error loading {file_path}: {e}")
                    curr_id_list = self.loader.load_multiple(file_path)

                    # TODO:
                    # Don't understand why some urdf are treated as multiple entities
                    # Needs to figure out a better way to load!
                    for id in curr_id_list:
                        if len(id):
                            curr_id = id
                            break
                # builder = self.loader.load_file_as_articulation_builder(file_path)
                if isinstance(curr_id, list):
                    ## HACK
                    curr_id = curr_id[0]
                curr_id.set_pose(sapien_core.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))


                self.object_ids[object.name] = curr_id
                self.object_joint_order[object.name] = []

            if isinstance(object, (ArticulationObjCfg, BaseRobotCfg)):
                self.link_ids[object.name] = self.object_ids[object.name].get_links()
                self._previous_dof_pos_target[object.name] = np.zeros(
                    (len(self.object_joint_order[object.name]),), dtype=np.float32
                )
                self._previous_dof_vel_target[object.name] = np.zeros(
                    (len(self.object_joint_order[object.name]),), dtype=np.float32
                )
                self._previous_dof_torque_target[object.name] = np.zeros(
                    (len(self.object_joint_order[object.name]),), dtype=np.float32
                )
            else:
                self.link_ids[object.name] = []

            # elif agent.type == "capsule":
            #     actor_builder = self.scene.create_actor_builder()
            #     material = get_material(self.scene, agent.rigid_shape_property)
            #     actor_builder.add_capsule_collision(
            #         radius=agent.radius, half_length=agent.length, density=agent.density, material=material
            #     )
            #     actor_builder.add_capsule_visual(
            #         radius=agent.radius, half_length=agent.length, color=agent.color if agent.color else [1.0, 1.0, 1.0]
            #     )
            #     capsule = actor_builder.build(name="capsule")  # Add a capsule
            #     capsule.set_pose(sapien.Pose(p=[*agent.pos], q=np.asarray(agent.rot)))
            #     capsule.set_damping(
            #         agent.rigid_shape_property.linear_damping, agent.rigid_shape_property.angular_damping
            #     )
            #     if agent.vel:
            #         capsule.set_velocity(agent.vel)
            #     if agent.ang_vel:
            #         capsule.set_angular_velocity(agent.ang_vel)
            #     if agent.fix_base_link:
            #         capsule.lock_motion()
            #     agent.instance = capsule
        #nastavení jmen objektů

        for obj in self.object_ids:
            self.object_ids[obj].set_name(obj)
            print(self.object_ids[obj].name)
        # Add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        # self.scene.add_directional_light(
        #     self.sim_params.directional_light_pos, self.sim_params.directional_light_target
        # )

        # Create viewer and adjust camera position
        # if not self.viewer_params.headless:
        if not self.headless:
            self.viewer = Viewer(self.renderer)  # Create a viewer (window)
            self.viewer.set_scene(self.scene)  # Bind the viewer and the scene

        if not self.headless:
            camera_pos = np.array([1.5, -1.5, 1.5])
            camera_target = np.array([0.0, 0.0, 0.0])
            # if self.viewer_params.viewer_rot != None:
            #     camera_z = np.array([0.0, 0.0, 1.0])
            #     camera_rot = np.array(self.viewer_params.viewer_rot)
            #     camera_target = camera_pos + quat_apply(camera_rot, camera_z)
            # else:
            #     camera_target = np.array(self.viewer_params.target_pos)
            direction_vector = camera_target - camera_pos
            yaw = math.atan2(direction_vector[1], direction_vector[0])
            pitch = math.atan2(
                direction_vector[2], math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
            )  # 计算 roll 角（绕 X 轴的旋转角度）
            roll = 0
            # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
            # The principle axis of the camera is the x-axis
            self.viewer.set_camera_xyz(x=camera_pos[0], y=camera_pos[1], z=camera_pos[2])
            # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
            self.viewer.set_camera_rpy(r=roll, p=pitch, y=-yaw)
            # TODO:
            # UNABLE TO REMOVE THE AXIS AND CAMERA LINES FOR EARLY VERSIONS OF SAPIEN3
            # self.viewer.toggle_axes(show=False)
            # self.viewer.toggle_camera_lines(show=False)

        # self.viewer.set_fovy(self.viewer_params.horizontal_fov)
        # self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=self.viewer_params.fovy / 2) # the /2 is to align with isaac-gym

        # List for debug points
        self.debug_points = []
        self.debug_lines = []

        self.scene.update_render()
        for camera_name, camera_id in self.camera_ids.items():
            camera_id.take_picture()
    """def get_contact(self, contacts: list[sapien.physx.PhysxContact], filter_obj_name: str | None = None) -> list[dict]:

       # Zpracuje kontakty ze Sapien scény a převede je na čitelný formát shodný s Genesis.
       # Pokud je zadán filter_obj_name, vrátí jen kontakty týkající se daného objektu.

        readable_collisions = []

        # Získání instance objektu pro filtrování (pokud chceme jen robotovy kolize)
        filter_inst = None
        if filter_obj_name and filter_obj_name in self.object_ids:
            filter_inst = self.object_ids[filter_obj_name]

        for contact in contacts:
            # Sapien vrací dvojici actorů (bodies)
            actor_a = contact.bodies[0]
            actor_b = contact.bodies[1]

            # Funkce pro získání jména objektu a linku
            def resolve_name(actor):
                # Statické objekty (často ground) nebo null
                if actor is None or (hasattr(actor, "type") and actor.type == "static") or actor.entity.name == 'ground':
                    return "World", "Ground"

                # Pokud je to Articulation (Robot)
                # V Sapien 3 má actor odkaz na svou artikulaci
                articulation = actor.get_articulation()
                if articulation is not None:
                    return articulation.get_name(), actor.name

                # Pokud je to Rigid Body (Kostka, Koule...)
                return actor.name, "base_link"

            obj_a, link_a = resolve_name(actor_a)
            obj_b, link_b = resolve_name(actor_b)

            # --- FILTROVÁNÍ ---
            # Pokud filtrujeme podle jména (např. 'robot'), ignorujeme kolize, kde robot není
            if filter_obj_name:
                if obj_a != filter_obj_name and obj_b != filter_obj_name:
                    continue

            # Ignorování self-collision (robot sám se sebou)
            if obj_a == obj_b:
                continue

            # V Sapienu (CPU) máme obvykle env_id=0, pokud nepoužíváme speciální batching
            entry = {
                "env_id": 0,
                "body_a": obj_a,
                "link_a": link_a,
                "body_b": obj_b,
                "link_b": link_b,
                "formatted": f"{obj_a}::{link_a} <-> {obj_b}::{link_b}"
            }
            if entry not in readable_collisions:
                readable_collisions.append(entry)

        return readable_collisions"""
    def get_contact(self, contacts: list[sapien.physx.PhysxContact], filter_obj_name: str | None = None) -> list[dict]:
        """
        Zpracuje kontakty ze Sapien scény a přidá informace o síle.
        """
        readable_collisions = []

        # Získání časového kroku pro výpočet síly (F = Impuls / dt)
        dt = self.scene.get_timestep()

        for contact in contacts:
            # Sapien vrací dvojici actorů (bodies)
            actor_a = contact.bodies[0]
            actor_b = contact.bodies[1]

            # Funkce pro získání jména objektu a linku
            def resolve_name(actor):
                # Statické objekty (často ground) nebo null
                if actor is None or (hasattr(actor, "type") and actor.type == "static") or actor.entity.name == 'ground':
                    return "World", "Ground"

                # Pokud je to Articulation (Robot)
                articulation = actor.get_articulation()
                if articulation is not None:
                    return articulation.get_name(), actor.name

                # Pokud je to Rigid Body (Kostka, Koule...)
                return actor.name, "base_link"

            obj_a, link_a = resolve_name(actor_a)
            obj_b, link_b = resolve_name(actor_b)

            # --- FILTROVÁNÍ ---
            if filter_obj_name:
                if obj_a != filter_obj_name and obj_b != filter_obj_name:
                    continue

            # Ignorování self-collision
            if obj_a == obj_b:
                continue

            # --- VÝPOČET SÍLY ---
            # Kontakt může mít více bodů (contact points). Musíme sečíst jejich impulsy.
            total_impulse = np.zeros(3)
            for point in contact.points:
                total_impulse += point.impulse

            # Převedení impulsu na sílu: F = impuls / dt
            # total_impulse je vektor (x, y, z), získáme i jeho velikost (skalár)
            force_vec = total_impulse / dt
            force_magnitude = np.linalg.norm(force_vec)

            entry = {
                "env_id": 0,
                "body_a": obj_a,
                "link_a": link_a,
                "body_b": obj_b,
                "link_b": link_b,
                "formatted": f"{obj_a}::{link_a} <-> {obj_b}::{link_b}",
                "force": float(force_magnitude),      # Skalární velikost síly v Newtonech
                "force_vec": force_vec.tolist()       # Vektor síly [fx, fy, fz]
            }

            # Poznámka: Pokud entry obsahuje list (force_vec), porovnání "if entry not in..."
            # může selhat nebo být pomalé. Zde to necháváme pro zachování logiky,
            # ale pro 'force_vec' používáme .tolist(), aby to bylo porovnatelné.
            if entry not in readable_collisions:
                readable_collisions.append(entry)

        return readable_collisions
    def _apply_action(self, instance: sapien_core.physx.PhysxArticulation, pos_action=None, vel_action=None):
        qf = instance.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        if pos_action is not None:
            for joint in instance.get_active_joints():
                name = joint.get_name()
                if name in pos_action:
                    target = pos_action[name]
                    joint.set_drive_target(float(target))

        if vel_action is not None:
            for joint in instance.get_active_joints():
                name = joint.get_name()
                if name in vel_action:
                    target = vel_action[name]
                    joint.set_drive_property(0.0, 5.0, mode="force")  # velocity control
                    joint.set_drive_velocity_target(float(target))
        instance.set_qf(qf)

    def set_dof_targets(self, obj_name, target: list[Action]):
        instance = self.object_ids[obj_name]
        if isinstance(instance, sapien_core.physx.PhysxArticulation):
            action = target[0]
            pos_target = action.get("dof_pos_target", None)
            if pos_target is None:
                action = action[obj_name]
                pos_target = action.get("dof_pos_target", None)

            vel_target = action.get("dof_vel_target", None)
            pos_target_arr = (
                np.array([pos_target[name] for name in self.object_joint_order[obj_name]]) if pos_target else None
            )
            vel_target_arr = (
                np.array([vel_target[name] for name in self.object_joint_order[obj_name]]) if vel_target else None
            )
            self._previous_dof_pos_target[obj_name] = pos_target_arr
            self._previous_dof_vel_target[obj_name] = vel_target_arr
            self._apply_action(instance, pos_target, vel_target)

    def _simulate(self):
        for i in range(self.scenario.decimation):
            self.scene.step()
            self.scene.update_render()
            if not self.headless:
                self.viewer.render()
        for camera_name, camera_id in self.camera_ids.items():
            camera_id.take_picture()

    def launch(self) -> None:
        self._build_sapien()

    def close(self):
        if not self.headless:
            self.viewer.close()
        self.scene = None

    def _get_link_states(self, obj_name: str) -> tuple[list, torch.Tensor]:
        link_name_list = []
        link_state_list = []

        if len(self.link_ids[obj_name]) == 0:
            return [], torch.zeros((0, 13), dtype=torch.float32)

        for link in self.link_ids[obj_name]:
            pose = link.get_pose()
            pos = torch.tensor(pose.p)
            rot = torch.tensor(pose.q)
            vel = torch.tensor(link.linear_velocity)
            ang_vel = torch.tensor(link.angular_velocity)
            link_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
            link_name_list.append(link.get_name())
            link_state_list.append(link_state)
        link_state_tensor = torch.cat(link_state_list, dim=0)
        return link_name_list, link_state_tensor

    def _get_states(self, env_ids=None) -> list[EnvState]:
        object_states = {}

        # načteme všechny kontakty ve scéně
        raw_scene_contacts = self.scene.get_contacts()
        for obj in self.objects:
            obj_inst = self.object_ids[obj.name]
            pose = obj_inst.get_pose()
            link_names, link_state = self._get_link_states(obj.name)
            if isinstance(obj, ArticulationObjCfg):
                assert isinstance(obj_inst, sapien_core.physx.PhysxArticulation)
                joints_names_arr = np.array(self.get_joint_names(obj.name))
                pos = torch.tensor(pose.p)
                rot = torch.tensor(pose.q)
                vel = torch.tensor(obj_inst.get_root_linear_velocity())
                ang_vel = torch.tensor(obj_inst.get_root_angular_velocity())
                root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
                state = ObjectState(
                    joint_names = joints_names_arr,
                    root_state=root_state,
                    body_names=link_names,
                    body_state=link_state.unsqueeze(0),
                    joint_pos=torch.tensor(obj_inst.get_qpos()).unsqueeze(0),
                    joint_vel=torch.tensor(obj_inst.get_qvel()).unsqueeze(0),
                )
            elif isinstance(obj, RigidObjCfg) and obj.fix_base_link:
                pos = torch.tensor(pose.p)
                rot = torch.tensor(pose.q)
                root_state = torch.cat([pos, rot], dim=-1).unsqueeze(0)
                state = ObjectState(root_state=root_state)
            else:
                assert isinstance(obj_inst, sapien_core.Entity)
                pos = torch.tensor(pose.p)
                rot = torch.tensor(pose.q)
                vel = torch.tensor(obj_inst.get_components()[1].get_linear_velocity())
                ang_vel = torch.tensor(obj_inst.get_components()[1].get_angular_velocity())
                root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
                state = ObjectState(root_state=root_state)
            object_states[obj.name] = state

        robot_states = {}
        for robot in [self.robot]:
            robot_inst = self.object_ids[robot.name]
            assert isinstance(robot_inst, sapien_core.physx.PhysxArticulation)

            readable_contacts = self.get_contact(raw_scene_contacts, filter_obj_name=robot.name)
            # if readable_contacts:
            #     print("detekována kolize:")
            #     for contact in readable_contacts:
            #         print(f" - {contact['formatted']}")
            pose = robot_inst.get_pose()
            pos = torch.tensor(pose.p)
            rot = torch.tensor(pose.q)
            vel = torch.tensor(robot_inst.root_linear_velocity)
            ang_vel = torch.tensor(robot_inst.root_angular_velocity)
            root_state = torch.cat([pos, rot, vel, ang_vel], dim=-1).unsqueeze(0)
            link_names, link_state = self._get_link_states(robot.name)
            pos_target = (
                torch.tensor(self._previous_dof_pos_target[robot.name]).unsqueeze(0)
                if self._previous_dof_pos_target[robot.name] is not None
                else None
            )
            vel_target = (
                torch.tensor(self._previous_dof_vel_target[robot.name]).unsqueeze(0)
                if self._previous_dof_vel_target[robot.name] is not None
                else None
            )
            effort_target = (
                torch.tensor(self._previous_dof_torque_target[robot.name]).unsqueeze(0)
                if self._previous_dof_torque_target[robot.name] is not None
                else None
            )

            joints_names_arr = np.array(self.get_joint_names(robot.name))

            state = RobotState(
                root_state=root_state,
                body_names=link_names,
                body_state=link_state.unsqueeze(0),
                joint_pos=torch.tensor(robot_inst.get_qpos()).unsqueeze(0),
                joint_vel=torch.tensor(robot_inst.get_qvel()).unsqueeze(0),
                joint_names=joints_names_arr,
                joint_pos_target=pos_target,
                joint_vel_target=vel_target,
                joint_effort_target=effort_target,
                contact=readable_contacts,
            )
            robot_states[robot.name] = state

        camera_states = {}
        for camera in self.cameras:
            cam_inst = self.camera_ids[camera.name]
            rgb = cam_inst.get_picture("Color")[..., :3]
            rgb = (rgb * 255).clip(0, 255).astype("uint8")
            rgb = torch.from_numpy(rgb.copy())
            depth = -cam_inst.get_picture("Position")[..., 2]
            depth = torch.from_numpy(depth.copy())
            state = CameraState(rgb=rgb.unsqueeze(0), depth=depth.unsqueeze(0))
            camera_states[camera.name] = state
        sensors = {}
        for sensor in self.scenario.sensors:
            if isinstance(sensor, GyroSensorCfg):
                gyro_data = sensor.get_data(robot_states,envs_ids=env_ids)
                gyro_tensor = torch.tensor(gyro_data, dtype=torch.float32).unsqueeze(1)
                sensors[sensor.name] = gyro_tensor
            elif isinstance(sensor, CommandCfg):
                command_data = sensor.get_command()
                #command_tensor = torch.tensor(command_data, dtype=torch.float32).unsqueeze(1)
                sensors[sensor.name] = command_data
            else:
                log.warning(f"Unknown sensor type: {sensor.cfg_type}, skipping...")

        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, sensors=sensors)

    def refresh_render(self):
        robot_inst = self.object_ids[self.robot.name]
        for name, obj in self.object_ids.items():
            if isinstance(self.object_dict[name], ArticulationObjCfg):

                # reset joint forces
                obj.set_qf(np.zeros(obj.dof))

                # reset joint velocities
                obj.set_qvel(np.zeros(obj.dof))

                # reset root velocity (linear and angular)
                obj.set_root_linear_velocity(np.zeros(3))
                obj.set_root_angular_velocity(np.zeros(3))


        self.scene.update_render()
        if not self.headless:
            self.viewer.render()
        for camera_name, camera_id in self.camera_ids.items():
            camera_id.take_picture()

    def _set_states(self, states, env_ids=None):
        states_flat = [state["objects"] | state["robots"] for state in states]
        for name, val in states_flat[0].items():
            if name not in self.object_ids:
                continue
            # assert name in self.object_ids
            # Reset joint state
            obj_id = self.object_ids[name]
            if val["pos"].device != torch.device("cpu"):
                val["pos"] = val["pos"].cpu()
            if val["rot"].device != torch.device("cpu"):
                val["rot"] = val["rot"].cpu()

            if isinstance(self.object_dict[name], ArticulationObjCfg):
                joint_names = self.object_joint_order[name]
                qpos_list = []
                for i, joint_name in enumerate(joint_names):
                    qpos_list.append(val["dof_pos"][joint_name])
                obj_id.set_qpos(np.array(qpos_list))
            # Reset base position and orientation
            obj_id.set_pose(sapien_core.Pose(p=val["pos"], q=val["rot"]))

    @property
    def actions_cache(self) -> list[Action]:
        return self._actions_cache

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def get_joint_names(self, obj_name: str, sort: bool = False) -> list[str]:
        if isinstance(self.object_dict[obj_name], ArticulationObjCfg):
            joint_names = deepcopy(self.object_joint_order[obj_name])
            if sort:
                joint_names.sort()
            return joint_names
        else:
            return []

    def get_body_names(self, obj_name, sort=True):
        body_names = deepcopy([link.name for link in self.link_ids[obj_name]])
        if sort:
            return sorted(body_names)
        else:
            return deepcopy(body_names)


Sapien3Env: type[EnvWrapper[Sapien3Handler]] = GymEnvWrapper(Sapien3Handler)

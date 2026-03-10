
from __future__ import annotations

from multiprocessing.util import debug
from typing import Literal

import torch
from loguru import logger as log
import numpy as np


from metasim.wrapper.gym_vec_env import MetaSimVecEnv
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R


#from roboverse_learn.rl.rsl_rl.rsl_rl import env

class StableBaseline3VecEnv(VecEnv):
    """Vectorized environment for Stable Baselines 3 that supports parallel RL training."""

    def __init__(self, env: MetaSimVecEnv):
        """Initialize the environment."""
        joint_limits = env.scenario.robots[0].joint_limits
        self.action_space = spaces.Box(
            low=np.array([lim[0] for lim in joint_limits.values()]),
            high=np.array([lim[1] for lim in joint_limits.values()]),
            shape=(len(joint_limits),),
            dtype=np.float32,
        )
        robot_name = env.scenario.robots[0].name

        num_joints = len(joint_limits)

        states = env.env.handler.get_states()
        self.main_robot_link_names = [
                                "left_endeffector",
                                "endeffector",
                                "torso_link",
                                "pelvis",
                                'left_shoulder_pitch_link',
                                'left_shoulder_roll_link',
                                'left_shoulder_yaw_link',
                                'left_elbow_link',
                                'left_wrist_roll_link',
                                'left_wrist_pitch_link',
                                'left_wrist_yaw_link',
                                 'right_shoulder_pitch_link',
                                 'right_shoulder_roll_link',
                                 'right_shoulder_yaw_link',
                                 'right_elbow_link',
                                 'right_wrist_roll_link',
                                 'right_wrist_pitch_link',
                                 'right_wrist_yaw_link',
                                 'left_hand_thumb_0_link',
                                 'left_hand_middle_0_link',
                                 'left_hand_index_0_link',
                                 'right_hand_thumb_0_link',
                                 'right_hand_middle_0_link',
                                 'right_hand_index_0_link',
                                 'left_hand_thumb_1_link',
                                 'left_hand_thumb_2_link',
                                 'left_hand_middle_1_link',
                                 'right_hand_thumb_1_link',
                                 'right_hand_thumb_2_link',
                                 'right_hand_middle_1_link'
                                 ]
        self.indexes = [states.robots[robot_name].body_names.index(link) for link in self.main_robot_link_names]

        num_robot_bodies = len(self.main_robot_link_names)  # pozice (3) + orientace (4) pro každý link
        obs_shape = num_joints + (num_robot_bodies * 7) + 7 + 7

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape,),
            dtype=np.float32,
        )
        self.env = env
        self.render_mode = None
        self.timesteps = torch.zeros(env.num_envs, dtype=torch.float32, device=("cuda" if env.scenario.sim == 'isaaclab' or env.scenario.sim == 'genesis' else "cpu"))
        self.action = None #TODO debug holder
        self.finger_current_positions = {} #TODO debug holder
        super().__init__(env.num_envs, self.observation_space, self.action_space)

    def add_extra_to_obs(self, obs: np.ndarray) -> np.ndarray:
        """extend obs with extra data."""
        states = self.env.env.handler.get_states()
        robot_name = self.env.scenario.robots[0].name
        chair = states.objects["chair"]

        # 1. vybrané body_states robota
        # Vybere pozici a orientaci (prvních 7 hodnot) pro všechny linky.
        # Shape: (num_envs, num_bodies, 7) -> po reshape: (num_envs, num_bodies * 7)

        robot_body_states = states.robots[robot_name].body_state[:, self.indexes, :7].reshape(self.num_envs, -1).cpu().numpy()
        # 2. Získání indexů a stavů pro cílové body na židli
        target_left_idx = chair.body_names.index("target_hand_left")
        target_right_idx = chair.body_names.index("target_hand_right")

        target_left_pos_ori = chair.body_state[:, target_left_idx, :7].cpu().numpy()
        target_right_pos_ori = chair.body_state[:, target_right_idx, :7].cpu().numpy()

        # 3. Sloučení extra dat
        other_pos = np.concatenate([robot_body_states, target_left_pos_ori, target_right_pos_ori], axis=1)

        # Zajištění správného rozměru původního obs (klouby)
        obs = obs.reshape(self.num_envs, -1)

        # Vrácení spojeného finálního pole obs + extra data
        return np.concatenate([obs, other_pos], axis=1).astype(np.float32)

    def _combine_obs(self, obs: np.ndarray) -> np.ndarray:
        """Spojí joint states a gyro data pro všechna envs."""
        states = self.env.env.handler.get_states()
        gyrodata = states.sensors["gyro0"].cpu().numpy()  # shape (num_envs, 3)
        gyrodata = gyrodata.reshape(self.num_envs, 3)
        obs = obs.reshape(self.num_envs, -1)       # (num_envs, dof_count)
        return np.concatenate([obs, gyrodata], axis=1).astype(np.float32)


    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset()
        obs = obs.cpu().numpy()
        #obs = self._combine_obs(obs)
        obs = self.add_extra_to_obs(obs)
        self.timesteps.zero_()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Asynchronously step the environment."""

        # --- RYCHLÁ CESTA PRO GENESIS ---
        if self.env.scenario.sim == 'genesis':
            # Akce si uložíme rovnou jako numpy array, žádné slovníky!
            self.raw_actions = actions
            self.action_dicts = None

        # --- POMALÁ CESTA PRO OSTATNÍ SIMULÁTORY ---
        else:
            self.raw_actions = None
            self.action_dicts = [
                {
                    self.env.scenario.robots[0].name: {
                        "dof_pos_target": dict(zip(self.env.scenario.robots[0].joint_limits.keys(), action))
                    }
                }
                for action in actions
            ]
    def step_wait(self):
        """Wait for the step to complete."""
        #------------------------------------
        #--------------DEBUG-----------------
        #------------------------------------
        # debug = 2
        # if debug == 0:
        #     actions = self.debug0()
        #     obs, rewards, unsuccess, timeout, _ = self.env.step(actions)
        # elif debug == 1:
        #     if self.timesteps[0] % 20 == 0:
        #         self.action = self.ik_solver()
        #     obs, rewards, unsuccess, timeout, _ = self.env.step([{"g1_slider": {"dof_pos_target": self.action}}])
        # elif debug == 2:
        #     obs, rewards, unsuccess, timeout, _ = self.env.step(self.debug2())
        #end debug

        if self.env.scenario.sim == 'genesis' and self.raw_actions is not None:
            actions_to_pass = self.raw_actions
        else:
            actions_to_pass = self.action_dicts

        # Provedení kroku s vybraným formátem akcí
        obs, rewards, unsuccess, timeout, _ = self.env.step(actions_to_pass)
        obs = obs.cpu().numpy()
        obs = self.add_extra_to_obs(obs)
        #obs = self._combine_obs(obs)

        # --- Done flag ---
        dones = timeout.to(unsuccess.device) | unsuccess

        # --- Update time counters ---
        self.timesteps += (~unsuccess).float()

        # --- Připrav info dicty ---
        infos = [{} for _ in range(self.num_envs)]

        # --- Masky ---
        unsuccess_mask = unsuccess.cpu().numpy().astype(bool)
        timeout_mask = timeout.cpu().numpy().astype(bool)

        # --- Reset neúspěšných envů ---
        if unsuccess_mask.any():
            self.timesteps[unsuccess_mask] = 0.0
            unsuccess_ids = np.nonzero(unsuccess_mask)[0].tolist()
            self.env.reset(env_ids=unsuccess_ids)
            for i in unsuccess_ids:
                infos[i]["is_success"] = False
                infos[i]["TimeLimit.truncated"] = False

        # --- Reset úspěšných envů (timeout = úspěch) ---
        if timeout_mask.any():
            self.timesteps[timeout_mask] = 0.0
            timeout_ids = np.nonzero(timeout_mask)[0].tolist()
            self.env.reset(env_ids=timeout_ids)
            for i in timeout_ids:
                infos[i]["is_success"] = True
                infos[i]["TimeLimit.truncated"] = True

        return obs, rewards.cpu().numpy(), dones.cpu().numpy(), infos
    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    ############################################################
    ## Abstract methods
    ############################################################
    def get_images(self):
        """Get images from the environment."""
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        """Get an attribute of the environment."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.env.handler, attr_name)] * len(indices)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """Set an attribute of the environment."""
        raise NotImplementedError

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call a method of the environment."""
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if the environment is wrapped by a given wrapper class."""
        raise NotImplementedError


    def ik_solver(self) -> dict:
        import numpy as np
        from ikpy.chain import Chain
        from scipy.spatial.transform import Rotation as R

        # 1. SETUP PROSTŘEDÍ
        env = self.env
        robot_cfg = env.scenario.robots[0]
        states = env.env.handler.get_states()
        robot_state = states.robots[robot_cfg.name]

        # Inicializace výstupního slovníku (všechny klouby na 0)
        full_joint_dict = {name: 0.0 for name in list(robot_cfg.joint_limits.keys())}

        # 2. VÝPOČET BÁZE (TORSO)
        # Získáme globální pozici a orientaci torsa, která je společná pro obě ruce
        torso_idx = robot_state.body_names.index("torso_link")
        base_pos = robot_state.body_state[0, torso_idx, :3].cpu().numpy()
        q_raw = robot_state.body_state[0, torso_idx, 3:7].cpu().numpy()

        # Fix Quaternionu (WXYZ -> XYZW) pro MuJoCo
        if abs(q_raw[0]) > 0.9:
            base_quat = [q_raw[1], q_raw[2], q_raw[3], q_raw[0]]
        else:
            base_quat = q_raw

        rot_base = R.from_quat(base_quat)

        # Vytvoření inverzní matice báze (pro převod Svět -> Lokální řetězec)
        base_matrix = np.eye(4)
        base_matrix[:3, :3] = rot_base.as_matrix()
        base_matrix[:3, 3] = base_pos
        base_inv = np.linalg.inv(base_matrix)

        # 3. DEFINICE KONFIGURACE PRO OBĚ RUCE
        # Zde definujeme cesty k URDF a názvy targetů pro smyčku
        configs = [
            {
                "side": "left",
                "urdf": "/home/roboversepc/code/RoboVerse/roboverse_data/robots/g1/urdf/g1_rotslider_for_IK_left.urdf",
                "target_name": "target_hand_left",
                # Rotace pro levou ruku (dle tvého funkčního kódu)
                #"rot_fix": R.from_euler('z', -90, degrees=True),
                #"rot_approach": R.from_euler('y', -90, degrees=True)
            },
            {
                "side": "right",
                "urdf": "/home/roboversepc/code/RoboVerse/roboverse_data/robots/g1/urdf/g1_rotslider_for_IK_right.urdf",
                "target_name": "target_hand_right",
                # Rotace pro pravou ruku (Zrcadlově? Často bývá approach 'y' +90 nebo stejně, nutno vyzkoušet)
                # Zde nechávám stejné nastavení, pokud je URDF symetrické, možná bude třeba upravit approach.
                "rot_fix": R.from_euler('z', -90, degrees=True), # Pravděpodobně opačně pro druhou ruku
                #"rot_approach": R.from_euler('y', -90, degrees=True)
            }
        ]

        # 4. HLAVNÍ SMYČKA PRO LEVOU A PRAVOU RUKU
        chair = states.objects["chair"]

        for cfg in configs:
            # A) Načtení řetězce
            chain = Chain.from_urdf_file(cfg["urdf"], base_elements=["torso_link"])

            # B) Získání Targetu
            try:
                target_idx = chair.body_names.index(cfg["target_name"])
            except ValueError:
                print(f"⚠️ Target {cfg['target_name']} nenalezen, přeskakuji {cfg['side']} ruku.")
                continue

            target_pos = chair.body_state[0, target_idx, :3].cpu().numpy()
            q_tgt_raw = chair.body_state[0, target_idx, 3:7].cpu().numpy()
            # Fix target quaternionu
            target_quat = [q_tgt_raw[1], q_tgt_raw[2], q_tgt_raw[3], q_tgt_raw[0]]

            # C) Výpočet finální orientace cíle
            r_target = R.from_quat(target_quat)
            #final_rot = r_target * cfg["rot_fix"] #* cfg["rot_approach"]

            # D) Transformace do lokálního prostoru (Chain Frame)
            target_matrix_world = np.eye(4)
            target_matrix_world[:3, :3] = r_target.as_matrix()
            target_matrix_world[:3, 3] = target_pos

            target_in_chain = base_inv @ target_matrix_world

            # E) Výpočet IK
            ik_joints = chain.inverse_kinematics_frame(
                target_in_chain,
                orientation_mode="all",
                optimizer="least_squares"
            )

            # F) Mapování na názvy kloubů (Links -> Joints)
            for i, link in enumerate(chain.links):
                link_name = link.name
                val = ik_joints[i]

                # Logika pro nalezení správného klíče
                if link_name in full_joint_dict:
                    full_joint_dict[link_name] = val
                elif link_name.replace("_link", "_joint") in full_joint_dict:
                    full_joint_dict[link_name.replace("_link", "_joint")] = val

        return full_joint_dict

    def debug0(self):
        # [{"g1_slider": {"dof_pos_target": self.action}}]
        actions = [{"g1_slider": {"dof_pos_target":
                                  {
                        "baseslide_joint": 0.0, #y
                        "baseslide_joint2": 0.0, #x
                        "baserot_joint": 0.0,
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.34,
                        "left_shoulder_pitch_joint": 0.0,
                        "left_shoulder_roll_joint": 0.0,
                        "left_shoulder_yaw_joint": 0.0,
                        "left_elbow_joint": 1.0,
                        "left_wrist_roll_joint": 0.0,
                        "left_wrist_pitch_joint": 0.0,
                        "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 1.0,
                        "right_wrist_roll_joint": 0.0,
                        "right_wrist_pitch_joint": 0.0,
                        "right_wrist_yaw_joint": 0.0,
                        # Left hand fingers
                        "left_hand_thumb_0_joint": 0.0,
                        "left_hand_thumb_1_joint": 0.0,
                        "left_hand_thumb_2_joint": 0.0,
                        "left_hand_middle_0_joint": 0.0,
                        "left_hand_middle_1_joint": 0.0,
                        "left_hand_index_0_joint": 0.0,
                        "left_hand_index_1_joint": 0.0,
                        # Right hand fingers
                        "right_hand_thumb_0_joint": 0.0,
                        "right_hand_thumb_1_joint": 0.0,
                        "right_hand_thumb_2_joint": 0.0,
                        "right_hand_middle_0_joint": 0.0,
                        "right_hand_middle_1_joint": 0.0,
                        "right_hand_index_0_joint": 0.0,
                        "right_hand_index_1_joint": 0.0
                                  }
                                    }}]
        return actions
    def debug2(self):
        """
        Drží tělo ve fixní pozici a postupně zavírá prsty.
        Zastaví pohyb prstu pouze pokud síla kontaktu na ŠPIČCE přesáhne práh.

        Implementace využívá optimalizované tensorové operace kompatibilní s Genesis.
        """
        FORCE_THRESHOLD = 1.0
        STEP_SIZE = 0.02
        robot_name = "g1_slider"

        # 1. DEFINICE CÍLOVÝCH HODNOT (Limity kam až se mají prsty zavřít)
        finger_limits = {
            # --- LEVÁ RUKA ---
            "left_hand_thumb_0_joint": 1.0, "left_hand_thumb_1_joint": 0.7, "left_hand_thumb_2_joint": 1.0,
            "left_hand_middle_0_joint": -1.5, "left_hand_middle_1_joint": -1.7,
            "left_hand_index_0_joint": -1.5, "left_hand_index_1_joint": -1.7,
            # --- PRAVÁ RUKA ---
            "right_hand_thumb_0_joint": -1.0, "right_hand_thumb_1_joint": -0.7, "right_hand_thumb_2_joint": -1.0,
            "right_hand_middle_0_joint": 1.5, "right_hand_middle_1_joint": 1.7,
            "right_hand_index_0_joint": 1.5, "right_hand_index_1_joint": 1.7
        }

        # Base pose zbytku těla
        base_pose = {
            'baseslide_joint': 1.0338146e-05, 'baseslide_joint2': 0.042001966, 'baserot_joint': -7.5084286e-06,
            'waist_yaw_joint': -4.4286382e-05, 'waist_roll_joint': 8.922054e-05, 'waist_pitch_joint': 0.008053156,
            'left_shoulder_pitch_joint': -0.7364295, 'right_shoulder_pitch_joint': -0.72817403,
            'left_shoulder_roll_joint': 0.5458939, 'right_shoulder_roll_joint': -0.5318888,
            'left_shoulder_yaw_joint': -0.57718706, 'right_shoulder_yaw_joint': 0.5727658,
            'left_elbow_joint': 0.2027091, 'right_elbow_joint': 0.18453476,
            'left_wrist_roll_joint': 0.8287475, 'right_wrist_roll_joint': -0.80135757,
            'left_wrist_pitch_joint': 0.56613743, 'right_wrist_pitch_joint': 0.5907418,
            'left_wrist_yaw_joint': -0.24424289, 'right_wrist_yaw_joint': 0.24177466
        }

        # 2. DEFINICE SKUPIN PRSTŮ (Které klouby patří ke kterému "senzoru" na špičce)
        # Mapování: ID skupiny -> (Seznam kloubů, Klíčové slovo pro link špičky)
        finger_groups = {
            0: (["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint"], "left_hand_thumb_2_link"),
            1: (["left_hand_index_0_joint", "left_hand_index_1_joint"], "left_hand_index_1_link"),
            2: (["left_hand_middle_0_joint", "left_hand_middle_1_joint"], "left_hand_middle_1_link"),
            3: (["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"], "right_hand_thumb_2_link"),
            4: (["right_hand_index_0_joint", "right_hand_index_1_joint"], "right_hand_index_1_link"),
            5: (["right_hand_middle_0_joint", "right_hand_middle_1_joint"], "right_hand_middle_1_link"),
        }

        handler = self.env.env.handler
        device = handler.device
        states = handler.get_states()
        num_envs = self.num_envs

        # --- INICIALIZACE STAVU (běží jen poprvé) ---
        if not hasattr(self, "_debug2_cache_init"):
            # 1. Inicializace aktuálních pozic prstů (Tenzor [num_envs, num_finger_joints])
            self._finger_joint_names = list(finger_limits.keys())
            self._finger_joint_limits = torch.tensor([finger_limits[n] for n in self._finger_joint_names], device=device)
            self._current_finger_pos = torch.zeros((num_envs, len(self._finger_joint_names)), device=device)

            # 2. Mapování ID linků na ID skupiny prstů (stejné jako v GraspForceReward)
            global_map = states.extras.get("global_link_map", {})
            num_bodies = states.extras.get("num_bodies_per_env", 1000)

            # Vytvoříme mapu: Global Link ID -> Finger Group ID (0-5) nebo -1
            self._idx_to_group = torch.full((num_bodies,), -1, dtype=torch.long, device=device)
            chair_ids = []

            for idx, (o_name, l_name) in global_map.items():
                if o_name == robot_name:
                    for grp_id, (_, tip_link_name) in finger_groups.items():
                        # Hledáme přesnou shodu špičky prstu
                        if tip_link_name in l_name:
                            self._idx_to_group[idx] = grp_id
                elif o_name == "chair":
                    chair_ids.append(idx)

            self._chair_ids = torch.tensor(chair_ids, device=device)
            self._num_bodies = num_bodies
            self._debug2_cache_init = True

            log.info("Debug2: Initialization complete.")

        # --- DETEKCE KONTAKTŮ (Tenzorově) ---
        # Získáme kontakty přímo z robota
        contact_data = states.robots[robot_name].contact

        # Maska blokovaných skupin prstů [num_envs, 6] (6 skupin)
        blocked_groups = torch.zeros((num_envs, len(finger_groups)), dtype=torch.bool, device=device)

        if contact_data is not None:
            link_a = contact_data['link_a']
            link_b = contact_data['link_b']
            valid_mask = contact_data['valid_mask']

            # Získání sil
            forces = contact_data.get('force_b', contact_data.get('force', None))
            if forces is None:
                forces = torch.zeros((*link_a.shape, 3), device=device)
            force_mags = torch.norm(forces, dim=-1)

            # Modulo pro získání base indexů (pro multi-env)
            base_a = link_a % self._num_bodies
            base_b = link_b % self._num_bodies

            # Zjištění, kdo je židle a kdo je prst
            a_is_chair = torch.isin(base_a, self._chair_ids)
            b_is_chair = torch.isin(base_b, self._chair_ids)

            # Mapování na skupiny (pokud není prst, vrátí -1)
            group_a = self._idx_to_group[base_a]
            group_b = self._idx_to_group[base_b]

            # Která skupina prstů se dotýká židle?
            # Pokud b je židle -> kontakt je na a. Pokud a je židle -> kontakt je na b.
            contact_group = torch.where(b_is_chair, group_a, torch.where(a_is_chair, group_b, torch.tensor(-1, device=device)))

            # Validní kontakt: (Je to kontakt prst-židle) AND (Validní v simulaci) AND (Síla > Threshold)
            valid_interaction = (contact_group >= 0) & valid_mask & (force_mags > FORCE_THRESHOLD)

            # Vyplnění masky blokovaných skupin
            # Pro každou skupinu zjistíme, zda má v daném envu validní silný kontakt
            for grp_id in range(len(finger_groups)):
                # Má tento env nějaký kontakt pro tuto skupinu?
                has_contact = (valid_interaction & (contact_group == grp_id)).any(dim=1)
                blocked_groups[:, grp_id] = has_contact

        # --- AKTUALIZACE POZIC ---
        # Iterujeme přes definované klouby a aktualizujeme pozice
        for i, joint_name in enumerate(self._finger_joint_names):
            # Zjistíme, do které skupiny tento kloub patří
            target_grp = -1
            for grp_id, (joints, _) in finger_groups.items():
                if joint_name in joints:
                    target_grp = grp_id
                    break

            # Získáme limit pro tento kloub
            limit = self._finger_joint_limits[i]

            # Maska: Kde MŮŽEME hýbat? (Kde NENÍ skupina blokovaná)
            can_move = ~blocked_groups[:, target_grp]

            # Logika pohybu směrem k limitu
            current_val = self._current_finger_pos[:, i]

            # Vypočítáme krok (plus nebo mínus podle směru k limitu)
            direction = torch.sign(limit - current_val)
            step = direction * STEP_SIZE

            # Nová hodnota (před oříznutím)
            next_val = current_val + step

            # Ošetření přehmitu (clamp mezi current a limit nefunguje jednoduše, uděláme to logicky)
            # Pokud jsme blízko limitu, nastavíme limit
            close_to_limit = torch.abs(current_val - limit) < STEP_SIZE
            next_val = torch.where(close_to_limit, limit, next_val)

            # Aplikujeme změnu jen tam, kde není blokace
            self._current_finger_pos[:, i] = torch.where(can_move, next_val, current_val)

        # --- SESTAVENÍ AKCE ---
        # Protože vracíme list[dict] pro SB3 wrapper, musíme převést tensor zpět (nebo použít tensor přímo, pokud wrapper podporuje)
        # Zde sestavíme full dictionary, kde base_pose je statická a prsty dynamické.

        # Poznámka: Aby to bylo rychlé, ideálně bychom měli posílat Tensor přímo do handleru,
        # ale SB3VecEnv ve vaší implementaci očekává list dictů. Uděláme to tedy hybridně.

        actions = []
        cpu_finger_pos = self._current_finger_pos.cpu().numpy()

        for env_id in range(num_envs):
            # Kopie base pose
            dof_targets = base_pose.copy()

            # Update prstů pro tento env
            for i, name in enumerate(self._finger_joint_names):
                dof_targets[name] = float(cpu_finger_pos[env_id, i])

            actions.append({robot_name: {"dof_pos_target": dof_targets}})

        return actions

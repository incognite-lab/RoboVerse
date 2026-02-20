"""Exit Door task for humanoid robots."""

from __future__ import annotations
from xml.sax import handler

import torch

from metasim.cfg.checkers import _ChairManChecker
from metasim.cfg.objects import RigidObjCfg, ArticulationObjCfg
from metasim.constants import PhysicStateType
from metasim.types import EnvState
from metasim.utils import configclass, humanoid_reward_util, humanoid_robot_util

from .base_cfg import HumanoidBaseReward, HumanoidTaskCfg, StableReward

from metasim.utils.humanoid_robot_util import neck_height_tensor



HEIGHT_THRESHOLD = 0.4
class TerminationCfg(HumanoidBaseReward):
    """Termination condition based on humanoid's neck height."""
    def __init__(self):
        super().__init__()
    def __call__(self, states: EnvState, robot_name) -> list[bool]:
        neck_heights = neck_height_tensor(states, robot_name)[:]
        terminated = torch.tensor([0.0] * len(neck_heights))
        for i in range(len(neck_heights)):
            if neck_heights[i] < HEIGHT_THRESHOLD:
                terminated[i] = 1.0
        return terminated
class DeltaActionRateCfg(HumanoidBaseReward):
    """Reward function for minimizing change in action rate."""
    def __init__(self, robot_name="g1_with_hands"):
        """Initialize the delta action rate reward."""
        super().__init__(robot_name)
        self.prev_actions = None

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute the delta action rate reward."""
        actions = states.robots[robot_name].joint_pos_target
        if self.prev_actions is None:
            self.prev_actions = actions
            return torch.zeros(actions.shape[0])

        delta_actions = torch.abs(actions - self.prev_actions)
        self.prev_actions = actions
        action_rate_penalty = torch.sum(torch.square(delta_actions), dim=1)
        #print(f"Delta action rate penalty: {action_rate_penalty.mean().item():.6f}")
        return action_rate_penalty
class DoFVelocityAccelerationCfg(HumanoidBaseReward):
    """
    Penalize high joint velocities and accelerations (excluding fingers).
    According to DoorMan paper Table 2:
    - DoF velocity penalty weight: -1.0 x 10^-3
    - DoF acceleration penalty weight: -1.0 x 10^-5
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.prev_joint_vel = None
        self.fingers = None


    def __call__(self, states: list[EnvState], robot_name: str = None, weights=[-1.0e-3, -1.0e-5]) -> torch.FloatTensor:
        """Compute penalty for joint velocities and accelerations (excluding fingers)."""
        robot = states.robots[robot_name]
        joint_vel = robot.joint_vel  # [num_envs, num_dof]
        device = joint_vel.device
        if self.fingers is None:
            self.fingers = []
            for idx, joint in enumerate(robot.joint_names):
                if "hand" in joint:
                    self.fingers.append(idx)
        if self.fingers:
            num_dof = joint_vel.shape[1]
            all_indices = torch.arange(num_dof, device=device)
            # Maska: True pro klouby, které NEJSOU prsty (not in finger_indices)
            non_finger_mask = ~torch.isin(all_indices, torch.tensor(self.fingers, device=device))
            target_vel = joint_vel[:, non_finger_mask]
        else:
            target_vel = joint_vel

        # 2. Velocity Penalty: ||q_dot_upper, non-finger||^2
        vel_penalty = torch.sum(torch.square(target_vel), dim=-1)

        # ||q_ddot||^2 ~ ||(vel_t - vel_t-1)||^2
        if self.prev_joint_vel is None:
            acc_penalty = torch.zeros_like(vel_penalty)
            self.prev_joint_vel = joint_vel.detach().clone()
        else:
            # Získání předchozích rychlostí pro relevantní klouby
            if self.fingers:
                prev_target_vel = self.prev_joint_vel[:, non_finger_mask]
            else:
                prev_target_vel = self.prev_joint_vel
            delta_vel = target_vel - prev_target_vel
            acc_penalty = torch.sum(torch.square(delta_vel), dim=-1)

            self.prev_joint_vel = joint_vel.detach().clone()
        total_penalty = (weights[0] * vel_penalty) + (weights[1] * acc_penalty)
        #print(f"Velocity penalty: {vel_penalty.mean().item():.6f}, Acceleration penalty: {acc_penalty.mean().item():.6f}, Total penalty: {total_penalty.mean().item():.6f}")
        return total_penalty
class DofPositionLimitsCfg(HumanoidBaseReward):
    """Penalty for exceeding joint position limits."""
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.joint_limits: dict[str, tuple[float, float]] = {

        "waist_yaw_joint": (-2.618, 2.618),
        "waist_roll_joint": (-0.52, 0.52),
        "waist_pitch_joint": (-0.52, 0.52),
        "left_shoulder_pitch_joint": (-3.0892, 2.6704),
        "left_shoulder_roll_joint": (-1.5882, 2.2515),
        "left_shoulder_yaw_joint": (-2.618, 2.618),
        "left_elbow_joint": (-1.0472, 2.0944),
        "left_wrist_roll_joint": (-1.972222054, 1.972222054),
        "left_wrist_pitch_joint": (-1.614429558, 1.614429558),
        "left_wrist_yaw_joint": (-1.614429558, 1.614429558),
        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
        "right_shoulder_roll_joint": (-2.2515, 1.5882),
        "right_shoulder_yaw_joint": (-2.618, 2.618),
        "right_elbow_joint": (-1.0472, 2.0944),
        "right_wrist_roll_joint": (-1.972222054, 1.972222054),
        "right_wrist_pitch_joint": (-1.614429558, 1.614429558),
        "right_wrist_yaw_joint": (-1.614429558, 1.614429558),

    }
        self.limit_buffer = 0.05

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute penalty for exceeding joint position limits."""
        robot = states.robots[robot_name]
        joint_pos = robot.joint_pos  # [num_envs, num_dof]
        joint_names = robot.joint_names
        device = joint_pos.device
        for i, name in enumerate(joint_names):
            if name not in self.joint_limits:
                continue
                #raise ValueError(f"Joint {name} limits not defined.")
            low, high = self.joint_limits[name]
            low_tensor = torch.tensor(low, device=device)
            high_tensor = torch.tensor(high, device=device)
            # Penalizuj překročení limitů
            below_low = torch.relu((low_tensor + self.limit_buffer) - joint_pos[:, i])
            above_high = torch.relu(joint_pos[:, i] - (high_tensor - self.limit_buffer))
            penalty = below_low + above_high
            if i == 3:
                total_penalty = penalty
            else:
                total_penalty += penalty
        #print(f"Position limits penalty: {total_penalty.mean().item():.6f}")
        return total_penalty
class HumanlyDofLimitCfg(HumanoidBaseReward):
    """
    Humanly DoF limit: Penalizace za překročení 'lidsky přirozených' limitů.
    Váha dle paperu: -1.0

    Tato funkce nahrazuje mechanické limity robota (které jsou často příliš volné)
    přísnějšími limity, které odpovídají rozsahu pohybu člověka.

    Vzorec: sum( ( clip(q - q_lower, max=0) + clip(q - q_upper, min=0) )^2 )
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)

        # Cache pro tenzory (optimalizace rychlosti)
        self.dof_indices = None
        self.q_lower_tensor = None
        self.q_upper_tensor = None

        self.limit_buffer = 0.05  # Malá tolerance, než začne penalizace (soft limit)

        # DEFINICE LIDSKÝCH LIMITŮ (Upraveno z mechanických rozsahů G1)
        self.human_limits: dict[str, tuple[float, float]] = {
            # --- NOHY (LEGS) ---
            # Kyčle Pitch (Předkopávání/Zakopávání): Člověk nezakopne nohu o 145° dozadu (-2.5)
            "left_hip_pitch_joint": (-0.5, 1.75),    # Human: extenze ~-30°, flexe ~100°
            "right_hip_pitch_joint": (-0.5, 1.75),

            # Kyčle Roll (Rozkročování):
            "left_hip_roll_joint": (-0.2, 0.8),      # Omezeno, aby nedělal provaz
            "right_hip_roll_joint": (-0.8, 0.2),     # Pozor na symetrii znamének u G1

            # Kyčle Yaw (Rotace nohy): Mechanicky +/- 2.7 (nesmysl), člověk cca +/- 0.5
            "left_hip_yaw_joint": (-0.5, 0.5),
            "right_hip_yaw_joint": (-0.5, 0.5),

            # Kolena: Mechanicky -0.08 až 2.8. Člověk nemá hyperextenzi (záporné).
            "left_knee_joint": (0.0, 2.6),           # 0.0 = rovná noha
            "right_knee_joint": (0.0, 2.6),

            # Kotníky: Zhruba ponecháno, rozsah je malý
            "left_ankle_pitch_joint": (-0.6, 0.4),
            "left_ankle_roll_joint": (-0.26, 0.26),
            "right_ankle_pitch_joint": (-0.6, 0.4),
            "right_ankle_roll_joint": (-0.26, 0.26),

            # --- TRUP (WAIST) ---
            # Yaw (Rotace trupu): Mechanicky +/- 2.6 (skoro 360°). Člověk max +/- 1.0 (cca 60°)
            "waist_yaw_joint": (-1.0, 1.0),

            # Roll/Pitch (Úklony): Člověk se neohne o 0.5 rad do strany jen v pase bez páteře
            "waist_roll_joint": (-0.3, 0.3),
            "waist_pitch_joint": (-0.3, 0.5),        # Předklon povolen víc než záklon

            # --- PAŽE (ARMS) ---
            # Ramena Pitch (Zvedání ruky):
            "left_shoulder_pitch_joint": (-2.8, 2.5), # Velký rozsah je OK, ale oříznut extrém
            "right_shoulder_pitch_joint": (-2.8, 2.5),

            # Ramena Roll (Upažování): Omezeno křížení rukou přes hrudník
            "left_shoulder_roll_joint": (-0.5, 2.0),
            "right_shoulder_roll_joint": (-2.0, 0.5),

            # Ramena Yaw (Rotace v rameni):
            "left_shoulder_yaw_joint": (-1.6, 1.6),   # +/- 90° je zdravé maximum
            "right_shoulder_yaw_joint": (-1.6, 1.6),

            # Lokty: Mechanicky -1.0 (hyperextenze). Člověk 0.0 (rovná ruka) až flexe.
            # Vaše init pozice je 1.0, což je v pořádku (pokrčená ruka).
            "left_elbow_joint": (0.0, 2.1),           # Oříznuta hyperextenze (-1.0 -> 0.0)
            "right_elbow_joint": (0.0, 2.1),

            # Zápěstí: Ponecháno volnější pro manipulaci, ale oříznuty extrémy
            "left_wrist_roll_joint": (-1.5, 1.5),
            "left_wrist_pitch_joint": (-1.0, 1.0),
            "left_wrist_yaw_joint": (-1.0, 1.0),
            "right_wrist_roll_joint": (-1.5, 1.5),
            "right_wrist_pitch_joint": (-1.0, 1.0),
            "right_wrist_yaw_joint": (-1.0, 1.0),

            # Prsty zde neřešíme (řeší je FingerPrimitiveLimitsCfg)
        }

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute penalty for exceeding human-like joint limits."""
        robot = states.robots[robot_name]
        joint_pos = robot.joint_pos
        device = joint_pos.device

        # 1. Optimalizace: Vytvoření tensorů pouze při prvním spuštění
        if self.dof_indices is None:
            self.dof_indices = []
            lower_vals = []
            upper_vals = []

            # Projdeme všechny klouby robota a přiřadíme limity těm, které známe
            for i, name in enumerate(robot.joint_names):
                if name in self.human_limits:
                    self.dof_indices.append(i)
                    limits = self.human_limits[name]
                    # Přidáme buffer (soft limit)
                    lower_vals.append(limits[0] + self.limit_buffer)
                    upper_vals.append(limits[1] - self.limit_buffer)

            # Pokud bychom nenašli žádné klouby (což by bylo divné), vrátíme nulu
            if not self.dof_indices:
                return torch.zeros(joint_pos.shape[0], device=device)

            # Konverze na GPU tensory a uložení
            self.dof_indices = torch.tensor(self.dof_indices, device=device, dtype=torch.long)
            # Tvar (1, num_active_joints)
            self.q_lower_tensor = torch.tensor(lower_vals, device=device).unsqueeze(0)
            self.q_upper_tensor = torch.tensor(upper_vals, device=device).unsqueeze(0)

        # 2. Získání pozic pouze pro sledované klouby
        q_active = joint_pos[:, self.dof_indices]

        # 3. Výpočet penalizace dle vzorce z paperu (Table 2)
        # clip(q - q_lower, max=0) -> záporná hodnota, pokud q < lower
        violation_lower = torch.clamp(q_active - self.q_lower_tensor, max=0.0)

        # clip(q - q_upper, min=0) -> kladná hodnota, pokud q > upper
        violation_upper = torch.clamp(q_active - self.q_upper_tensor, min=0.0)

        # Součet "chyby" (jedna bude vždy 0, nebo obě 0)
        total_violation = violation_lower + violation_upper

        # Umocnění na druhou a suma
        penalty = torch.sum(torch.square(total_violation), dim=-1)
        #print(f"Humanly DoF limit penalty: {penalty.mean().item():.6f}")
        return penalty

class UprightPenaltyCfg(HumanoidBaseReward):
    """
    Upright penalty: Nutí robota držet trup svisle (osa Z).
    Podle DoorMan paperu (Table 2) je váha -1.0.

    Vzorec: || R_torso * [0, 0, 1]^T - [0, 0, 1]^T ||^2
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        # Target vector je světová osa Z [0, 0, 1]
        self.target_z = torch.tensor([0.0, 0.0, 1.0])

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute penalty for deviation from upright orientation."""
        robot = states.robots[robot_name]

        # Získání orientace trupu (root) jako quaternion [x, y, z, w]
        # Shape: (num_envs, 4)
        root_quat = robot.root_state[:, 3:7]
        device = root_quat.device

        # Ujistíme se, že target je na správném zařízení
        if self.target_z.device != device:
            self.target_z = self.target_z.to(device)

        # Extrakce Z-osy z rotace (quaternionu)
        # Pokud R je rotační matice odpovídající q, pak R * [0,0,1]^T je přesně 3. sloupec matice R.
        # Vzorec pro 3. sloupec matice z quaternionu [x, y, z, w]:
        # z_x = 2(xz + yw)
        # z_y = 2(yz - xw)
        # z_z = 1 - 2(x^2 + y^2)

        w, y, z, x = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]

        current_z_x = 2 * (x * z + y * w)
        current_z_y = 2 * (y * z - x * w)
        current_z_z = 1 - 2 * (x * x + y * y)

        # Sestavení vektoru aktuální osy Z [num_envs, 3]
        current_z_axis = torch.stack([current_z_x, current_z_y, current_z_z], dim=1)

        # Výpočet rozdílu vektorů: || current_z - target_z ||
        # Target Z je [0, 0, 1], rozbroadcastujeme ho pro odečtení
        diff = current_z_axis - self.target_z

        # Výpočet druhé mocniny normy (squared euclidean distance)
        # ||v||^2 = sum(v_i^2)
        penalty = torch.sum(torch.square(diff), dim=-1)
        #print(f"Upright penalty: {penalty.mean().item():.6f}")
        return penalty

class StageProgressCfg(HumanoidBaseReward):
    """
    Stage progress: Odměna za aktuální dosažený stage.
    Podle DoorMan paperu (Table 2) je váha 1.0.

    Formula: stage_current
    Funguje jako dense reward, který motivuje robota zůstat ve vyšších fázích.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        # Pokud není actual_stage inicializováno, vrátíme 0
        if self.completed_stages.any():
            ret = self.completed_stages * self.actual_stage.float()
            self.completed_stages = torch.zeros_like(self.completed_stages) # Reset pro další výpočet
            return ret
        else:
            return torch.zeros_like(self.completed_stages)

#---------------------stage 0----------------------

class WalkToChairReward(HumanoidBaseReward):
    """
    Stage 0: Walk to chair
    Gaussian odměna za minimalizaci vzdálenosti a směru k cíli (velocity tracking).
    Aplikuje se POUZE ve Stage 0. Robot nyní automaticky brzdí, aby zastavil před židlí.
    """
    def __init__(self, robot_name="g1_slider", target_speed=0.6):
        super().__init__(robot_name)
        self.sigma = 0.15

        # Maximální cílová rychlost chůze (v m/s)
        self.target_speed = target_speed

        # Parametry brzdění
        self.stop_distance = 0.7  # Vzdálenost od středu židle, kde má robot mít rychlost 0 (odpovídá stage0 thresholdům)
        self.braking_distance = 0.4  # Na jaké dráze začne robot zpomalovat z target_speed na 0

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        # 1. Kontrola Stage
        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        # 2. Vytvoření masky pro Stage 0
        stage_mask = (self.actual_stage == 0)
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        # 3. Získání pozic a rychlostí
        base_link_idx = robot.body_names.index("pelvis")
        root_pos = robot.body_state[:, base_link_idx, :3]
        root_vel = robot.body_state[:, base_link_idx, 7:10] # Lineární rychlost

        chair_base_link_idx = chair.body_names.index("base_link")
        target_pos = chair.body_state[:, chair_base_link_idx, :3] # Pozice židle

        # 4. Výpočet směrového vektoru d_chair a vzdálenosti
        vec_to_chair = target_pos - root_pos
        vec_to_chair[:, 2] = 0.0 # Ignorujeme Z složku (chůze po rovině)

        dist = torch.norm(vec_to_chair, dim=-1, keepdim=True)
        dir_to_chair = vec_to_chair / (dist + 1e-6)

        # --- NOVÉ: Dynamický výpočet cílové rychlosti ---
        # dist_to_stop říká, kolik metrů zbývá k místu zastavení
        dist_to_stop = dist - self.stop_distance

        # speed_factor klesá lineárně od 1.0 do 0.0 na úseku braking_distance
        speed_factor = torch.clamp(dist_to_stop / self.braking_distance, min=0.0, max=1.0)

        # Výsledná rychlost (robot brzdí, když je blízko)
        dynamic_speed = self.target_speed * speed_factor

        # 5. Cílový vektor rychlosti
        target_vel_vec = dynamic_speed * dir_to_chair

        # 6. Výpočet chyby rychlosti: ||v_robot - target_vel_vec||^2
        vel_error_sq = torch.sum(torch.square(root_vel - target_vel_vec), dim=-1)

        # 7. Výpočet odměn
        # Odměna za rychlost (bude tlačit robota do pohybu, nebo do zastavení - podle toho, kde stojí)
        vel_reward = torch.exp(-vel_error_sq / (2 * self.sigma**2))


        total_reward = 1.0 * vel_reward

        # 8. Aplikace masky
        total_reward = total_reward * stage_mask.float()

        return total_reward
class FaceChairReward(HumanoidBaseReward):
    """
    Face chair: Penalizace za špatnou orientaci (Yaw) vůči židli.
    Aktivní ve Stages: 0, 1, 2, 5.

    Interpretace: Robot musí srovnat své natočení (Yaw) s natočením rámu židle.
    To zajistí, že ve Stage 0 jde kolmo ke židli a ve Stage 5 pokračuje rovně skrz ně
    (neotáčí se zpět na židli).

    Formula: |wrap_pi( ||axis-angle(R_chair)||_2 )|
    Weight: -1.0
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        # Stages: 0-2 (příchod, úchop) a 5 (průchod)
        self.active_stages = [0, 1, 2, 5]

    def _wrap_to_pi(self, angle):
        """Převede úhel do intervalu [-pi, pi]."""
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        # 1. Kontrola Stage
        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        active_stages_tensor = torch.tensor(self.active_stages, device=device)
        stage_mask = torch.isin(self.actual_stage, active_stages_tensor)

        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        # 2. Získání Yaw (natočení) robota
        # Root state: [pos(3), quat(4), ...]
        q_r = robot.root_state[:, 3:7] # x, y, z, w
        x, y, z, w = q_r[:, 0], q_r[:, 1], q_r[:, 2], q_r[:, 3]

        # Vzorec pro Yaw z quaternionu
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        robot_yaw = torch.atan2(siny_cosp, cosy_cosp)

        # 3. Získání Yaw (natočení) židle
        # Předpokládáme, že "chair" objekt reprezentuje rám (frame), který se nehýbe
        q_d = chair.root_state[:, 3:7]
        xd, yd, zd, wd = q_d[:, 0], q_d[:, 1], q_d[:, 2], q_d[:, 3]

        siny_cosp_d = 2 * (wd * zd + xd * yd)
        cosy_cosp_d = 1 - 2 * (yd * yd + zd * zd)
        door_yaw = torch.atan2(siny_cosp_d, cosy_cosp_d)

        # POZNÁMKA: Zde záleží na tom, jak jsou dveře v simulaci otočeny.
        # Pokud "forward" osa dveří směřuje tam, kam má robot jít, chceme rozdíl 0.
        # Pokud dveře směřují "proti" robotovi, chtěli bychom rozdíl PI.
        # Standardně v DoorMan (Stage 5 pass through) chceme, aby robot a dveře měli
        # shodnou orientaci směru průchodu.

        # 4. Výpočet chyby orientace (rozdíl úhlů)
        yaw_error = self._wrap_to_pi(robot_yaw - door_yaw)

        # Absolutní hodnota chyby
        penalty = torch.abs(yaw_error)

        # 5. Aplikace masky
        #print(f"Face chair penalty: {penalty.mean().item():.6f}")
        return penalty * stage_mask.float()

#---------------------stage 1----------------------

class ReachChairReward(HumanoidBaseReward):
    """
    Stage 1: Reach chair (Dual Arm)
    Reward for minimizing distance between robot end-effectors and target points on the chair.

    Paper Reference: Table 2, Stage 1 "Pre-grasp target distance"
    Formula: exp(-||p_hand - p_target||^2 / (2 * sigma^2))
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.sigma = 0.15 # Precision sigma from DoorMan paper
        self.active_stages = [1]

        # Define body names based on user prompt
        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        # 1. Check Stage
        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        active_stages_tensor = torch.tensor(self.active_stages, device=device)
        stage_mask = torch.isin(self.actual_stage, active_stages_tensor)

        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        # 2. Get Robot Hand Positions
        # We assume these body names exist in the URDF/Sim
        try:
            r_left_idx = robot.body_names.index(self.robot_left_hand)
            r_right_idx = robot.body_names.index(self.robot_right_hand)
            p_hand_left = robot.body_state[:, r_left_idx, :3]
            p_hand_right = robot.body_state[:, r_right_idx, :3]

            # 3. Get Chair Target Positions
            c_left_idx = chair.body_names.index(self.chair_target_left)
            c_right_idx = chair.body_names.index(self.chair_target_right)
            p_target_left = chair.body_state[:, c_left_idx, :3]
            p_target_right = chair.body_state[:, c_right_idx, :3]
        except ValueError as e:
            # Fallback for safety if names are wrong during testing
            print(f"Body name error: {e}")
            return torch.zeros(num_envs, device=device)

        # 4. Compute Squared Distances
        dist_sq_left = torch.sum(torch.square(p_hand_left - p_target_left), dim=-1)
        dist_sq_right = torch.sum(torch.square(p_hand_right - p_target_right), dim=-1)

        # 5. Compute Gaussian Reward (Average of both hands)
        # DoorMan uses exp(-error^2 / 2sigma^2)
        rew_left = torch.exp(-dist_sq_left / (2 * self.sigma**2))
        rew_right = torch.exp(-dist_sq_right / (2 * self.sigma**2))

        total_reward = (rew_left + rew_right) / 2.0

        # 6. Apply Mask
        return total_reward * stage_mask.float()
class HandOrientationReward(HumanoidBaseReward):
    """
    Stage 1: Hand Orientation
    Reward for aligning hand orientation with the target orientation.

    Paper Reference: Table 2, Stage 1 "Hand-handle orientation"
    Formula: exp(-wrap(axis_angle(R_hand - R_target))^2 / (2 * sigma^2))
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.sigma = 0.6 # Looser sigma for orientation as per paper
        self.active_stages = [1]

        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

    def _quat_diff_angle(self, q1, q2):
        """Calculates 2 * acos(|<q1, q2>|) to get angle difference."""
        # Quaternion dot product
        dot = torch.sum(q1 * q2, dim=-1)
        # Clamp for numerical stability
        dot = torch.clamp(torch.abs(dot), max=1.0)
        return 2.0 * torch.acos(dot)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None: return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any(): return torch.zeros(num_envs, device=device)

        try:
            # Indices
            rl_idx = robot.body_names.index(self.robot_left_hand)
            rr_idx = robot.body_names.index(self.robot_right_hand)
            cl_idx = chair.body_names.index(self.chair_target_left)
            cr_idx = chair.body_names.index(self.chair_target_right)

            # Quaternions [x, y, z, w] -> Reordering to [w, x, y, z] might be needed depending on sim
            # Assuming sim provides standard quats.
            # Note: metasim usually provides [x, y, z, w] or [w, x, y, z].
            # The dot product method works regardless of order as long as consistent.
            q_hand_left = robot.body_state[:, rl_idx, 3:7]
            q_hand_right = robot.body_state[:, rr_idx, 3:7]
            q_target_left = chair.body_state[:, cl_idx, 3:7]
            q_target_right = chair.body_state[:, cr_idx, 3:7]

        except ValueError:
            return torch.zeros(num_envs, device=device)

        # Calculate angular errors
        angle_diff_left = self._quat_diff_angle(q_hand_left, q_target_left)
        angle_diff_right = self._quat_diff_angle(q_hand_right, q_target_right)

        # Gaussian Reward
        rew_left = torch.exp(-torch.square(angle_diff_left) / (2 * self.sigma**2))
        rew_right = torch.exp(-torch.square(angle_diff_right) / (2 * self.sigma**2))

        return ((rew_left + rew_right) / 2.0) * stage_mask.float()
class StandStillPenalty(HumanoidBaseReward):
    """
    Stage 1: Stability / Stand Still
    Penalizes root velocity during manipulation stages to ensure stable grasp.

    Paper Reference: Table 2, Stage 1 "Penalty not standing still"
    Formula: ||v_root||^2
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [1, 2] # Active during pre-grasp and grasp

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None: return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))

        # Root linear velocity (usually indices 7:10 in body_state for the root/pelvis)
        # Assuming pelvis is body 0 or explicitly named
        base_idx = robot.body_names.index("pelvis")
        root_vel = robot.body_state[:, base_idx, 7:10]

        velocity_sq = torch.sum(torch.square(root_vel), dim=-1)

        # Paper weight is -1.0 for this penalty
        return velocity_sq * stage_mask.float()
class OpenGraspReward(HumanoidBaseReward):
    """
    Stage 1: Open Grasp Reward
    Forces the hand to stay open (target position 0.0) and still during the pre-grasp phase.

    Based on provided limits, 0.0 corresponds to the fully extended (open) state
    for both left (negative flexion) and right (positive flexion) hands.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        # Váhy a sigma dle paperu Doorman (Table 2)
        self.sigma_pos = 0.3
        self.sigma_vel = 0.2
        self.target_angle = 0.0   # 0.0 je otevřená ruka pro vaše limity
        self.active_stages = [1]  # Aktivní pouze v Pre-grasp fázi

        # Cache pro indexy
        self.finger_indices = None
        self.target_tensor = None

        # Seznam klíčových slov pro identifikaci prstů
        # Můžeme být specifičtí dle vašeho seznamu (thumb, index, middle)
        self.finger_keywords = [
            "thumb", "index", "middle", "pinky", "ring", "hand"
        ]

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        # 1. Kontrola Stage (pouze pokud je definována)
        if self.actual_stage is not None:
            active_stages_tensor = torch.tensor(self.active_stages, device=device)
            stage_mask = torch.isin(self.actual_stage, active_stages_tensor)
            if not stage_mask.any():
                return torch.zeros(num_envs, device=device)
        else:
            # Fallback pokud stage neexistuje (např. testování), aplikujeme stále
            stage_mask = torch.ones(num_envs, device=device, dtype=torch.bool)

        # 2. Inicializace indexů (pouze při prvním průchodu)
        if self.finger_indices is None:
            self.finger_indices = []

            for idx, joint_name in enumerate(robot.joint_names):
                # Kontrola, zda je kloub prstem
                if any(k in joint_name for k in self.finger_keywords):
                    self.finger_indices.append(idx)

            if not self.finger_indices:
                # Pokud nenajdeme prsty, vrátíme nuly (prevence pádu)
                return torch.zeros(num_envs, device=device)

            self.finger_indices = torch.tensor(self.finger_indices, device=device, dtype=torch.long)

            # Vytvoříme tensor cílových hodnot (samé nuly)
            # Shape: (1, num_fingers) pro broadcasting
            self.target_tensor = torch.full((1, len(self.finger_indices)), self.target_angle, device=device)

        # 3. Získání aktuálních hodnot
        # Shape: (num_envs, num_fingers)
        q_finger = robot.joint_pos[:, self.finger_indices]
        #dq_finger = robot.joint_vel[:, self.finger_indices]

        # 4. Reward za POZICI (Position tracking)
        # Snažíme se dostat q_finger na 0.0
        # Formula: exp(-||q - 0||^2 / 2sigma^2)
        pos_error_sq = torch.sum(torch.square(q_finger - self.target_tensor), dim=-1)
        pos_reward = torch.exp(-pos_error_sq / (2 * self.sigma_pos**2))

        # 5. Reward za RYCHLOST (Velocity tracking)
        # Snažíme se mít prsty v klidu (dq = 0)
        # vel_error_sq = torch.sum(torch.square(dq_finger), dim=-1)
        # vel_reward = torch.exp(-vel_error_sq / (2 * self.sigma_vel**2))

        # 6. Celkový reward
        # Paper Doorman sčítá oba členy (Tabulka 2: track(...) + track(...))
        total_reward = pos_reward# + vel_reward

        return total_reward * stage_mask.float()
#---------------------Stage 2----------------------
#---------------------Stage 2----------------------

class CloseGraspReward(HumanoidBaseReward):
    """
    Stage 2: Close Grasp Reward
    Odměňuje robota za to, že zavírá prsty směrem k definovaným limitům (pevný úchop).

    Paper Reference: Table 2, Stage 2 "Grasp finger DoF pose"
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.sigma_pos = 0.3
        self.active_stages = [2]  # Aktivní pouze ve Stage 2

        # Cílové pozice prstů pro pevný úchop (z vaší předchozí konfigurace)

        # Hodnoty jsou vypočítány jako: (bod dotyku ze states) + (0.15 rad ve směru sevření)
        self.trashold = 0.1 # Přidáváme 0.15 rad pro pevnější sevření oproti původním limitům
        self.finger_targets_dict = {
            # --- LEVÁ RUKA ---
            # Palec se zavírá do PLUSU
            "left_hand_thumb_0_joint": 0.396 - self.trashold,   # (původně 0.396)
            "left_hand_thumb_1_joint": 0.214 - self.trashold,   # (původně 0.214)
            "left_hand_thumb_2_joint": 0.357 - self.trashold,   # (původně 0.357)
            # Ostatní prsty se zavírají do MÍNUSU
            "left_hand_middle_0_joint": -0.523 + self.trashold, # (původně -0.523)
            "left_hand_middle_1_joint": -0.527 + self.trashold, # (původně -0.527)
            "left_hand_index_0_joint": -0.485 + self.trashold,  # (původně -0.485)
            "left_hand_index_1_joint": -0.542 + self.trashold,  # (původně -0.542)

            # --- PRAVÁ RUKA ---
            # Palec se zavírá do MÍNUSU
            "right_hand_thumb_0_joint": -0.389 - self.trashold, # (původně -0.389)
            "right_hand_thumb_1_joint": -0.208 - self.trashold, # (původně -0.208)
            "right_hand_thumb_2_joint": -0.358 - self.trashold, # (původně -0.358)
            # Ostatní prsty se zavírají do PLUSU
            "right_hand_middle_0_joint": 0.505 + self.trashold, # (původně 0.505)
            "right_hand_middle_1_joint": 0.518 + self.trashold, # (původně 0.518)
            "right_hand_index_0_joint": 0.485 + self.trashold,  # (původně 0.485)
            "right_hand_index_1_joint": 0.541 + self.trashold   # (původně 0.541)
        }

        self.finger_indices = None
        self.target_tensor = None

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        # 1. Kontrola Stage
        if self.actual_stage is not None:
            active_stages_tensor = torch.tensor(self.active_stages, device=device)
            stage_mask = torch.isin(self.actual_stage, active_stages_tensor)
            if not stage_mask.any():
                return torch.zeros(num_envs, device=device)
        else:
            stage_mask = torch.ones(num_envs, device=device, dtype=torch.bool)

        # 2. Inicializace (pouze při prvním běhu)
        if self.finger_indices is None:
            indices = []
            targets = []
            for name, target_val in self.finger_targets_dict.items():
                if name in robot.joint_names:
                    index = list(robot.joint_names).index(name)
                    indices.append(index)
                    targets.append(target_val)

            if not indices:
                return torch.zeros(num_envs, device=device)

            self.finger_indices = torch.tensor(indices, device=device, dtype=torch.long)
            # Tvar pro broadcasting
            self.target_tensor = torch.tensor(targets, device=device).unsqueeze(0)

        # 3. Získání pozic prstů
        q_finger = robot.joint_pos[:, self.finger_indices]

        # 4. Výpočet Gaussianské odměny (Distance to target pose)
        # exp(-||q - q_closed||^2 / 2sigma^2)
        pos_error_sq = torch.sum(torch.square(q_finger - self.target_tensor), dim=-1)
        reward = torch.exp(-pos_error_sq / (2 * self.sigma_pos**2))

        return reward * stage_mask.float()


class GraspForceReward(HumanoidBaseReward):
    """
    Stage 2: Grasp Force Reward
    Odměňuje robota za generování síly do prstů při kontaktu se židlí.
    Síla se vyhodnocuje pro každý prst (palec, ukazovák, prostředník pro obě ruce) zvlášť.
    Maximální odměny (1.0) je dosaženo pouze tehdy, když VŠECHNY prsty působí silou větší
    nebo rovnou `force_threshold`.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [2]

        # Mapování konkrétních prstů na indexy (6 prstů celkem)
        self.finger_categories = {
            "left_hand_thumb_2_link": 0,
            "left_hand_index_1_link": 1,
            "left_hand_middle_1_link": 2,
            "right_hand_thumb_2_link": 3,
            "right_hand_index_1_link": 4,
            "right_hand_middle_1_link": 5
        }
        self.num_fingers = len(self.finger_categories)

        # Cílová síla pro KAŽDÝ prst zvlášť.
        # Pokud je threshold 1.0, znamená to, že každý prst musí tlačit alespoň silou 1N.
        self.force_threshold = 1.0

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        # 1. Kontrola, zda jsme ve správné stage
        if self.actual_stage is not None:
            stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
            if not stage_mask.any():
                return torch.zeros(num_envs, device=device)
        else:
            stage_mask = torch.ones(num_envs, device=device, dtype=torch.bool)

        # 2. Inicializace tensoru pro síly: [num_envs, počet_prstů]
        # Pro každé prostředí a každý prst uchováváme maximální detekovanou sílu
        finger_forces = torch.zeros((num_envs, self.num_fingers), device=device)

        # 3. Zpracování kontaktů ze state bufferu
        if hasattr(robot, 'contact') and robot.contact is not None:
            for c in robot.contact:
                # Ochrana: získání env_id
                env_id = c.get('env_id', None)
                if env_id is None or env_id >= num_envs:
                    continue

                # Zohledníme pouze kontakt se židlí
                is_chair = (c.get('body_a') == "chair" or c.get('body_b') == "chair")
                if not is_chair:
                    continue

                # Určení názvu článku (linku) robota
                robot_link = c['link_a'] if c.get('body_b') == "chair" else c['link_b']

                # Identifikace konkrétního prstu, který se židle dotýká
                finger_idx = None
                for prefix, idx in self.finger_categories.items():
                    if prefix in robot_link:
                        finger_idx = idx
                        break

                # Pokud kontakt patří jednomu z našich sledovaných prstů
                if finger_idx is not None:
                    force = c.get('force', 0.0)

                    # Získání skalární hodnoty síly
                    if isinstance(force, (list, tuple)):
                        import numpy as np
                        force = float(np.linalg.norm(force))
                    elif hasattr(force, "item"):
                        force = float(force.item())

                    # Pro daný prst si uložíme maximální naměřenou sílu v tomto kroku
                    if force > finger_forces[env_id, finger_idx]:
                        finger_forces[env_id, finger_idx] = force

        # 4. Výpočet odměny
        # Pro každý prst spočítáme dílčí odměnu (poměr k thresholdu, max 1.0)
        # Výsledek bude tensor o velikosti [num_envs, 6], kde hodnoty jsou 0.0 až 1.0
        finger_rewards = torch.clamp(finger_forces / self.force_threshold, max=1.0)

        # Celková odměna je průměrem odměn všech prstů.
        # Díky tomu robot dostane 1.0 jen tehdy, pokud má na všech 6 prstech odměnu 1.0.
        # (Používáme průměr místo "all()", aby robot dostával postupnou odměnu za každý přidaný prst)
        reward = torch.mean(finger_rewards, dim=1)

        # Aplikování stage masky
        return reward * stage_mask.float()



TERMINATION_WEIGHT = -1000.0
DELTA_ACTION_RATE_WEIGHT = -0.01
DOF_VELOCITY_ACCELERATION_WEIGHT = 1.0
DOF_POSITION_LIMITS_WEIGHT = -5.0
HUMANLY_DOF_LIMIT_WEIGHT = -1.0
UPRIGHT_PENALTY_WEIGHT = -1.0
STAGE_PROGRESS_WEIGHT = 1.0
WALK_TO_CHAIR_REWARD_WEIGHT = 5.0
FACE_CHAIR_REWARD_WEIGHT = -1.0
REACH_CHAIR_REWARD_WEIGHT = 6.0
REACH_ORIENTATION_REWARD_WEIGHT = 3.0
STAND_STILL_PENALTY_WEIGHT = -1.0
OPEN_GRASP_REWARD_WEIGHT = 1.5
CLOSE_GRASP_REWARD_WEIGHT = 3.0
FORCE_GRASP_REWARD_WEIGHT = 3.0


@configclass
class ChairmanCfg(HumanoidTaskCfg):
    """Chair task for humanoid robots."""




    success_bar = 0.9
    episode_length = 400
    objects = [
        ArticulationObjCfg(
            name="chair",
            urdf_path="roboverse_data/assets/humanoidbench/chairs/chair1/foldable_chair_debug.urdf",
            default_position= [0.0, 0.0, 0.0],
            fix_base_link=True,
            colapse_fixed_joints=False,
            batch_fixed_verts=True
        )
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/chair/initial_state_v2.json"
    checker = _ChairManChecker()
    reward_weights = [
        TERMINATION_WEIGHT,
        DELTA_ACTION_RATE_WEIGHT,
        DOF_VELOCITY_ACCELERATION_WEIGHT,
        DOF_POSITION_LIMITS_WEIGHT,
        HUMANLY_DOF_LIMIT_WEIGHT,
        UPRIGHT_PENALTY_WEIGHT,
        STAGE_PROGRESS_WEIGHT,
        WALK_TO_CHAIR_REWARD_WEIGHT,
        FACE_CHAIR_REWARD_WEIGHT,
        REACH_CHAIR_REWARD_WEIGHT,
        REACH_ORIENTATION_REWARD_WEIGHT,
        STAND_STILL_PENALTY_WEIGHT,
        OPEN_GRASP_REWARD_WEIGHT,
        CLOSE_GRASP_REWARD_WEIGHT,
        FORCE_GRASP_REWARD_WEIGHT
    ]
    #function_index_success_save_time = 10 #TODO hloupé řešení ale budiž to tak (potřeba opravit)
    reward_functions = [TerminationCfg(),
                        DeltaActionRateCfg(),
                        DoFVelocityAccelerationCfg(),
                        DofPositionLimitsCfg(),
                        HumanlyDofLimitCfg(),
                        UprightPenaltyCfg(),
                        StageProgressCfg(),
                        WalkToChairReward(),
                        FaceChairReward(),
                        ReachChairReward(),
                        HandOrientationReward(),
                        StandStillPenalty(),
                        OpenGraspReward(),
                        CloseGraspReward(),
                        GraspForceReward()
                        ]
    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}

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
        torso_link_idx = robot.body_names.index("torso_link")
        root_quat = robot.body_state[:, torso_link_idx, 3:7]
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

        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]

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
class ContinuousStageReward(HumanoidBaseReward):
    """
    Continuous Stage Reward: Dává permanentní odměnu za to, ve kterém Stage se robot nachází.
    Stage 0 = 0 bodů
    Stage 1 = 1 * váha
    Stage 2 = 2 * váha
    ... atd.

    Tímto robotovi jasně říkáme, že udržet se v pozdějších fázích je matematicky
    nejvýhodnější věc v celé hře.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        # 1. Ochrana pro úplně první krok, kdy stage ještě nemusí být zinicializován
        if self.actual_stage is None:
            robot = states.robots[robot_name]
            num_envs = robot.joint_pos.shape[0]
            device = robot.joint_pos.device
            return torch.zeros(num_envs, device=device)

        # 2. Jednoduše vrátíme aktuální číslo stage (0, 1, 2, 3...)
        # Váš framework (Metasim/Gym wrapper) tuto hodnotu následně
        # automaticky vynásobí váhou, kterou máte definovanou v configu.
        return self.actual_stage.float()

#---------------------stage 0----------------------

class WalkToChairReward(HumanoidBaseReward):
    """
    Stage 0: Walk to chair
    Kombinuje velocity tracking (pro plynulou chůzi) a penalizaci za couvání.
    """
    def __init__(self, robot_name="g1_slider", target_speed=0.8):
        super().__init__(robot_name)
        self.sigma = 0.15
        self.target_speed = target_speed
        self.active_stages = [0,1,2]
        self.stop_distance = 0.76
        self.braking_distance = 0.5

        # Váha trestu za couvání. Musí být dost velká, aby přebila zisk z následného pohybu vpřed.
        # Pokud je 5.0, tak za každý 1 m/s rychlosti dozadu dostane -5 bodů.
        self.backward_penalty_weight = 50.0

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None: return torch.zeros(num_envs, device=device)
        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any(): return torch.zeros(num_envs, device=device)

        base_link_idx = robot.body_names.index("pelvis")
        root_pos = robot.body_state[:, base_link_idx, :3]
        root_vel = robot.body_state[:, base_link_idx, 7:10]

        chair_base_link_idx = chair.body_names.index("base_link")
        target_pos = chair.body_state[:, chair_base_link_idx, :3]

        vec_to_chair = target_pos - root_pos
        vec_to_chair[:, 2] = 0.0

        # Vypočítáme vzdálenost [N]
        dist = torch.norm(vec_to_chair, dim=-1)
        # Normalizovaný směr k židli [N, 3]
        dir_to_chair = vec_to_chair / (dist.unsqueeze(-1) + 1e-6)

        # --- ČÁST 1: Cílová rychlost (Gaussian Reward) ---
        dist_to_stop = dist - self.stop_distance
        speed_factor = torch.clamp(dist_to_stop / self.braking_distance, min=0.0, max=1.0)
        dynamic_speed = self.target_speed * speed_factor

        target_vel_vec = dynamic_speed.unsqueeze(-1) * dir_to_chair
        vel_error_sq = torch.sum(torch.square(root_vel - target_vel_vec), dim=-1)

        # Kladná odměna za správný pohyb (0 až 1)
        vel_reward = torch.exp(-vel_error_sq / (2 * self.sigma**2))

        # --- ČÁST 2: Penalizace za couvání (Backward Penalty) ---
        # Spočítáme projekci rychlosti robota do směru k židli
        # Kladné číslo = jde k židli, Záporné číslo = couvá
        velocity_projection = torch.sum(root_vel * dir_to_chair, dim=-1)

        # Vezmeme jen záporné hodnoty (couvání) a ořízneme kladné na 0
        backward_movement = torch.clamp(velocity_projection, max=0.0)

        # Vynásobíme velkou vahou (např. 5.0).
        # Výsledek bude záporné číslo (např. -0.5 m/s * 5.0 = -2.5 reward)
        backward_penalty = backward_movement * self.backward_penalty_weight

        # --- Celkový reward ---
        # Pokud couvá, dostane (malý vel_reward) + (velký záporný penalty)
        total_reward = (1.0 * vel_reward) + backward_penalty

        return total_reward * stage_mask.float()

class FaceChairReward(HumanoidBaseReward):
    """
    Face Chair: Udržuje pohled robota na židli (Trychtýřová odměna & Trest za odvracení)
    Odměňuje robota za to, že osa X jeho hlavy směřuje k židli.
    Tvrdě penalizuje, pokud úhlová rychlost hlavy směřuje pohled pryč od židle.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        # Může být aktivní ve všech fázích, kdy chceme, aby robot sledoval cíl
        self.active_stages = [0, 1, 2, 3, 4, 5]

        # O kolik metrů výše nad base_link židle se má robot dívat (na sedák)
        self.chair_look_z_offset = 0.4

        # Váha trestu za odvracení zraku (úhlová rychlost pryč od cíle)
        self.look_away_penalty_weight = 2.0

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None: return torch.zeros(num_envs, device=device)
        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any(): return torch.zeros(num_envs, device=device)

        try:
            head_link_idx = robot.body_names.index("head_link")
            chair_base_idx = chair.body_names.index("base_link")

            # 1. Pozice hlavy a židle
            head_pos = robot.body_state[:, head_link_idx, :3]
            chair_pos = chair.body_state[:, chair_base_idx, :3]

            # 2. Úhlová rychlost hlavy [N, 3] (indexy 10:13)
            head_ang_vel = robot.body_state[:, head_link_idx, 10:13]

            # 3. Orientace hlavy (Quaternion)
            q = robot.body_state[:, head_link_idx, 3:7]
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        except ValueError:
            return torch.zeros(num_envs, device=device)

        # --- A. VÝPOČET SMĚRŮ ---
        # Zvedneme cíl pohledu na úroveň sedáku
        target_pos = chair_pos.clone()
        target_pos[:, 2] += self.chair_look_z_offset

        # Vektor od hlavy k židli (Normalizovaný)
        vec_to_target = target_pos - head_pos
        dir_to_target = vec_to_target / (torch.norm(vec_to_target, dim=-1, keepdim=True) + 1e-6)

        # Vektor, kam reálně hlava KOUKÁ (Osa X z quaternionu)
        forward_x = 1 - 2 * (y**2 + z**2)
        forward_y = 2 * (x*y + w*z)
        forward_z = 2 * (x*z - w*y)
        head_forward_vec = torch.stack([forward_x, forward_y, forward_z], dim=-1)

        # --- B. ODMĚNA ZA POHLED (Trychtýřová odměna) ---
        # Dot product: 1.0 = kouká přesně tam, -1.0 = kouká dozadu
        alignment = torch.sum(head_forward_vec * dir_to_target, dim=-1)

        # Uděláme z toho chybu: 0.0 = perfektní, 2.0 = nejhorší
        look_error = 1.0 - alignment

        # Trychtýř (Inverse Distance): Čím menší chyba, tím strměji roste odměna k 1.0
        rew_look = 1.0 / (1.0 + 5.0 * look_error)

        # --- C. PENALIZACE ZA ODVRACENÍ ZRAKU (Angular Velocity Penalty) ---
        # Křížový součin (Cross Product) nám dá OSU, kolem které se musí hlava
        # otočit, aby se forward_vec srovnal s dir_to_target.
        correction_axis = torch.cross(head_forward_vec, dir_to_target, dim=-1)

        # Promítneme reálnou úhlovou rychlost hlavy na tuto ideální korekční osu.
        # - Kladné číslo = hlava se otáčí K židli (Správně)
        # - Záporné číslo = hlava se otáčí PRYČ od židle (Špatně!)
        turn_progress = torch.sum(head_ang_vel * correction_axis, dim=-1)

        # Ořízneme kladné hodnoty (neodměňujeme za rychlost otáčení, chceme jen klidný pohled)
        # a ponecháme jen záporné hodnoty (odvracení zraku)
        turning_away = torch.clamp(turn_progress, max=0.0)

        # Aplikace trestu
        penalty_turn = turning_away * self.look_away_penalty_weight

        # --- D. CELKOVÉ SKÓRE ---
        # Robot dostává body za to, že kouká na židli (rew_look),
        # ale pokud cukne hlavou jinam, dostane facku (penalty_turn).
        total_reward = rew_look + penalty_turn

        return total_reward * stage_mask.float()
class ArmRestingPosePenaltyCfg(HumanoidBaseReward):
    """
    Stage 0: Penalizace za rozhazování rukama během chůze.
    Aktivní pouze ve Stage 0.

    Nutí robota držet ruce v klidové poloze podél těla. Povoluje pouze malý
    kývavý pohyb (cca +/- 0.3 rad) nutný pro přirozenou chůzi.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [0, 5]

        # Cache
        self.dof_indices = None
        self.q_lower_tensor = None
        self.q_upper_tensor = None

        # Přísné limity pro ruce podél těla (Stage 0)
        # Výchozí póza G1 má ruce svisle dolů. Povolíme jen malý kyv pro rovnováhu.
        self.resting_limits: dict[str, tuple[float, float]] = {
            # Ramena Pitch (předpažování/zapažování) - povolíme lehký kyv
            "left_shoulder_pitch_joint": (-0.3, 0.3),
            "right_shoulder_pitch_joint": (-0.3, 0.3),

            # Ramena Roll (upažování) - zakážeme máchání do stran
            "left_shoulder_roll_joint": (-0.1, 0.1),
            "right_shoulder_roll_joint": (-0.1, 0.1),

            # Ramena Yaw (rotace v rameni)
            "left_shoulder_yaw_joint": (-0.1, 0.1),
            "right_shoulder_yaw_joint": (-0.1, 0.1),

            # Lokty - G1 by je měl mít natažené (0.0), dovolíme max mírné pokrčení
            "left_elbow_joint": (-0.1, 0.3),
            "right_elbow_joint": (-0.1, 0.3),
        }

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        joint_pos = robot.joint_pos
        device = joint_pos.device
        num_envs = joint_pos.shape[0]

        # 1. Kontrola Stage
        if self.actual_stage is None: return torch.zeros(num_envs, device=device)
        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any(): return torch.zeros(num_envs, device=device)

        # 2. Inicializace (pouze poprvé)
        if self.dof_indices is None:
            self.dof_indices = []
            lower_vals = []
            upper_vals = []

            for i, name in enumerate(robot.joint_names):
                if name in self.resting_limits:
                    self.dof_indices.append(i)
                    limits = self.resting_limits[name]
                    lower_vals.append(limits[0])
                    upper_vals.append(limits[1])

            if not self.dof_indices:
                return torch.zeros(num_envs, device=device)

            self.dof_indices = torch.tensor(self.dof_indices, device=device, dtype=torch.long)
            self.q_lower_tensor = torch.tensor(lower_vals, device=device).unsqueeze(0)
            self.q_upper_tensor = torch.tensor(upper_vals, device=device).unsqueeze(0)

        # 3. Výpočet chyby
        q_active = joint_pos[:, self.dof_indices]

        violation_lower = torch.clamp(q_active - self.q_lower_tensor, max=0.0)
        violation_upper = torch.clamp(q_active - self.q_upper_tensor, min=0.0)

        total_violation = violation_lower + violation_upper
        penalty = torch.sum(torch.square(total_violation), dim=-1)

        return penalty * stage_mask.float()


#---------------------stage 1----------------------
class ReachChairReward(HumanoidBaseReward):
    """
    Stage 1: Reach chair (Distance & Retreat Penalty)
    Odměňuje robota výhradně za zkracování vzdálenosti k cíli a tvrdě penalizuje,
    pokud ruce pohybují směrem od cíle.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [1,2,3]
        self.robot_left_hand = "left_endeffector"
        self.robot_right_hand = "endeffector"
        self.chair_target_left = "target_hand_left"
        self.chair_target_right = "target_hand_right"

        # Váha trestu za to, že ruka letí pryč od cíle (čím větší číslo, tím tvrdší trest)
        self.retreat_penalty_weight = 5.0

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None: return torch.zeros(num_envs, device=device)
        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any(): return torch.zeros(num_envs, device=device)

        try:
            r_left_idx = robot.body_names.index(self.robot_left_hand)
            r_right_idx = robot.body_names.index(self.robot_right_hand)

            # 1. Získání 3D POZIC rukou
            p_hand_left = robot.body_state[:, r_left_idx, :3]
            p_hand_right = robot.body_state[:, r_right_idx, :3]

            # 2. Získání lineárních RYCHLOSTÍ rukou (pro trest za oddalování)
            v_hand_left = robot.body_state[:, r_left_idx, 7:10]
            v_hand_right = robot.body_state[:, r_right_idx, 7:10]

            c_left_idx = chair.body_names.index(self.chair_target_left)
            c_right_idx = chair.body_names.index(self.chair_target_right)

            # Získání POZIC cílů
            p_target_left = chair.body_state[:, c_left_idx, :3]
            p_target_right = chair.body_state[:, c_right_idx, :3]

        except ValueError:
            return torch.zeros(num_envs, device=device)

        # --- VÝPOČET PRO LEVOU RUKU ---
        # Vektor od ruky k madlu
        vec_left = p_target_left - p_hand_left
        dist_left = torch.norm(vec_left, dim=-1)
        dir_left = vec_left / (dist_left.unsqueeze(-1) + 1e-6) # Normalizovaný směr

        # A) Odměna za vzdálenost (1 / (1 + 10 * dist))
        # Vzdálenost 1m = 0.09 bodů | 10cm = 0.5 bodů | 2cm = 0.83 bodů | 0cm = 1.0 bodů
        rew_dist_left = 1.0 / (1.0 + 10.0 * dist_left)

        # B) Penalizace za ucuknutí rukou (záporná projekce rychlosti)
        vel_proj_left = torch.sum(v_hand_left * dir_left, dim=-1)
        # Bereme pouze situace, kdy je rychlost k cíli záporná (tj. ruka se vzdaluje)
        retreat_left = torch.clamp(vel_proj_left, max=0.0)
        penalty_left = retreat_left * self.retreat_penalty_weight

        # --- VÝPOČET PRO PRAVOU RUKU ---
        vec_right = p_target_right - p_hand_right
        dist_right = torch.norm(vec_right, dim=-1)
        dir_right = vec_right / (dist_right.unsqueeze(-1) + 1e-6)

        rew_dist_right = 1.0 / (1.0 + 10.0 * dist_right)

        vel_proj_right = torch.sum(v_hand_right * dir_right, dim=-1)
        retreat_right = torch.clamp(vel_proj_right, max=0.0)
        penalty_right = retreat_right * self.retreat_penalty_weight

        # --- CELKOVÉ SKÓRE ---
        # Poskládání dohromady: Ruka je tažena magnetem (rew_dist), ale kope ho proud, když cukne pryč (penalty) XD
        total_left = rew_dist_left + penalty_left
        total_right = rew_dist_right + penalty_right

        total_reward = (total_left + total_right) / 2.0

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
        self.active_stages = [1,2,3]

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
    Stage 1, 2, 4, 5: Stability / Stand Still
    Penalizes movement from a saved anchor position.
    The penalty grows exponentially with the distance from the anchor.
    The anchor is reset upon entering Stage 1, 4, or 5.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [1, 2, 4, 5] # Active during pre-grasp, grasp, and keep chair still

        # Stavové proměnné pro logiku uložení pozice
        self.saved_positions = None
        self.prev_stages = None

        # Koeficient strmosti exponenciály (čím vyšší, tím rychleji penalizace roste)
        self.alpha = 5.0

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))

        # Root pozice robota [num_envs, 3]
        base_idx = robot.body_names.index("pelvis")
        current_pos = robot.body_state[:, base_idx, :3]

        # 1. Inicializace tenzorů při úplně prvním kroku
        if self.saved_positions is None:
            self.saved_positions = current_pos.clone()
            self.prev_stages = self.actual_stage.clone()

        # 2. Logika pro aktualizaci uložené pozice
        # Zjišťujeme, zda se prostředí právě přepnulo do nové stage
        stage_changed = (self.actual_stage != self.prev_stages)

        # Nechceme přepsat pozici při přechodu ze Stage 1 do Stage 2,
        # protože tam má pořád stát na tom samém místě jako při přípravě na úchop.
        is_1_to_2 = (self.prev_stages == 1) & (self.actual_stage == 2)

        # Maska pro updatování pozice (např. vlezl do Stage 1, Stage 4 nebo Stage 5)
        update_mask = stage_changed & stage_mask & ~is_1_to_2

        # Pokud nějaké prostředí splňuje podmínky, přepíšeme jeho uloženou pozici
        if update_mask.any():
            self.saved_positions[update_mask] = current_pos[update_mask].clone()

        # Uložíme si aktuální stage pro kontrolu v dalším kroku
        self.prev_stages = self.actual_stage.clone()

        # 3. Výpočet vzdálenosti od uloženého bodu (ve 3D)
        dist = torch.norm(current_pos - self.saved_positions, dim=-1)

        # 4. Exponenciální penalizace: exp(alpha * dist) - 1.0
        # - Pokud dist = 0 -> exp(0) - 1 = 0
        # - Pokud dist = 0.2 (20 cm) -> exp(5 * 0.2) - 1 = exp(1) - 1 = 1.71
        # - Pokud dist = 0.5 (50 cm) -> exp(5 * 0.5) - 1 = exp(2.5) - 1 = 11.18
        penalty = torch.exp(self.alpha * dist) - 1.0

        # Ochrana proti explozi gradientu (pokud by fyzikální engine vystřelil robota do vesmíru)
        penalty = torch.clamp(penalty, max=50.0)

        # Váha v configu (STAND_STILL_PENALTY_WEIGHT) se postará o to, aby toto číslo bylo záporné
        return penalty * stage_mask.float()
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
        self.active_stages = [0, 1, 4]  # Aktivní pouze v Pre-grasp fázi

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
class CloseGraspReward(HumanoidBaseReward):
    """
    Stage 2: Close Grasp Reward
    Odměňuje robota za to, že zavírá prsty směrem k definovaným limitům (pevný úchop).

    Paper Reference: Table 2, Stage 2 "Grasp finger DoF pose"
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.sigma_pos = 0.3
        self.active_stages = [2,3]  # Aktivní pouze ve Stage 2 a stage 3

        # Cílové pozice prstů pro pevný úchop (z vaší předchozí konfigurace)

        # Hodnoty jsou vypočítány jako: (bod dotyku ze states) + (0.15 rad ve směru sevření)
        self.trashold = 0.1 # Přidáváme 0.15 rad pro pevnější sevření oproti původním limitům
        self.finger_targets_dict = {
            # --- LEVÁ RUKA ---
            # Palec se zavírá do PLUSU
            "left_hand_thumb_0_joint": 0.396 + self.trashold,   # (původně 0.396)
            "left_hand_thumb_1_joint": 0.214 + self.trashold,   # (původně 0.214)
            "left_hand_thumb_2_joint": 0.357 + self.trashold,   # (původně 0.357)
            # Ostatní prsty se zavírají do MÍNUSU
            "left_hand_middle_0_joint": -0.523 - self.trashold, # (původně -0.523)
            "left_hand_middle_1_joint": -0.527 - self.trashold, # (původně -0.527)
            "left_hand_index_0_joint": -0.485 - self.trashold,  # (původně -0.485)
            "left_hand_index_1_joint": -0.542 - self.trashold,  # (původně -0.542)

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
    Vektorizovaná odměna za generování síly do prstů.
    Využívá 100% PyTorch tenzorové operace.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [2,3]

        self.finger_categories = {
            "left_hand_thumb": 0, "left_hand_index": 1, "left_hand_middle": 2,
            "right_hand_thumb": 3, "right_hand_index": 4, "right_hand_middle": 5
        }
        self.force_threshold = 1.0

        # Cached GPU tensors
        self.base_idx_to_finger_cat = None
        self.chair_ids = None

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is not None:
            stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
            if not stage_mask.any(): return torch.zeros(num_envs, device=device)
        else:
            stage_mask = torch.ones(num_envs, device=device, dtype=torch.bool)

        contact_data = robot.contact
        if contact_data is None:
            return torch.zeros(num_envs, device=device)

        # 1. JEDNORÁZOVÁ INICIALIZACE INDEXŮ
        if self.base_idx_to_finger_cat is None:
            global_map = states.extras.get("global_link_map", {})
            num_bodies = states.extras.get("num_bodies_per_env", 1000)

            idx_to_cat = torch.full((num_bodies,), -1, dtype=torch.long, device=device)
            chair_ids = []

            for idx, (o_name, l_name) in global_map.items():
                if o_name == robot_name:
                    for cat_name, cat_id in self.finger_categories.items():
                        if cat_name in l_name:
                            idx_to_cat[idx] = cat_id
                elif o_name == "chair":
                    chair_ids.append(idx)

            self.base_idx_to_finger_cat = idx_to_cat
            self.chair_ids = torch.tensor(chair_ids, device=device)
            self.num_bodies = num_bodies

        # 2. RYCHLÉ TENZOROVÉ OPERACE
        link_a = contact_data['link_a'] # [num_envs, max_contacts]

        # --- OPRAVA: Kontrola, zda existují vůbec nějaké kontakty ---
        # Pokud je max_contacts == 0, okamžitě vracíme nuly, abychom
        # zabránili pádu funkce torch.max() o pár řádků níže.
        if link_a.shape[1] == 0:
            return torch.zeros(num_envs, device=device)
        # -------------------------------------------------------------

        link_b = contact_data['link_b']
        valid_mask = contact_data['valid_mask']

        forces = contact_data.get('force_b', contact_data.get('force', None))
        if forces is None:
            forces = torch.zeros((*link_a.shape, 3), device=device)

        force_mags = torch.norm(forces, dim=-1) # [num_envs, max_contacts]

        base_a = link_a % self.num_bodies
        base_b = link_b % self.num_bodies

        a_is_chair = torch.isin(base_a, self.chair_ids)
        b_is_chair = torch.isin(base_b, self.chair_ids)

        cat_a = self.base_idx_to_finger_cat[base_a]
        cat_b = self.base_idx_to_finger_cat[base_b]

        contact_cat = torch.where(b_is_chair, cat_a, torch.where(a_is_chair, cat_b, torch.tensor(-1, device=device)))
        valid_interaction = (contact_cat >= 0) & valid_mask

        finger_forces = torch.zeros((num_envs, len(self.finger_categories)), device=device)

        for cat_id in range(len(self.finger_categories)):
            cat_mask = valid_interaction & (contact_cat == cat_id)
            cat_forces = force_mags * cat_mask.float()

            # Bez "early exitu" výše by tento řádek spadnul na prázdných tenzorech
            max_f, _ = torch.max(cat_forces, dim=1)

            finger_forces[:, cat_id] = max_f

        # 3. VÝPOČET ODMĚNY
        finger_rewards = torch.clamp(finger_forces / self.force_threshold, max=1.0)
        reward = torch.mean(finger_rewards, dim=1)

        return reward * stage_mask.float()
#---------------------Stage 3----------------------
class PullChairDistanceReward(HumanoidBaseReward):
    """
    Stage 3: Pull Chair Distance
    Odměňuje robota za to, že se židle blíží k cílové pozici (1 metr dozadu).
    Dense reward pomocí Gaussovy funkce.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [3]
        self.sigma = 0.4  # Tolerance

        # Výchozí pozice židle je [0.75, 0.0, 0.1].
        # O 1 metr dozadu v ose X to znamená [-0.25, 0.0, 0.1].
        self.target_chair_pos = torch.tensor([-0.25, 0.0, 0.1])

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None: return torch.zeros(num_envs, device=device)
        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any(): return torch.zeros(num_envs, device=device)

        chair_base_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_base_idx, :3]

        target_pos = self.target_chair_pos.to(device)

        # Spočítáme chybu - jak daleko je židle od cíle
        dist_sq = torch.sum(torch.square(chair_pos - target_pos), dim=-1)

        # Gaussovská odměna
        reward = torch.exp(-dist_sq / (2 * self.sigma**2))

        return reward * stage_mask.float()
class PullRobotVelocityReward(HumanoidBaseReward):
    """
    Stage 3: Pull Velocity & Smooth Braking
    1. Motivuje robota couvat maximální rychlostí např. -0.5 m/s.
    2. V posledních centimetrech před cílem (braking_distance) začne cílová rychlost plynule klesat k 0.
    3. V cíli (target_distance) je cílová rychlost přesně 0.0 m/s.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [3]
        self.sigma = 0.3

        self.pull_speed = 0.5          # Jak rychle má robot maximálně couvat (m/s)
        self.target_distance = 1.0     # Cílová vzdálenost, kde už má stát
        self.braking_distance = 0.4    # Posledních 30 cm před cílem začne plynule brzdit

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None: return torch.zeros(num_envs, device=device)
        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any(): return torch.zeros(num_envs, device=device)

        # 1. Získání lineární rychlosti robota
        base_idx = robot.body_names.index("pelvis")
        root_vel = robot.body_state[:, base_idx, 7:10]

        # 2. Získání vzdálenosti, o kterou se židle už posunula
        chair_base_idx = chair.body_names.index("base_link")
        chair_pos = chair.body_state[:, chair_base_idx, :3]
        initial_chair_pos = torch.tensor([0.75, 0.0, 0.1], device=device)

        moved_dist = torch.norm(chair_pos - initial_chair_pos, dim=-1)

        # 3. VÝPOČET PLYNULÉHO BRZDĚNÍ
        # Kolik metrů ještě zbývá do cíle?
        dist_remaining = self.target_distance - moved_dist

        # Vypočítáme faktor rychlosti od 0.0 do 1.0
        # - Pokud zbývá více než 0.3m -> faktor je 1.0 (plná rychlost)
        # - Pokud zbývá 0.15m -> faktor je 0.5 (poloviční rychlost)
        # - Pokud už je v cíli (zbývá <= 0) -> faktor je 0.0 (stojí)
        speed_factor = torch.clamp(dist_remaining / self.braking_distance, min=0.0, max=1.0)

        # 4. Aplikace cílové rychlosti
        target_vel = torch.zeros_like(root_vel)

        # Osa X je u vás couvání (proto mínus). Rychlost škálujeme naším faktorem.
        target_vel[:, 0] = -self.pull_speed * speed_factor

        # 5. Výpočet Gaussovské odměny za sledování této dynamické rychlosti
        vel_error_sq = torch.sum(torch.square(root_vel - target_vel), dim=-1)
        reward = torch.exp(-vel_error_sq / (2 * self.sigma**2))

        return reward * stage_mask.float()
#---------------------Stage 4----------------------
class KeepChairStillPenalty(HumanoidBaseReward):
    """
    Stage 4: Keep Chair Still
    Penalizuje jakýkoliv pohyb židle (lineární i úhlovou rychlost).
    Zajišťuje, že robot pustí židli jemně a neodhodí ji.
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [0,1,3,5]

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        chair = states.objects["chair"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        if self.actual_stage is None: return torch.zeros(num_envs, device=device)
        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any(): return torch.zeros(num_envs, device=device)

        # Rychlosti židle (root_state obsahuje pos[0:3], quat[3:7], lin_vel[7:10], ang_vel[10:13])
        base_link_idx = chair.body_names.index("base_link")

        chair_lin_vel = chair.body_state[:, base_link_idx, 7:10]
        chair_ang_vel = chair.body_state[:, base_link_idx, 10:13]

        # Výpočet kvadratické odchylky od nuly (čím rychleji letí, tím větší trest)
        lin_vel_sq = torch.sum(torch.square(chair_lin_vel), dim=-1)
        ang_vel_sq = torch.sum(torch.square(chair_ang_vel), dim=-1)

        # Sečteme penalty (úhlovou rychlost penalizujeme trochu méně,
        # protože mírné zhoupnutí při puštění je fyzikálně přirozené)
        total_penalty = lin_vel_sq + 0.5 * ang_vel_sq

        return total_penalty * stage_mask.float()
#---------------------Stage 5----------------------
class DropArmsReward(HumanoidBaseReward):
    """
    Stage 5: Drop Arms Reward
    Odměňuje robota (Gaussian reward) za to, že stahuje ramena a lokty k nule
    (tj. spouští paže volně podél těla).
    """
    def __init__(self, robot_name="g1_slider"):
        super().__init__(robot_name)
        self.active_stages = [5]
        self.sigma = 0.5  # Tolerance pro Gaussovu křivku

        self.arm_indices = None

        # Sledujeme ty samé klouby jako v Checkeru pro Stage 5
        self.arm_joints_to_track = [
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint", "right_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
            "left_elbow_joint", "right_elbow_joint"
        ]

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        # 1. Kontrola Stage
        if self.actual_stage is None: return torch.zeros(num_envs, device=device)
        stage_mask = torch.isin(self.actual_stage, torch.tensor(self.active_stages, device=device))
        if not stage_mask.any(): return torch.zeros(num_envs, device=device)

        # 2. Inicializace (pouze poprvé)
        if self.arm_indices is None:
            indices = []
            for name in self.arm_joints_to_track:
                if name in robot.joint_names:
                    indices.append(list(robot.joint_names).index(name))

            if not indices:
                return torch.zeros(num_envs, device=device)

            self.arm_indices = torch.tensor(indices, device=device, dtype=torch.long)

        # 3. Získání pozic paží
        q_arms = robot.joint_pos[:, self.arm_indices]

        # 4. Výpočet Gaussianské odměny (Cílová póza je 0.0 pro všechny tyto klouby)
        # exp(-||q_arms - 0||^2 / 2sigma^2)
        pos_error_sq = torch.sum(torch.square(q_arms), dim=-1)
        reward = torch.exp(-pos_error_sq / (2 * self.sigma**2))

        return reward * stage_mask.float()


TERMINATION_WEIGHT = -1000.0
DELTA_ACTION_RATE_WEIGHT = -0.01
DOF_VELOCITY_ACCELERATION_WEIGHT = 1.0
DOF_POSITION_LIMITS_WEIGHT = -5.0
HUMANLY_DOF_LIMIT_WEIGHT = -1.0
UPRIGHT_PENALTY_WEIGHT = -1.0
#STAGE_PROGRESS_WEIGHT = 4.0
CONTINUOUS_REWARD_WEIGHT= 1.0

#stage 0
WALK_TO_CHAIR_REWARD_WEIGHT = 3.5
FACE_CHAIR_REWARD_WEIGHT = 1.0
#stage 1
REACH_CHAIR_REWARD_WEIGHT = 2.5
REACH_ORIENTATION_REWARD_WEIGHT = 1.5
STAND_STILL_PENALTY_WEIGHT = -1.0
OPEN_GRASP_REWARD_WEIGHT = 1.0
#stage 2
CLOSE_GRASP_REWARD_WEIGHT = 2.5
FORCE_GRASP_REWARD_WEIGHT = 1.0

PULL_CHAIR_DISTANCE_WEIGHT = 5.0
PULL_ROBOT_VELOCITY_WEIGHT = 4.0
KEEP_CHAIR_STILL_PENALTY_WEIGHT = -1.0
ARM_RESTING_POSE_PENALTY_WEIGHT = -0.01

@configclass
class ChairmanCfg(HumanoidTaskCfg):
    """Chair task for humanoid robots."""




    success_bar = 0.9
    episode_length = 800
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
        #STAGE_PROGRESS_WEIGHT,
        WALK_TO_CHAIR_REWARD_WEIGHT,
        #FACE_CHAIR_REWARD_WEIGHT,
        REACH_CHAIR_REWARD_WEIGHT,
        REACH_ORIENTATION_REWARD_WEIGHT,
        #STAND_STILL_PENALTY_WEIGHT,
        OPEN_GRASP_REWARD_WEIGHT,
        CLOSE_GRASP_REWARD_WEIGHT,
        FORCE_GRASP_REWARD_WEIGHT,
        PULL_CHAIR_DISTANCE_WEIGHT,
        PULL_ROBOT_VELOCITY_WEIGHT,
        KEEP_CHAIR_STILL_PENALTY_WEIGHT,
        ARM_RESTING_POSE_PENALTY_WEIGHT,
        #CONTINUOUS_REWARD_WEIGHT
    ]
    #function_index_success_save_time = 10 #TODO hloupé řešení ale budiž to tak (potřeba opravit)
    reward_functions = [TerminationCfg(),
                        DeltaActionRateCfg(),
                        DoFVelocityAccelerationCfg(),
                        DofPositionLimitsCfg(),
                        HumanlyDofLimitCfg(),
                        UprightPenaltyCfg(),
                        #StageProgressCfg(),
                        WalkToChairReward(),
                        #FaceChairReward(),
                        ReachChairReward(),
                        HandOrientationReward(),
                        #StandStillPenalty(),
                        OpenGraspReward(),
                        CloseGraspReward(),
                        GraspForceReward(),
                        PullChairDistanceReward(),
                        PullRobotVelocityReward(),
                        KeepChairStillPenalty(),
                        ArmRestingPosePenaltyCfg(),
                        #ContinuousStageReward()
                        ]
    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}

"""Exit Door task for humanoid robots."""

from __future__ import annotations
from xml.sax import handler

import torch

from metasim.cfg.checkers import _DoorManChecker
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

        return total_penalty

class DofPositionLimitsCfg(HumanoidBaseReward):
    """Penalty for exceeding joint position limits."""
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.joint_limits: dict[str, tuple[float, float]] = {
        "left_hip_pitch_joint": (-2.5307, 2.8798),
        "left_hip_roll_joint": (-0.5236, 2.9671),
        "left_hip_yaw_joint": (-2.7576, 2.7576),
        "left_knee_joint": (-0.087267, 2.8798),
        "left_ankle_pitch_joint": (-0.87267, 0.5236),
        "left_ankle_roll_joint": (-0.2618, 0.2618),
        "right_hip_pitch_joint": (-2.5307, 2.8798),
        "right_hip_roll_joint": (-2.9671, 0.5236),
        "right_hip_yaw_joint": (-2.7576, 2.7576),
        "right_knee_joint": (-0.087267, 2.8798),
        "right_ankle_pitch_joint": (-0.87267, 0.5236),
        "right_ankle_roll_joint": (-0.2618, 0.2618),
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
        # Left hand fingers
        "left_hand_thumb_0_joint": (-1.04719755, 1.04719755),
        "left_hand_thumb_1_joint": (-0.72431163, 1.04719755),
        "left_hand_thumb_2_joint": (0.0, 1.74532925),
        "left_hand_middle_0_joint": (-1.57079632, 0.0),
        "left_hand_middle_1_joint": (-1.74532925, 0.0),
        "left_hand_index_0_joint": (-1.57079632, 0.0),
        "left_hand_index_1_joint": (-1.74532925, 0.0),
        # Right hand fingers
        "right_hand_thumb_0_joint": (-1.04719755, 1.04719755),
        "right_hand_thumb_1_joint": (-1.04719755, 0.72431163),
        "right_hand_thumb_2_joint": (-1.74532925, 0.0),
        "right_hand_middle_0_joint": (0.0, 1.57079632),
        "right_hand_middle_1_joint": (0.0, 1.74532925),
        "right_hand_index_0_joint": (0.0, 1.57079632),
        "right_hand_index_1_joint": (0.0, 1.74532925),
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
                raise ValueError(f"Joint {name} limits not defined.")
            low, high = self.joint_limits[name]
            low_tensor = torch.tensor(low, device=device)
            high_tensor = torch.tensor(high, device=device)
            # Penalizuj překročení limitů
            below_low = torch.relu((low_tensor + self.limit_buffer) - joint_pos[:, i])
            above_high = torch.relu(joint_pos[:, i] - (high_tensor - self.limit_buffer))
            penalty = below_low + above_high
            if i == 0:
                total_penalty = penalty
            else:
                total_penalty += penalty
        return total_penalty

class FingerPrimitiveLimitsCfg(HumanoidBaseReward):
    """
    Finger primitive limits: Udržuje prsty v 'lidsky přirozeném' rozsahu.
    Podle DoorMan paperu (Table 2) je váha -1.0.

    Limity jsou zde nastaveny přísněji než mechanické limity URDF,
    aby simulovaly přirozený rozsah pohybu a zabránily nepřirozeným polohám.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.finger_indices = None
        # Cache pro limity
        self.q_lower_tensor = None
        self.q_upper_tensor = None

        # Buffer nula, protože limity už jsou "měkké" (přirozené), nikoliv tvrdé mechanické
        self.limit_buffer = 0.0

        # Definice PŘIROZENÝCH limitů (užší než mechanické)
        # Změny oproti mechanice:
        # 1. Palec (Thumb): Výrazné zúžení rotace (joint 0) a flexe, aby se držel v opozici.
        # 2. Prsty (Index/Middle): Omezení maximální flexe (zavření), aby nedocházelo k 'drcení'.
        #    Extenze (0.0) ponechána pro možnost pustit kliku.
        self.finger_limits: dict[str, tuple[float, float]] = {
            # --- LEVÁ RUKA (Left Hand) ---
            # Palec: Omezíme rotaci (0_joint), aby palec neutíkal do stran
            "left_hand_thumb_0_joint": (-0.7, 0.7),      # Mechanicky: +/- 1.04
            "left_hand_thumb_1_joint": (-0.5, 0.8),      # Mechanicky: -0.7 až 1.0
            "left_hand_thumb_2_joint": (0.0, 1.4),       # Mechanicky: 0 až 1.74 (zúženo zavírání)

            # Prostředníček: Negativní hodnoty znamenají zavírání (flexi) u levé ruky G1 (dle vašeho URDF)
            "left_hand_middle_0_joint": (-1.4, 0.0),     # Mechanicky: -1.57 až 0 (zúženo zavírání)
            "left_hand_middle_1_joint": (-1.5, 0.0),     # Mechanicky: -1.74 až 0 (zúženo zavírání)

            # Ukazováček
            "left_hand_index_0_joint": (-1.4, 0.0),      # Mechanicky: -1.57 až 0
            "left_hand_index_1_joint": (-1.5, 0.0),      # Mechanicky: -1.74 až 0

            "right_hand_thumb_0_joint": (-0.7, 0.7),     # Zúžená rotace
            "right_hand_thumb_1_joint": (-0.8, 0.5),     # Symetricky upraveno dle levé (ale pozor na znaménka v URDF)
            "right_hand_thumb_2_joint": (-1.4, 0.0),     # Mechanicky: -1.74 až 0

            # Prostředníček: Pozitivní hodnoty pro zavírání? (dle vašeho zadání (0.0, 1.57))
            "right_hand_middle_0_joint": (0.0, 1.4),     # Mechanicky: 0 až 1.57
            "right_hand_middle_1_joint": (0.0, 1.5),     # Mechanicky: 0 až 1.74

            # Ukazováček
            "right_hand_index_0_joint": (0.0, 1.4),      # Mechanicky: 0 až 1.57
            "right_hand_index_1_joint": (0.0, 1.5),      # Mechanicky: 0 až 1.74
        }

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        """Compute penalty for exceeding finger joint limits."""
        robot = states.robots[robot_name]
        joint_pos = robot.joint_pos
        device = joint_pos.device

        # 1. Inicializace (provede se pouze poprvé)
        if self.finger_indices is None:
            self.finger_indices = []
            lower_vals = []
            upper_vals = []

            for i, name in enumerate(robot.joint_names):
                # Hledáme pouze klouby definované v 'finger_limits'
                if name in self.finger_limits:
                    self.finger_indices.append(i)
                    limits = self.finger_limits[name]
                    # Zde již nepřidáváme buffer, protože samotné limity jsou "bufferem" oproti mechanice
                    lower_vals.append(limits[0])
                    upper_vals.append(limits[1])

            # Konverze na tenzory a uložení do cache
            self.finger_indices = torch.tensor(self.finger_indices, device=device, dtype=torch.long)
            # Tvar (1, num_fingers) pro broadcasting
            self.q_lower_tensor = torch.tensor(lower_vals, device=device).unsqueeze(0)
            self.q_upper_tensor = torch.tensor(upper_vals, device=device).unsqueeze(0)

        # Pojistka
        if len(self.finger_indices) == 0:
            return torch.zeros(joint_pos.shape[0], device=device)

        # 2. Získání aktuálních pozic prstů
        q_finger = joint_pos[:, self.finger_indices]

        # 3. Výpočet penalizace (Linear hinge loss)
        # Penalizujeme vše, co je mimo náš "přirozený" interval
        violation_lower = torch.relu(self.q_lower_tensor - q_finger)
        violation_upper = torch.relu(q_finger - self.q_upper_tensor)

        penalty = torch.sum(violation_lower + violation_upper, dim=-1)

        return penalty

class HumanlyDofLimitCfg(HumanoidBaseReward):
    """
    Humanly DoF limit: Penalizace za překročení 'lidsky přirozených' limitů.
    Váha dle paperu: -1.0

    Tato funkce nahrazuje mechanické limity robota (které jsou často příliš volné)
    přísnějšími limity, které odpovídají rozsahu pohybu člověka.

    Vzorec: sum( ( clip(q - q_lower, max=0) + clip(q - q_upper, min=0) )^2 )
    """
    def __init__(self, robot_name="g1_with_hands"):
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

        return penalty

class UndesiredContactCfg(HumanoidBaseReward):
    """
    Undesired contact: Penalizace za nežádoucí kolize s dveřmi.

    Pravidla:
    1. Ruka (hand/finger/...) se smí dotýkat POUZE kliky (handle/knob/bar).
       Dotyk ruky s rámem nebo panelem je trestán.
    2. Ostatní části těla se nesmí dotýkat NIČEHO na dveřích.

    Vrací 1.0 při prvním porušení pravidel v daném kroku (binární penalizace).
    """
    def __init__(self, robot_name="g1_with_hands", door_name="door"):
        super().__init__(robot_name)
        self.door_name = door_name

        # Části robota, které se považují za ruku
        self.hand_keywords = ["hand", "thumb", "palm", "wrist", "endeffector"]

        # Části dveří, které se považují za kliku (povolený cíl pro ruku)
        self.handle_keywords = ["door_handle", "door_handle_stem"]

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]

        # Inicializace penalizací
        num_envs = robot.joint_pos.shape[0]
        device = robot.joint_pos.device
        penalty = torch.zeros(num_envs, device=device)

        # Načtení kontaktů
        contacts = robot.contact
        if not contacts:
            return penalty

        # Získání seznamu všech linků dveří pro ověření
        if self.door_name in states.objects:
            door_body_names = states.objects[self.door_name].body_names
        else:
            door_body_names = []

        for c in contacts:
            env_id = c["env_id"]

            # OPTIMALIZACE: Pokud už tento env má trest, neztrácíme čas další kontrolou
            if penalty[env_id] == 1.0:
                continue

            # Rozlišení těl v kontaktu
            name_a = c["body_a"]
            link_a = c["link_a"]
            name_b = c["body_b"]
            link_b = c["link_b"]

            # Zjištění, která část je robot a která dveře
            robot_link = None
            door_link = None

            if name_a == robot_name and name_b == self.door_name:
                robot_link = link_a
                door_link = link_b
            elif name_b == robot_name and name_a == self.door_name:
                robot_link = link_b
                door_link = link_a

            # Pokud kolize není s dveřmi, ignorujeme (např. robot-země)
            if robot_link is None:
                continue

            # Ověření, že door_link skutečně patří dveřím (pro jistotu)
            # Tím zachytíme rám, panel, kliku, panty atd.
            is_door_contact = any(k in door_link.lower() for k in door_body_names)

            if is_door_contact:
                # 1. Je to ruka?
                is_hand = any(k in robot_link.lower() for k in self.hand_keywords)

                if is_hand:
                    # Ruka se smí dotýkat POUZE kliky
                    is_handle = any(k in door_link.lower() for k in self.handle_keywords)
                    if not is_handle:
                        # Ruka se dotkla něčeho jiného než kliky (např. panelu) -> TREST
                        penalty[env_id] = 1.0
                else:
                    # Není to ruka (je to tělo/noha/hlava) -> nesmí se dotknout ničeho na dveřích -> TREST
                    penalty[env_id] = 1.0

        return penalty
class DoorContactForceCfg(HumanoidBaseReward):
    def __init__(self, robot_name="g1_with_hands", door_name="door"):
        super().__init__(robot_name)
        self.door_name = door_name

        # Části robota, které se považují za ruku
        self.hand_keywords = ["hand", "thumb", "palm", "wrist", "endeffector"]

        # Části dveří, které se považují za kliku (povolený cíl pro ruku)
        self.handle_keywords = ["door_handle", "door_handle_stem"]

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]

        # Inicializace penalizací (suma sil)
        num_envs = robot.joint_pos.shape[0]
        device = robot.joint_pos.device
        penalty = torch.zeros(num_envs, device=device)

        # Načtení kontaktů
        contacts = robot.contact
        if not contacts:
            return penalty

        # Získání seznamu všech linků dveří pro ověření
        if self.door_name in states.objects:
            door_body_names = states.objects[self.door_name].body_names
        else:
            door_body_names = []

        for c in contacts:
            env_id = c["env_id"]

            # Poznámka: Zde NEPOUŽÍVÁME optimalizaci 'continue', protože chceme sečíst
            # síly všech špatných kontaktů (např. loket naráží do rámu + koleno do panelu).

            # Rozlišení těl v kontaktu
            name_a = c["body_a"]
            link_a = c["link_a"]
            name_b = c["body_b"]
            link_b = c["link_b"]

            # Zjištění, která část je robot a která dveře
            robot_link = None
            door_link = None

            if name_a == robot_name and name_b == self.door_name:
                robot_link = link_a
                door_link = link_b
            elif name_b == robot_name and name_a == self.door_name:
                robot_link = link_b
                door_link = link_a

            # Pokud kolize není s dveřmi, ignorujeme
            if robot_link is None:
                continue

            # Ověření, že door_link skutečně patří dveřím
            is_door_contact = any(k in door_link.lower() for k in door_body_names)

            if is_door_contact:
                # Získáme velikost síly z kontaktu
                # Předpokládáme, že vaše funkce get_contact() vrací ve slovníku klíč "force"
                # Pokud tam není, použijeme 0.0 (nebo fallback hodnotu)
                contact_force = c.get("force", 0.0)

                # 1. Je to ruka?
                is_hand = any(k in robot_link.lower() for k in self.hand_keywords)

                if is_hand:
                    # Ruka se smí dotýkat POUZE kliky
                    is_handle = any(k in door_link.lower() for k in self.handle_keywords)
                    if not is_handle:
                        # Ruka se dotkla něčeho jiného než kliky (např. panelu) -> TREST SÍLOU
                        penalty[env_id] += contact_force
                else:
                    # Není to ruka (tělo) -> nesmí se dotknout ničeho -> TREST SÍLOU
                    penalty[env_id] += contact_force

        return penalty

class UprightPenaltyCfg(HumanoidBaseReward):
    """
    Upright penalty: Nutí robota držet trup svisle (osa Z).
    Podle DoorMan paperu (Table 2) je váha -1.0.

    Vzorec: || R_torso * [0, 0, 1]^T - [0, 0, 1]^T ||^2
    """
    def __init__(self, robot_name="g1_with_hands"):
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

        return penalty

class StageProgressCfg(HumanoidBaseReward):
    """
    Stage progress: Odměna za aktuální dosažený stage.
    Podle DoorMan paperu (Table 2) je váha 1.0.

    Formula: stage_current
    Funguje jako dense reward, který motivuje robota zůstat ve vyšších fázích.
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        # Pokud není actual_stage inicializováno, vrátíme 0
        if self.completed_stages.any():
            ret = self.completed_stages * self.actual_stage.float()
            return ret
        else:
            return torch.zeros_like(self.completed_stages)




class SuccessSaveTimeCfg(HumanoidBaseReward):
    """
    Success save time: Bonus za rychlost.
    Podle DoorMan paperu (Table 2) je váha 0.5.

    Formula: 1_{success} * (remaining_time / episode_length)
    """
    def __init__(self, robot_name="g1_with_hands", episode_length=400):
        super().__init__(robot_name)
        self.episode_length = episode_length
        self.current_env_steps = None

    def reset_steps(self, env_ids: torch.Tensor):
        """Volat z reset_doorman při resetu (fail nebo timeout)."""
        if self.current_env_steps is not None:
            self.current_env_steps[env_ids] = 0

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        # Inicializace
        if self.current_env_steps is None:
            self.current_env_steps = torch.zeros(num_envs, device=device)

        # Inkrementace času
        self.current_env_steps += 1

        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        # Úspěch je definován jako dosažení stage 6
        is_success = (self.actual_stage == 6)

        # Výpočet poměru zbývajícího času
        # clamp( (400 - steps) / 400, 0 )
        remaining_ratio = torch.clamp(
            (self.episode_length - self.current_env_steps) / self.episode_length,
            min=0.0
        )

        # Aplikace odměny (jen pro úspěšné)
        reward = is_success.float() * remaining_ratio

        # DŮLEŽITÉ: Protože stage 6 vyvolá reset prostředí hned v dalším kroku checkeru,
        # musíme si zde resetovat počítadlo kroků pro ty, co uspěli.
        if is_success.any():
            self.current_env_steps[is_success] = 0

        return reward
class TaskCompleted(HumanoidBaseReward):
    """
    Task completed: Binární odměna za dokončení úkolu.
    Podle DoorMan paperu (Table 2) je váha 1.0.

    Formula: 1_{success}
    Kde success je True, pokud robot dosáhl poslední stage (6).
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        if self.actual_stage is None:
            device = states.robots[robot_name].joint_pos.device
            return torch.zeros(states.robots[robot_name].joint_pos.shape[0], device=device)

        # Vektorizovaný výpočet: Kde je stage 6, tam je 1.0, jinak 0.0
        return (self.actual_stage == 6).float()

#-----------------------------------------STAGE 0-------------------------------------------------
class WalkToDoorReward(HumanoidBaseReward):
    """
    Stage 0: Walk to door
    Gaussian odměna za minimalizaci vzdálenosti a směru k cíli (velocity tracking).
    Aplikuje se POUZE ve Stage 0.

    Formula: exp(-||v_robot - v_target * d_door||^2 / (2 * sigma^2))
    [cite_start]Sigma: 0.15 [cite: 8, 507]
    [cite_start]Weight: 5.0 [cite: 507]
    """
    def __init__(self, robot_name="g1_with_hands", target_speed=0.6):
        super().__init__(robot_name)
        self.sigma = 0.15
        # Cílová rychlost chůze (v m/s)
        self.target_speed = target_speed

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        door = states.objects["door"]
        device = robot.joint_pos.device
        num_envs = robot.joint_pos.shape[0]

        # 1. Kontrola Stage: Pokud actual_stage není definováno, vracíme 0
        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        # 2. Vytvoření masky pro Stage 0
        # Odměnu počítáme jen pro ty, kteří jsou ve Stage 0
        stage_mask = (self.actual_stage == 0)

        # Optimalizace: Pokud nikdo není ve Stage 0, vrátíme nuly rovnou
        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        # 3. Získání pozic a rychlostí
        root_pos = robot.root_state[:, :3]
        root_vel = robot.root_state[:, 7:10] # Lineární rychlost
        target_pos = door.root_state[:, :3] # Pozice dveří

        # 4. Výpočet směrového vektoru d_door (normalized)
        vec_to_door = target_pos - root_pos
        vec_to_door[:, 2] = 0.0 # Ignorujeme Z složku (chůze po rovině)

        dist = torch.norm(vec_to_door, dim=-1, keepdim=True)
        dir_to_door = vec_to_door / (dist + 1e-6)

        # 5. Cílový vektor rychlosti (v_target * d_door)
        target_vel_vec = self.target_speed * dir_to_door

        # 6. Výpočet chyby rychlosti: ||v_robot - v_target_vec||^2
        vel_error_sq = torch.sum(torch.square(root_vel - target_vel_vec), dim=-1)

        # 7. Gaussian Reward
        reward = torch.exp(-vel_error_sq / (2 * self.sigma**2))

        # 8. Aplikace masky (vynulování odměny pro ty, co nejsou ve Stage 0)
        reward = reward * stage_mask.float()

        return reward
#TODO zkontrolovat
class UpperBodyDeviationReward(HumanoidBaseReward):
    """
    Upper body deviation: Penalizace za odchylku horní části těla od klidového stavu.
    Aktivní ve Stages: 0, 5.

    Formula: ||q_upper, non-finger - q_resting||_1
    Weight: -1.0
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        self.upper_body_indices = None
        self.q_resting_tensor = None

        # Stages, ve kterých je tato penalizace aktivní
        self.active_stages = [0, 5]

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        joint_pos = robot.joint_pos
        device = joint_pos.device
        num_envs = joint_pos.shape[0]

        # 1. Kontrola Stage
        if self.actual_stage is None:
            return torch.zeros(num_envs, device=device)

        # Maska: True pokud je actual_stage v seznamu [0, 5]
        # Vytvoříme tensor aktivních stage pro porovnání
        active_stages_tensor = torch.tensor(self.active_stages, device=device)
        # isin vrátí True tam, kde je actual_stage přítomen v active_stages
        stage_mask = torch.isin(self.actual_stage, active_stages_tensor)

        if not stage_mask.any():
            return torch.zeros(num_envs, device=device)

        # 2. Inicializace (pouze poprvé)
        if self.upper_body_indices is None:
            self.upper_body_indices = []
            resting_vals = []
            upper_keywords = ["waist", "shoulder", "elbow", "wrist"]

            for i, name in enumerate(robot.joint_names):
                is_upper = any(k in name for k in upper_keywords)
                is_finger = "hand" in name or "finger" in name

                if is_upper and not is_finger:
                    self.upper_body_indices.append(i)
                    # Načtení klidové pozice
                    val = 0.0
                    if hasattr(self, 'initial_pos') and name in self.initial_pos:
                        val = self.initial_pos[name]
                    resting_vals.append(val)

            self.upper_body_indices = torch.tensor(self.upper_body_indices, device=device, dtype=torch.long)
            self.q_resting_tensor = torch.tensor(resting_vals, device=device).unsqueeze(0)

        if len(self.upper_body_indices) == 0:
            return torch.zeros(num_envs, device=device)

        # 3. Výpočet odchylky
        q_upper = joint_pos[:, self.upper_body_indices]
        deviation = torch.sum(torch.abs(q_upper - self.q_resting_tensor), dim=-1)

        # 4. Aplikace masky (vynulovat penalizaci pro neaktivní stage)
        return deviation * stage_mask.float()
#TODO zkontrolovat
class FaceDoorReward(HumanoidBaseReward):
    """
    Face door: Penalizace za špatnou orientaci (Yaw) vůči dveřím.
    Aktivní ve Stages: 0, 1, 2, 5.

    Interpretace: Robot musí srovnat své natočení (Yaw) s natočením rámu dveří.
    To zajistí, že ve Stage 0 jde kolmo ke dveřím a ve Stage 5 pokračuje rovně skrz ně
    (neotáčí se zpět na dveře).

    Formula: |wrap_pi( ||axis-angle(R_door)||_2 )|
    Weight: -1.0
    """
    def __init__(self, robot_name="g1_with_hands"):
        super().__init__(robot_name)
        # Stages: 0-2 (příchod, úchop) a 5 (průchod)
        self.active_stages = [0, 1, 2, 5]

    def _wrap_to_pi(self, angle):
        """Převede úhel do intervalu [-pi, pi]."""
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi

    def __call__(self, states: list[EnvState], robot_name: str = None) -> torch.FloatTensor:
        robot = states.robots[robot_name]
        door = states.objects["door"]
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

        # 3. Získání Yaw (natočení) dveří
        # Předpokládáme, že "door" objekt reprezentuje rám (frame), který se nehýbe
        q_d = door.root_state[:, 3:7]
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
        return penalty * stage_mask.float()
#-----------------------------------------STAGE 1-------------------------------------------------

@configclass
class DoorStandCfg(HumanoidTaskCfg):
    """Door task for humanoid robots."""
    success_bar = 0.9
    episode_length = 400
    objects = [
        ArticulationObjCfg(
            name="door",
            urdf_path="roboverse_data/assets/humanoidbench/door/door.urdf",
            default_position= [0.0, 0.0, 0.0],
            fix_base_link=True,
            colapse_fixed_joints=False
        )
    ]
    traj_filepath = "roboverse_data/trajs/humanoidbench/door/initial_state_v2.json"
    checker = _DoorManChecker()
    reward_weights = [-1000.0, -0.01, 1.0,-5.0, -1.0, -1.0,-0.2, -0.1, -1.0, 1.0, 0.5,4.0,5.0,-1.0,-1.0]
    function_index_success_save_time = 10 #TODO hloupé řešení ale budiž to tak (potřeba opravit)
    reward_functions = [TerminationCfg(),
                        DeltaActionRateCfg(),
                        DoFVelocityAccelerationCfg(),
                        DofPositionLimitsCfg(),
                        FingerPrimitiveLimitsCfg(),
                        HumanlyDofLimitCfg(),
                        UndesiredContactCfg(),
                        DoorContactForceCfg(),
                        UprightPenaltyCfg(),
                        StageProgressCfg(),
                        SuccessSaveTimeCfg(),
                        TaskCompleted(),
                        WalkToDoorReward(),
                        UpperBodyDeviationReward(),
                        FaceDoorReward()
                        ]
    def extra_spec(self):
        """This task does not require any extra observations."""
        return {}

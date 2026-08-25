import pickle
import os
import random
from time import time
from pathlib import Path
import torch
import threading

from metasim.utils.humanoid_robot_util import (
    neck_height_tensor,
    right_palm_position,
    right_palm_orientation,
    robot_position_tensor,
    door_angle_tensor,
)
from metasim.types import EnvState
from metasim.utils.chair_navigation import (
    CHAIR_FINAL_DISTANCE,
    CHAIR_FINAL_TOLERANCE,
    chair_back_direction_xy,
    forward_direction_xy,
)
try:
    from metasim.sim import BaseSimHandler
except:
    pass

STAGE_TIMEOUTS = {
    # Reference step counts at the original 50 Hz task-control rate.  The
    # checker converts them to the active dt below, so changing simulation
    # decimation no longer halves/doubles the physical time available.
    0: 400,  # Dojít k židli (8 s)
    1: 500,  # Reach + orientace + ustálení obou rukou (10 s)
    2: 400,  # Postupné zavření všech prstů a vytvoření kontaktů (8 s)
    3: 500,  # Zatažení za židli
    4: 100,  # Zastavení židle
    5: 100   # Svěšení rukou
}
STAGE_TIMEOUT_REFERENCE_DT = 0.02
VELOCITY_THRESHOLD = 0.2
HEIGHT_THRESHOLD = 0.4
FACING_CHAIR_THRESHOLD = 0.90

HAND_VELOCITY_THRESHOLD = 0.15

# The target links are reference points near the palms, not tiny physical
# sockets.  Seven centimetres keeps both hands inside the 10 cm Stage-2 drift
# envelope while allowing residual whole-body sway from the walking policy.
DISTANCE_TO_CHAIR_HANDLE_THRESHOLD = 0.07
ORIENTATION_DISTANCE_HANDLE_THRESHOLD = 0.03
GRASP_DRIFT_THRESHOLD = 0.1
GRASP_FORCE_THRESHOLD = 0.5
GRASP_MIN_TIPS_PER_HAND = 2
GRASP_MIN_CLOSURE = 0.55
STAGE0_HOLD_STEPS = 10
STAGE1_HOLD_STEPS = 5
STAGE2_HOLD_STEPS = 10

POS_THRESHOLD = 0.4
ORI_DOT_PRODUCT_THRESHOLD = 0.9
CHAIR_PULL_DISTANCE_THRESHOLD = 1.0

ARM_RESTING_THRESHOLD = 0.35

GRASP_FINGER_TARGETS = {
    "left_hand_thumb_0_joint": 0.396,
    "left_hand_thumb_1_joint": 0.700,
    "left_hand_thumb_2_joint": 1.000,
    "left_hand_middle_0_joint": -1.500,
    "left_hand_middle_1_joint": -1.700,
    "left_hand_index_0_joint": -1.500,
    "left_hand_index_1_joint": -1.700,
    "right_hand_thumb_0_joint": -0.396,
    "right_hand_thumb_1_joint": -0.700,
    "right_hand_thumb_2_joint": -1.000,
    "right_hand_middle_0_joint": 1.500,
    "right_hand_middle_1_joint": 1.700,
    "right_hand_index_0_joint": 1.500,
    "right_hand_index_1_joint": 1.700,
}


def _held_condition(handler, name, idx, condition, required_steps):
    """Require a checker condition for consecutive control steps."""
    num_envs = handler.num_envs if hasattr(handler, "num_envs") else handler.env.num_envs
    counters = getattr(handler.task, name, None)
    if counters is None or counters.shape[0] != num_envs:
        counters = torch.zeros(num_envs, dtype=torch.long, device=idx.device)
        setattr(handler.task, name, counters)
    counters[idx] = torch.where(condition, counters[idx] + 1, torch.zeros_like(counters[idx]))
    return counters[idx] >= required_steps


# =========================================================
# SNAPSHOT CONFIG
# =========================================================

# Pokud True, při startu se načtou snapshoty z disku do RAM bufferu.
ENABLE_DISK_SNAPSHOT_LOAD = False

# Pokud True, nové snapshoty se budou průběžně zapisovat i na disk.
ENABLE_DISK_SNAPSHOT_SAVE = False

SNAPSHOT_DIR = Path("config_run/snapshots_chair/")
MAX_SNAPSHOTS = 100
# Pokud True, všechny envy vždy startují od stage 0
# a snapshot curriculum se zcela ignoruje.
FORCE_START_FROM_STAGE0 = False
RAM_SNAPSHOT_BUFFER = {1: [], 2: [], 3: [], 4: [], 5: []}
BUFFER_INITIALIZED = False
UNSAVED_COUNT = 0
SYNC_THRESHOLD = 30  # Každých 50 uložených snapshotů se jeden zapíše trvale na disk
LOCK = threading.Lock()

def init_ram_buffer():
    """Inicializuje RAM snapshot buffer. Volitelně načte snapshoty z disku."""
    global BUFFER_INITIALIZED, RAM_SNAPSHOT_BUFFER

    if BUFFER_INITIALIZED:
        return

    # Vždy začneme s čistým RAM bufferem
    RAM_SNAPSHOT_BUFFER = {1: [], 2: [], 3: [], 4: [], 5: []}

    if not ENABLE_DISK_SNAPSHOT_LOAD:
        BUFFER_INITIALIZED = True
        print("RAM Snapshot Buffer inicializován bez načítání z disku.")
        return

    print("Inicializuji RAM Snapshot Buffer z disku...")
    for stage in range(1, 6):
        stage_dir = SNAPSHOT_DIR / f"stage_{stage}"
        if stage_dir.exists():
            files = list(stage_dir.glob("*.pkl"))
            for f in files:
                try:
                    with open(f, "rb") as file:
                        data = pickle.load(file)
                        RAM_SNAPSHOT_BUFFER[stage].append(data)
                except Exception:
                    pass

    BUFFER_INITIALIZED = True
    counts = [len(RAM_SNAPSHOT_BUFFER[s]) for s in range(1, 6)]
    print(f"RAM Buffer načten. Počty snapshotů pro stages 1-5: {counts}")


def _sync_to_disk_worker(stage, snapshot_data, snapshot_idx):
    """Pracovník na pozadí, který uloží 1 soubor na disk bez zablokování tréninku."""
    if not ENABLE_DISK_SNAPSHOT_SAVE:
        return

    stage_dir = SNAPSHOT_DIR / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    filename = stage_dir / f"snapshot_{snapshot_idx}.pkl"
    try:
        with open(filename, "wb") as f:
            pickle.dump(snapshot_data, f)
    except Exception:
        pass


# ---------------------------------------------------------
# VEKTORIZOVANÉ POMOCNÉ FUNKCE (S PŘED-MASKOVÁNÍM)
# ---------------------------------------------------------

def check_movement_chair(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.BoolTensor:
    """Kontroluje pohyb židle POUZE pro aktivní indexy (idx)."""
    idx_base_chair = states.objects["chair"].body_names.index("base_link")
    chair_pos = states.objects["chair"].body_state[idx, idx_base_chair, :3]
    chair_ori = states.objects["chair"].body_state[idx, idx_base_chair, 3:7]

    initial_chair_ori = torch.tensor([7.0739e-01, 8.4260e-08, 0.0000e+00, 7.0683e-01], device=chair_ori.device)
    initial_chair_pos = torch.tensor([0.75, 0.0, 0.1], device=chair_pos.device)

    pos_diff = torch.norm(chair_pos - initial_chair_pos, dim=-1)
    dot_product = torch.abs(torch.sum(chair_ori * initial_chair_ori, dim=-1))

    return (pos_diff > POS_THRESHOLD) #| (dot_product < ORI_DOT_PRODUCT_THRESHOLD)

def common_chairman_checker(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor, stage_id: int) -> torch.BoolTensor:
    """Kontroluje pád robota a nově i časový limit (timeout) pro danou stage."""
    num_envs = states.robots[handler.robot.name].joint_pos.shape[0]
    device = idx.device

    # Inicializace paměti pro kroky (proběhne jen při prvním průchodu)
    if not hasattr(handler.task, "stage_steps"):
        handler.task.stage_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        # -1 znamená, že prostředí ještě nemá zapsanou žádnou stage
        handler.task.recorded_stage = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    # Zjistíme, jestli některé prostředí nepřešlo do nové stage od posledního kroku
    changed_mask = handler.task.recorded_stage[idx] != stage_id

    # Pokud ano, VYNULUJEME mu počítadlo času (dostává nový čas na novou stage)
    reset_idx = idx[changed_mask]
    if len(reset_idx) > 0:
        handler.task.stage_steps[reset_idx] = 0

    # Zapíšeme si aktuální stage
    handler.task.recorded_stage[idx] = stage_id

    # Inkrementujeme odpracovaný krok (o 1)
    handler.task.stage_steps[idx] += 1

    # --- 1. PODMÍNKA PÁDU ---
    is_fallen = neck_height_tensor(states, handler.robot.name)[idx] < HEIGHT_THRESHOLD

    # --- 2. PODMÍNKA TIMEOUTU ---
    reference_steps = STAGE_TIMEOUTS.get(stage_id, 9999)
    physics_dt = getattr(handler.scenario.sim_params, "dt", None) or 0.002
    task_dt = float(physics_dt) * int(handler.scenario.decimation)
    limit = max(
        1,
        int(round(reference_steps * STAGE_TIMEOUT_REFERENCE_DT / task_dt)),
    )
    is_timeout = handler.task.stage_steps[idx] > limit

    # Výsledek: Epizoda skončí, pokud robot spadne NEBO pokud mu vyprší čas
    return is_fallen | is_timeout

def get_batch_grasp_status(states: list[EnvState], handler: BaseSimHandler, force_threshold: float, idx: torch.Tensor) -> torch.Tensor:
    """Kontrola kontaktů POUZE pro indexy robotů ve Stage 2."""
    device = handler.device
    num_active = len(idx)

    contact_data = states.robots[handler.robot.name].contact
    if contact_data is None:
        return torch.zeros(num_active, dtype=torch.bool, device=device)

    # Vytažení POUZE aktivních řádků
    link_a = contact_data['link_a'][idx]
    link_b = contact_data['link_b'][idx]
    valid_mask = contact_data['valid_mask'][idx]

    forces = contact_data.get('force_b', contact_data.get('force', None))
    if forces is None:
        forces = torch.zeros((*link_a.shape, 3), device=device)
    else:
        forces = forces[idx]

    force_mags = torch.norm(forces, dim=-1)

    finger_tips = {"thumb_2": 0, "index_1": 1, "middle_1": 2}
    total_tips_per_hand = len(finger_tips)

    global_map = states.extras.get("global_link_map", {})
    num_bodies = states.extras.get("num_bodies_per_env", 1000)

    idx_to_tip_left = torch.full((num_bodies,), -1, dtype=torch.long, device=device)
    idx_to_tip_right = torch.full((num_bodies,), -1, dtype=torch.long, device=device)
    chair_ids = []

    for c_idx, (o_name, l_name) in global_map.items():
        if o_name == handler.robot.name:
            if "left" in l_name:
                for tip, t_id in finger_tips.items():
                    if tip in l_name: idx_to_tip_left[c_idx] = t_id
            elif "right" in l_name:
                for tip, t_id in finger_tips.items():
                    if tip in l_name: idx_to_tip_right[c_idx] = t_id
        elif o_name == "chair":
            chair_ids.append(c_idx)

    chair_ids = torch.tensor(chair_ids, device=device)

    base_a = link_a % num_bodies
    base_b = link_b % num_bodies

    a_is_chair = torch.isin(base_a, chair_ids)
    b_is_chair = torch.isin(base_b, chair_ids)

    contact_base = torch.where(b_is_chair, base_a, torch.where(a_is_chair, base_b, torch.tensor(-1, device=device)))
    valid_strong = valid_mask & (force_mags >= force_threshold) & (contact_base >= 0)

    left_status = torch.zeros((num_active, total_tips_per_hand), dtype=torch.bool, device=device)
    right_status = torch.zeros((num_active, total_tips_per_hand), dtype=torch.bool, device=device)

    for t_id in range(total_tips_per_hand):
        is_left_tip = (idx_to_tip_left[contact_base] == t_id)
        left_status[:, t_id] = torch.any(valid_strong & is_left_tip, dim=1)
        is_right_tip = (idx_to_tip_right[contact_base] == t_id)
        right_status[:, t_id] = torch.any(valid_strong & is_right_tip, dim=1)

    # Requiring all six tips at exactly the same instant made the transition
    # dominated by contact jitter. Two contacts per hand still constitutes a
    # real bilateral grasp and is robust to one unloaded fingertip.
    success_left = torch.sum(left_status, dim=1) >= GRASP_MIN_TIPS_PER_HAND
    success_right = torch.sum(right_status, dim=1) >= GRASP_MIN_TIPS_PER_HAND
    return success_left & success_right

def get_batch_any_grasp_status(states: list[EnvState], handler: BaseSimHandler, force_threshold: float, idx: torch.Tensor) -> torch.Tensor:
    """Kontrola kontaktů (Stage 3+) - Stačí, když se židle drží alespoň JEDNÍM prstem na každé ruce."""
    device = handler.device
    num_active = len(idx)

    contact_data = states.robots[handler.robot.name].contact
    if contact_data is None:
        return torch.zeros(num_active, dtype=torch.bool, device=device)

    link_a = contact_data['link_a'][idx]
    link_b = contact_data['link_b'][idx]
    valid_mask = contact_data['valid_mask'][idx]

    forces = contact_data.get('force_b', contact_data.get('force', None))
    if forces is None:
        forces = torch.zeros((*link_a.shape, 3), device=device)
    else:
        forces = forces[idx]

    force_mags = torch.norm(forces, dim=-1)

    finger_tips = {"thumb_2": 0, "index_1": 1, "middle_1": 2}
    total_tips_per_hand = len(finger_tips)

    global_map = states.extras.get("global_link_map", {})
    num_bodies = states.extras.get("num_bodies_per_env", 1000)

    idx_to_tip_left = torch.full((num_bodies,), -1, dtype=torch.long, device=device)
    idx_to_tip_right = torch.full((num_bodies,), -1, dtype=torch.long, device=device)
    chair_ids = []

    for c_idx, (o_name, l_name) in global_map.items():
        if o_name == handler.robot.name:
            if "left" in l_name:
                for tip, t_id in finger_tips.items():
                    if tip in l_name: idx_to_tip_left[c_idx] = t_id
            elif "right" in l_name:
                for tip, t_id in finger_tips.items():
                    if tip in l_name: idx_to_tip_right[c_idx] = t_id
        elif o_name == "chair":
            chair_ids.append(c_idx)

    chair_ids = torch.tensor(chair_ids, device=device)

    base_a = link_a % num_bodies
    base_b = link_b % num_bodies

    a_is_chair = torch.isin(base_a, chair_ids)
    b_is_chair = torch.isin(base_b, chair_ids)

    contact_base = torch.where(b_is_chair, base_a, torch.where(a_is_chair, base_b, torch.tensor(-1, device=device)))
    valid_strong = valid_mask & (force_mags >= force_threshold) & (contact_base >= 0)

    left_status = torch.zeros((num_active, total_tips_per_hand), dtype=torch.bool, device=device)
    right_status = torch.zeros((num_active, total_tips_per_hand), dtype=torch.bool, device=device)

    for t_id in range(total_tips_per_hand):
        is_left_tip = (idx_to_tip_left[contact_base] == t_id)
        left_status[:, t_id] = torch.any(valid_strong & is_left_tip, dim=1)
        is_right_tip = (idx_to_tip_right[contact_base] == t_id)
        right_status[:, t_id] = torch.any(valid_strong & is_right_tip, dim=1)

    # ZMĚNA ZDE: any() místo all() - stačí jeden prst (True) v daném sloupci
    success_left = torch.any(left_status, dim=1)
    success_right = torch.any(right_status, dim=1)

    return success_left & success_right
# ---------------------------------------------------------
# VEKTORIZOVANÉ CHECKERY S PŘED-MASKOVÁNÍM
# ---------------------------------------------------------

def stege0_chacker(states: list[EnvState], handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    num_envs = mask.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)

    # Získáme pouze indexy, kde je maska True (Optimalizace rychlosti GPU)
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return terminated, success

    term_common = common_chairman_checker(states, handler, idx, stage_id=0) | check_movement_chair(states, handler, idx)
    # Stejný finální bod jako ve WalkToChairProgressReward: 0.75 m za
    # opěradlem, nezávisle na natočení židle ve světě.
    robot_state = states.robots[handler.robot.name]
    base_link_idx = robot_state.body_names.index("pelvis")
    robot_base_state = robot_state.body_state[idx, base_link_idx]
    robot_pos = robot_base_state[:, :3]
    robot_quat = robot_base_state[:, 3:7]
    chair_base_idx = states.objects["chair"].body_names.index("base_link")
    chair_state = states.objects["chair"].body_state[idx, chair_base_idx]
    chair_pos = chair_state[:, :3]
    chair_back_dir = chair_back_direction_xy(chair_state[:, 3:7])
    final_target = chair_pos[:, :2] + CHAIR_FINAL_DISTANCE * chair_back_dir
    final_position_error = torch.norm(robot_pos[:, :2] - final_target, dim=-1)

    to_chair = chair_pos[:, :2] - robot_pos[:, :2]
    to_chair = to_chair / torch.clamp(torch.norm(to_chair, dim=-1, keepdim=True), min=1.0e-6)
    robot_forward = forward_direction_xy(robot_quat)
    facing_chair = torch.sum(robot_forward * to_chair, dim=-1)

    # --- NOVÉ: Výpočet rychlosti robota ---
    # root_state obsahuje: pos(0:3), quat(3:7), lin_vel(7:10), ang_vel(10:13)
    robot_lin_vel_xy = robot_base_state[:, 7:9]
    # Úspěch chůze posuzujeme v rovině podlahy. Vertikální kmit pánve při
    # balancování není pohyb směrem od cíle a nemá blokovat přechod do stage 1.
    vel_norm = torch.norm(robot_lin_vel_xy, dim=-1)

    # Úspěch: robot je u finálního bodu, stojí a je čelem k židli.
    success_now = (
        (final_position_error <= CHAIR_FINAL_TOLERANCE)
        & (vel_norm < VELOCITY_THRESHOLD)
        & (facing_chair >= FACING_CHAIR_THRESHOLD)
    )
    success_cond = _held_condition(
        handler, "stage0_success_steps", idx, success_now, STAGE0_HOLD_STEPS
    )
    # Zápis výsledků zpět na správné indexy do velkého tenzoru
    terminated[idx] = term_common | success_cond
    success[idx] = success_cond & (~term_common)

    return terminated, success

def stege1_chacker(states: list[EnvState], handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    num_envs = mask.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)

    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return terminated, success

    term_common = common_chairman_checker(states, handler, idx, stage_id=1) | check_movement_chair(states, handler, idx)

    # --- POZICE A ORIENTACE RUKOU ---
    right_ee_pos = right_palm_position(states, handler.robot.name, ee_name="endeffector")[idx]
    right_ee_ori = right_palm_orientation(states, handler.robot.name, ee_name="endeffector")[idx]
    left_ee_pos = right_palm_position(states, handler.robot.name, ee_name="left_endeffector")[idx]
    left_ee_ori = right_palm_orientation(states, handler.robot.name, ee_name="left_endeffector")[idx]

    # --- NOVÉ: ZÍSKÁNÍ RYCHLOSTI RUKOU ---
    robot_state = states.robots[handler.robot.name]
    l_ee_idx = robot_state.body_names.index("left_endeffector")
    r_ee_idx = robot_state.body_names.index("endeffector")

    # Indexy 7:10 v body_state obsahují lineární rychlost [x, y, z]
    left_ee_vel = robot_state.body_state[idx, l_ee_idx, 7:10]
    right_ee_vel = robot_state.body_state[idx, r_ee_idx, 7:10]

    # Vypočítáme velikost rychlosti v m/s
    left_vel_norm = torch.norm(left_ee_vel, dim=-1)
    right_vel_norm = torch.norm(right_ee_vel, dim=-1)
    # -------------------------------------

    chair = states.objects["chair"]
    r_idx = chair.body_names.index("target_hand_right")
    l_idx = chair.body_names.index("target_hand_left")

    r_handle_pos, r_handle_ori = chair.body_state[idx, r_idx, :3], chair.body_state[idx, r_idx, 3:7]
    l_handle_pos, l_handle_ori = chair.body_state[idx, l_idx, :3], chair.body_state[idx, l_idx, 3:7]

    left_dist = torch.norm(left_ee_pos - l_handle_pos, dim=-1)
    right_dist = torch.norm(right_ee_pos - r_handle_pos, dim=-1)

    left_dot = torch.abs(torch.sum(l_handle_ori * left_ee_ori, dim=-1))
    right_dot = torch.abs(torch.sum(r_handle_ori * right_ee_ori, dim=-1))

    l_ori_dist = 1.0 - left_dot
    r_ori_dist = 1.0 - right_dot

    # Předpokládáme, že máte nadefinováno HAND_VELOCITY_THRESHOLD (např. 0.1 nebo 0.05)
    # Můžete zde použít i váš stávající VELOCITY_THRESHOLD


    # --- PŘIDÁNA PODMÍNKA RYCHLOSTI ---
    success_now = (left_dist < DISTANCE_TO_CHAIR_HANDLE_THRESHOLD) & \
                   (right_dist < DISTANCE_TO_CHAIR_HANDLE_THRESHOLD) & \
                   (l_ori_dist < ORIENTATION_DISTANCE_HANDLE_THRESHOLD) & \
                   (r_ori_dist < ORIENTATION_DISTANCE_HANDLE_THRESHOLD) & \
                   (left_vel_norm < HAND_VELOCITY_THRESHOLD) & \
                   (right_vel_norm < HAND_VELOCITY_THRESHOLD)
    success_cond = _held_condition(
        handler, "stage1_success_steps", idx, success_now, STAGE1_HOLD_STEPS
    )

    terminated[idx] = term_common | success_cond
    success[idx] = success_cond & (~term_common)
    return terminated, success
def stege2_chacker(states: list[EnvState], handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    num_envs = mask.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)

    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return terminated, success

    term_common = common_chairman_checker(states, handler, idx, stage_id=2) | check_movement_chair(states, handler, idx)

    right_ee_pos = right_palm_position(states, handler.robot.name, ee_name="endeffector")[idx]
    left_ee_pos = right_palm_position(states, handler.robot.name, ee_name="left_endeffector")[idx]

    chair = states.objects["chair"]
    r_handle_pos = chair.body_state[idx, chair.body_names.index("target_hand_right"), :3]
    l_handle_pos = chair.body_state[idx, chair.body_names.index("target_hand_left"), :3]

    dist_right = torch.norm(right_ee_pos - r_handle_pos, dim=-1)
    dist_left = torch.norm(left_ee_pos - l_handle_pos, dim=-1)

    hands_near = (dist_right <= GRASP_DRIFT_THRESHOLD) & (dist_left <= GRASP_DRIFT_THRESHOLD)

    # A contact-only checker can be passed by a brief collision without ever
    # learning to close the hand. Explicit closure keeps the checker aligned
    # with CloseGraspReward.
    joint_names = list(states.robots[handler.robot.name].joint_names)
    finger_indices = [joint_names.index(name) for name in GRASP_FINGER_TARGETS]
    finger_targets = torch.tensor(
        list(GRASP_FINGER_TARGETS.values()),
        dtype=states.robots[handler.robot.name].joint_pos.dtype,
        device=mask.device,
    )
    q_finger = states.robots[handler.robot.name].joint_pos[idx][:, finger_indices]
    closure_per_joint = torch.clamp(
        1.0 - torch.abs(q_finger - finger_targets) / torch.clamp(torch.abs(finger_targets), min=0.1),
        min=0.0,
        max=1.0,
    )
    both_hands_closed = (
        torch.mean(closure_per_joint[:, :7], dim=-1) >= GRASP_MIN_CLOSURE
    ) & (
        torch.mean(closure_per_joint[:, 7:], dim=-1) >= GRASP_MIN_CLOSURE
    )
    contacts_ok = get_batch_grasp_status(states, handler, GRASP_FORCE_THRESHOLD, idx)
    success_now = hands_near & both_hands_closed & contacts_ok
    success_cond = _held_condition(
        handler, "stage2_success_steps", idx, success_now, STAGE2_HOLD_STEPS
    )

    # Losing the exact 10 cm reach pose is recoverable and therefore must not
    # reset the episode immediately. The reach/stillness rewards guide it back.
    terminated[idx] = term_common | success_cond
    success[idx] = success_cond & (~term_common)
    return terminated, success

def stege3_chacker(states: list[EnvState], handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    num_envs = mask.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)

    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return terminated, success

    # --- 1. Spadl robot, nebo vypršel čas? ---
    # Po Stage 2 už je pohyb židle žádoucí, proto zde nepoužíváme check_movement_chair.
    term_common = common_chairman_checker(states, handler, idx, stage_id=3)

    # --- 2. Kontrola Driftu (zda mu neujely ruce z madel) ---
    right_ee_pos = right_palm_position(states, handler.robot.name, ee_name="endeffector")[idx]
    left_ee_pos = right_palm_position(states, handler.robot.name, ee_name="left_endeffector")[idx]

    chair = states.objects["chair"]
    r_handle_pos = chair.body_state[idx, chair.body_names.index("target_hand_right"), :3]
    l_handle_pos = chair.body_state[idx, chair.body_names.index("target_hand_left"), :3]

    dist_right = torch.norm(right_ee_pos - r_handle_pos, dim=-1)
    dist_left = torch.norm(left_ee_pos - l_handle_pos, dim=-1)

    drift_fail = (dist_right > GRASP_DRIFT_THRESHOLD) | (dist_left > GRASP_DRIFT_THRESHOLD)

    # --- 3. Kontrola úchopu (alespoň 1 prstem každé ruky) ---
    has_any_grasp = get_batch_any_grasp_status(states, handler, GRASP_FORCE_THRESHOLD, idx)
    grasp_fail = ~has_any_grasp

    # --- 4. Kontrola posunu židle dozadu ---
    chair_base_idx = chair.body_names.index("base_link")
    chair_pos = chair.body_state[idx, chair_base_idx, :3]
    initial_chair_pos = torch.tensor([0.75, 0.0, 0.1], device=chair_pos.device)
    target_chair_pos = torch.tensor([-0.25, 0.0, 0.1], device=chair_pos.device)

    # Úspěch má odpovídat pozici, kterou vyžadují Stage 4/5.
    pulled_x = initial_chair_pos[0] - chair_pos[:, 0]
    chair_pos_diff = torch.norm(chair_pos - target_chair_pos, dim=-1)
    chair_moved_enough = (pulled_x >= CHAIR_PULL_DISTANCE_THRESHOLD) & (chair_pos_diff <= POS_THRESHOLD)

    # --- 5. Kontrola zastavení robota ---
    base_link_idx = states.robots[handler.robot.name].body_names.index("pelvis")
    robot_lin_vel = states.robots[handler.robot.name].body_state[idx, base_link_idx, 7:10]
    vel_norm = torch.norm(robot_lin_vel, dim=-1)
    standing_still = vel_norm < VELOCITY_THRESHOLD

    # --- VYHODNOCENÍ ---
    # Fail: pokud spadne, ujede mu ruka, nebo zcela ztratí kontakt prstů s židlí
    fail_cond = term_common | drift_fail | grasp_fail

    # Success: neselhal (ani nepustil židli), odtáhl židli dostatečně daleko a zastavil se
    success_cond = (~fail_cond) & chair_moved_enough & standing_still

    # Pokud selže, tak skončil epizodu. Pokud uspěje, taky ukončí checker, ale se sukcessem.
    terminated[idx] = fail_cond | success_cond
    success[idx] = success_cond

    return terminated, success
def stege4_chacker(states: list[EnvState], handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    num_envs = mask.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)

    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0: return terminated, success

    # --- 1. Kontrola pádu robota ---
    # Židle už má být odtažená; její stabilitu kontrolujeme níže vůči nové cílové pozici.
    term_common = common_chairman_checker(states, handler, idx, stage_id=4)

    # --- 2. Kontrola pohybu ŽIDLE (Nesmí se pohnout z NOVÉ pozice) ---
    chair_base_idx = states.objects["chair"].body_names.index("base_link")
    chair_pos = states.objects["chair"].body_state[idx, chair_base_idx, :3]
    chair_lin_vel = states.objects["chair"].body_state[idx, chair_base_idx, 7:10]

    # Cílová pozice ze Stage 3 (0.75 - 1.0 = -0.25)
    target_chair_pos = torch.tensor([-0.25, 0.0, 0.1], device=mask.device)
    chair_pos_diff = torch.norm(chair_pos - target_chair_pos, dim=-1)
    chair_vel_norm = torch.norm(chair_lin_vel, dim=-1)

    # Termination: Pokud židle ujela pryč z cíle, nebo do ní drbnul a získala rychlost
    chair_moved = (chair_pos_diff > POS_THRESHOLD) | (chair_vel_norm > VELOCITY_THRESHOLD)

    # --- 3. Kontrola pohybu ROBOTA (Nesmí se hýbat) ---
    base_link_idx = states.robots[handler.robot.name].body_names.index("pelvis")
    robot_lin_vel = states.robots[handler.robot.name].body_state[idx, base_link_idx, 7:10]
    robot_vel_norm = torch.norm(robot_lin_vel, dim=-1)

    # Termination: Pokud robot neudrží stabilitu a začne padat/couvat
    robot_moved = robot_vel_norm > VELOCITY_THRESHOLD

    # --- 4. Kontrola OTEVŘENÍ PRSTŮ ---
    # Automaticky najdeme indexy všech prstů
    joint_names = states.robots[handler.robot.name].joint_names
    finger_keywords = ["thumb", "index", "middle"]
    finger_indices = [i for i, name in enumerate(joint_names) if any(k in name for k in finger_keywords)]

    # Získáme pozice prstů a zkontrolujeme, zda jsou blízko 0.0 (otevřeno)
    q_fingers = states.robots[handler.robot.name].joint_pos[idx][:, finger_indices]

    # max() vrátí tuple (values, indices), chceme jen values na indexu [0]
    max_finger_angle = torch.max(torch.abs(q_fingers), dim=-1)[0]

    # Úspěch: Nejvíce ohnutý prst musí být pod naším thresholdem (0.15)
    FINGER_OPEN_THRESHOLD = 0.15
    fingers_open = max_finger_angle < FINGER_OPEN_THRESHOLD

    # --- VYHODNOCENÍ ---
    # FAIL: Pokud robot spadne, pohne židlí, nebo sám ztratí rovnováhu a začne se hýbat
    fail_cond = term_common | chair_moved | robot_moved

    # SUCCESS: Neselhal (vše stojí na místě) A ZÁROVEŇ jsou prsty plně otevřené
    success_cond = (~fail_cond) & fingers_open

    # Ukončení a zápis výsledků
    terminated[idx] = fail_cond | success_cond
    success[idx] = success_cond

    return terminated, success

def stege5_chacker(states: list[EnvState], handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    num_envs = mask.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)

    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return terminated, success

    # --- 1. Kontrola pádu robota ---
    # Židle už má být odtažená; její stabilitu kontrolujeme níže vůči nové cílové pozici.
    term_common = common_chairman_checker(states, handler, idx, stage_id=5)

    # --- 2. Kontrola pohybu ŽIDLE (Nesmí se pohnout z NOVÉ pozice) ---
    # Židle už je odtažená na pozici x = -0.25
    chair_base_idx = states.objects["chair"].body_names.index("base_link")
    chair_pos = states.objects["chair"].body_state[idx, chair_base_idx, :3]
    chair_lin_vel = states.objects["chair"].body_state[idx, chair_base_idx, 7:10]

    target_chair_pos = torch.tensor([-0.25, 0.0, 0.1], device=mask.device)
    chair_pos_diff = torch.norm(chair_pos - target_chair_pos, dim=-1)
    chair_vel_norm = torch.norm(chair_lin_vel, dim=-1)

    # Termination: Židle odjela, nebo do ní kopnul a rozjela se
    chair_moved = (chair_pos_diff > POS_THRESHOLD) | (chair_vel_norm > VELOCITY_THRESHOLD)

    # --- 3. Kontrola pohybu ROBOTA (Nesmí couvat ani jít vpřed) ---
    base_link_idx = states.robots[handler.robot.name].body_names.index("pelvis")
    robot_lin_vel = states.robots[handler.robot.name].body_state[idx, base_link_idx, 7:10]
    robot_vel_norm = torch.norm(robot_lin_vel, dim=-1)

    # Termination: Robot nezastavil, potácí se
    robot_moved = robot_vel_norm > VELOCITY_THRESHOLD

    # --- 4. Kontrola RUKOU PODÉL TĚLA (Success condition) ---
    # Zkontrolujeme klíčové klouby ramen a loktů, zda jsou blízko nuly
    joint_names = states.robots[handler.robot.name].joint_names
    arm_joints_to_check = [
        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "left_shoulder_roll_joint", "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
        "left_elbow_joint", "right_elbow_joint"
    ]

    arm_indices = []
    for i, name in enumerate(joint_names):
        if name in arm_joints_to_check:
            arm_indices.append(i)

    q_arms = states.robots[handler.robot.name].joint_pos[idx][:, arm_indices]

    # max() vrátí nejdále vychýlený kloub ze všech osmi sledovaných
    max_arm_angle = torch.max(torch.abs(q_arms), dim=-1)[0]

    # Jsou ruce spuštěné volně podél těla?
    arms_are_down = max_arm_angle < ARM_RESTING_THRESHOLD

    # --- VYHODNOCENÍ ---
    # FAIL: Pokud robot spadne, posune odloženou židli, nebo nezvládne zastavit a padá do stran
    fail_cond = term_common | chair_moved | robot_moved

    # SUCCESS: Vše stojí jak má (neselhal) A ZÁROVEŇ jsou obě paže volně podél těla
    success_cond = (~fail_cond) & arms_are_down

    # Ukončení a zápis výsledků
    terminated[idx] = fail_cond | success_cond
    success[idx] = success_cond

    return terminated, success

def reset_chairman(handler: BaseSimHandler, env_ids: list[int] | None = None):
    global BUFFER_INITIALIZED

    # 1. Inicializace RAM bufferu
    if not BUFFER_INITIALIZED:
        init_ram_buffer()

    states = [stage0_init(handler.robot.name)] * handler.num_envs
    if env_ids is None:
        env_ids = list(range(handler.num_envs))

    for counter_name in (
        "stage0_success_steps", "stage1_success_steps", "stage2_success_steps"
    ):
        counter = getattr(handler.task, counter_name, None)
        if counter is not None:
            counter[env_ids] = 0

    current_stages_tensor = handler.task.reward_functions[0].actual_stage
    if current_stages_tensor is None:
        current_stages_tensor = torch.tensor([0] * handler.num_envs, device=handler.device)
        current_stages_completed = torch.tensor([0] * handler.num_envs, device=handler.device)
        for i in range(len(handler.task.reward_functions)):
            handler.task.reward_functions[i].actual_stage = current_stages_tensor
            handler.task.reward_functions[i].completed_stages = current_stages_completed

    # =====================================================
    # Režim: vždy start od stage 0
    # =====================================================
    if FORCE_START_FROM_STAGE0:
        print(f"[reset_chairman] FORCE_START_FROM_STAGE0=True -> reset envs {env_ids} vždy do stage 0")

        for env in env_ids:
            for i in range(len(handler.task.reward_functions)):
                handler.task.reward_functions[i].actual_stage[env] = 0
                handler.task.reward_functions[i].completed_stages[env] = 0

            states[env] = stage0_init(handler.robot.name)
            print(f"[reset_chairman] env {env} -> stage 0")

        if hasattr(handler.task, "recorded_stage") and env_ids is not None:
            handler.task.recorded_stage[env_ids] = -1

        handler.set_states(states=states, env_ids=env_ids)
        states2 = handler.get_states()

        for reward_fn in handler.task.reward_functions:
            if hasattr(reward_fn, "reset"):
                reward_fn.reset(env_ids=env_ids, states=states2)

        return

    # =====================================================
    # Curriculum režim
    # =====================================================
    max_available_stage = 0
    for i in range(1, 6):
        if len(RAM_SNAPSHOT_BUFFER[i]) > 0:
            max_available_stage = i
        else:
            break

    #print(f"[reset_chairman] Max available stage found in RAM buffer: {max_available_stage}")

    for env in env_ids:
        new_stage = random.randint(0, max_available_stage)
        for i in range(len(handler.task.reward_functions)):
            handler.task.reward_functions[i].actual_stage[env] = new_stage
            handler.task.reward_functions[i].completed_stages[env] = 0

        #print(f"[reset_chairman] Resetting env {env} to stage {new_stage} (max available: {max_available_stage})")

        state = None
        if new_stage > 0:
            state = load_snapshot_chairman(stage=new_stage)

        if state is None:
            if new_stage > 0:
                #print(f"[reset_chairman] Warning: failed to load snapshot for stage {new_stage}, reverting env {env} to stage 0")
                for i in range(len(handler.task.reward_functions)):
                    handler.task.reward_functions[i].actual_stage[env] = 0
                    handler.task.reward_functions[i].completed_stages[env] = 0

            state = stage0_init(handler.robot.name)
            #print(f"[reset_chairman] env {env} -> using procedural stage0_init")
        else:
            #print(f"[reset_chairman] env {env} -> loaded snapshot for stage {new_stage}")
            pass

        states[env] = state

    if hasattr(handler.task, "recorded_stage") and env_ids is not None:
        handler.task.recorded_stage[env_ids] = -1

    handler.set_states(states=states, env_ids=env_ids)
    states2 = handler.get_states()

    for reward_fn in handler.task.reward_functions:
        if hasattr(reward_fn, "reset"):
            reward_fn.reset(env_ids=env_ids, states=states2)



def save_snapshot_chairman(handler: BaseSimHandler, env_id: int, stage: int) -> None:
    global UNSAVED_COUNT

    # Když chceme trénovat vždy od nuly, snapshoty vůbec neřešíme
    if FORCE_START_FROM_STAGE0:
        return

    full_states = handler.get_states()
    snapshot_data = {"robots": {}, "objects": {}}

    robot_name = handler.robot.name
    robot_states = full_states.robots[robot_name]
    joint_names = robot_states.joint_names.tolist()
    joint_pos = robot_states.joint_pos[env_id].detach().cpu().numpy()
    joint_vel = robot_states.joint_vel[env_id].detach().cpu().numpy()

    dof_pos = {name: pos for name, pos in zip(joint_names, joint_pos)}
    dof_vel = {name: vel for name, vel in zip(joint_names, joint_vel)}

    snapshot_data["robots"][robot_name] = {
        "pos": robot_states.root_state[env_id, :3].detach().cpu().clone(),
        "rot": robot_states.root_state[env_id, 3:7].detach().cpu().clone(),
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
    }

    for obj_name, obj_state in full_states.objects.items():
        if obj_name == "room":
            snapshot_data["objects"][obj_name] = {
                "pos": obj_state.root_state[env_id, :3].detach().cpu().clone(),
                "rot": obj_state.root_state[env_id, 3:7].detach().cpu().clone(),
            }

        obj_joint_names = obj_state.joint_names.tolist()
        obj_joint_pos = obj_state.joint_pos[env_id].detach().cpu().numpy()
        obj_joint_vel = obj_state.joint_vel[env_id].detach().cpu().numpy()

        o_dof_pos = {name: pos for name, pos in zip(obj_joint_names, obj_joint_pos)}
        o_dof_vel = {name: vel for name, vel in zip(obj_joint_names, obj_joint_vel)}

        snapshot_data["objects"][obj_name] = {
            "pos": obj_state.root_state[env_id, :3].detach().cpu().clone(),
            "rot": obj_state.root_state[env_id, 3:7].detach().cpu().clone(),
            "dof_pos": o_dof_pos,
            "dof_vel": o_dof_vel,
        }

    # Uložení do RAM
    with LOCK:
        if len(RAM_SNAPSHOT_BUFFER[stage]) < MAX_SNAPSHOTS:
            RAM_SNAPSHOT_BUFFER[stage].append(snapshot_data)
            idx = len(RAM_SNAPSHOT_BUFFER[stage]) - 1
        else:
            idx = random.randint(0, MAX_SNAPSHOTS - 1)
            RAM_SNAPSHOT_BUFFER[stage][idx] = snapshot_data

        UNSAVED_COUNT += 1
        trigger_sync = False
        if ENABLE_DISK_SNAPSHOT_SAVE and UNSAVED_COUNT >= SYNC_THRESHOLD:
            trigger_sync = True
            UNSAVED_COUNT = 0

    # Volitelný zápis na disk
    if trigger_sync:
        thread = threading.Thread(target=_sync_to_disk_worker, args=(stage, snapshot_data, idx))
        thread.start()

def load_snapshot_chairman(stage: int) -> dict | None:
    # Když chceme jet vždy od nuly, snapshoty se vůbec nepoužijí
    if FORCE_START_FROM_STAGE0:
        return None

    with LOCK:
        if not RAM_SNAPSHOT_BUFFER[stage]:
            return None
        data = random.choice(RAM_SNAPSHOT_BUFFER[stage])

    formatted_data = {"robots": {}, "objects": {}}

    for rob_name, rob_data in data["robots"].items():
        formatted_data["robots"][rob_name] = {
            "pos": torch.as_tensor(rob_data["pos"], dtype=torch.float32),
            "rot": torch.as_tensor(rob_data["rot"], dtype=torch.float32),
            "dof_pos": rob_data["dof_pos"],
            "dof_vel": rob_data["dof_vel"],
        }

    for obj_name, obj_data in data["objects"].items():
        formatted_data["objects"][obj_name] = {
            "pos": torch.as_tensor(obj_data["pos"], dtype=torch.float32),
            "rot": torch.as_tensor(obj_data["rot"], dtype=torch.float32),
            "dof_pos": obj_data["dof_pos"],
            "dof_vel": obj_data["dof_vel"],
        }

    return formatted_data
# Ponechejte zde zbytek vašich původních reset a save funkcí (save_snapshot_chairman atd.)
# Doporučuji ponechat rychlou verzi ukládání pomocí random.randint(0, MAX_SNAPSHOTS) jak jsme řešili minule.
# def reset_chairman(handler: BaseSimHandler, env_ids: list[int] | None = None):
#     """
#     Reset s logikou "Curriculum Learning":
#     1. Zkontroluje, které stage (1-5) již mají uložené snapshoty.
#     2. Náhodně vybere stage mezi 0 a maximální dostupnou stage.
#     3. Inicializuje robota (buď procedurálně pro stage 0, nebo načtením snapshotu).
#     """
#     states = [stage0_init(handler.robot.name)] * handler.num_envs

#     if env_ids is None:
#         env_ids = list(range(handler.num_envs))

#     # Inicializace pole pro trackování stage v reward function, pokud neexistuje
#     current_stages_tensor = handler.task.reward_functions[0].actual_stage
#     if current_stages_tensor is None:
#         current_stages_tensor = torch.tensor([0] * handler.num_envs, device=handler.device)
#         current_stages_completed = torch.tensor([0] * handler.num_envs, device=handler.device)
#         for i in range(len(handler.task.reward_functions)):
#             handler.task.reward_functions[i].actual_stage = current_stages_tensor
#             handler.task.reward_functions[i].completed_stages = current_stages_completed

#     # --- KROK 1: Zjištění maximální dostupné stage ---
#     # Projdeme složky a zjistíme, kam až jsme se dostali.
#     # Stage 0 je dostupná vždy (procedurální).
#     max_available_stage = 0

#     # Předpokládáme max stage 5 dle definice
#     for i in range(1, 3):#TODO 6
#         stage_dir = SNAPSHOT_DIR / f"stage_{i}"
#         # Stage považujeme za dostupnou, pokud složka existuje a obsahuje alespoň jeden .pkl soubor
#         if stage_dir.exists() and any(stage_dir.glob("*.pkl")):
#             max_available_stage = i
#         else:
#             # Pokud chybí např. stage 2, nemá smysl hledat stage 3 (curriculum je postupné)
#             break

#     #print(f"DEBUG: Max available stage found: {max_available_stage}")

#     # --- KROK 2: Resetování jednotlivých prostředí ---
#     for env in env_ids:
#         # Náhodná volba stage: 0 až max_available_stage
#         new_stage = random.randint(0, max_available_stage)
#         #new_stage = 2 #TODO DEBUG!!! --- IGNORE ---
#         # Aktualizace informace o stage v reward funkcích
#         for i in range(len(handler.task.reward_functions)):
#             handler.task.reward_functions[i].actual_stage[env] = new_stage
#             handler.task.reward_functions[i].completed_stages[env] = 0

#         #print(f"Resetting env {env} to stage {new_stage} (Max avail: {max_available_stage})")

#         state = None

#         # Pokus o načtení stavu pro stage > 0
#         if new_stage > 0:
#             state = load_snapshot_chairman(stage=new_stage)

#         # --- KROK 3: Fallback a Stage 0 ---
#         # Pokud je stage 0, NEBO pokud načtení vyšší stage selhalo (state je None),
#         # provedeme inicializaci na stage 0.
#         if state is None:
#             if new_stage > 0:
#                 print(f"Warning: Failed to load snapshot for stage {new_stage}, reverting env {env} to Stage 0.")
#                 # Musíme opravit i záznam v reward funkci zpět na 0
#                 for i in range(len(handler.task.reward_functions)):
#                     handler.task.reward_functions[i].actual_stage[env] = 0
#                     handler.task.reward_functions[i].completed_stages[env] = 0
#             state = stage0_init(handler.robot.name)

#         # Sestavení listu states pro handler (zachování původní logiky pole)
#         if states is None:
#             states = [state] * handler.num_envs
#         else:
#             states[env] = state

#     handler.set_states(states=states, env_ids=env_ids)


# def save_snapshot_chairman(handler: BaseSimHandler, env_id: int, stage: int) -> None:
#     """
#     Uloží aktuální stav prostředí (robota a objektů) do souboru pro danou stage.
#     Kontroluje limit 100 snapshotů - pokud je překročen, smaže nejstarší.
#     """
#     # 1. Příprava adresáře pro danou stage
#     stage_dir = SNAPSHOT_DIR / f"stage_{stage}"
#     stage_dir.mkdir(parents=True, exist_ok=True)
#     # 2. Získání aktuálního stavu z handleru
#     full_states = handler.get_states()
#     snapshot_data = {
#         "robots": {},
#         "objects": {}
#     }
#     # Extrahuje data robota (převedeme na CPU a Numpy pro uložení)
#     robot_name = handler.robot.name
#     robot_states = full_states.robots[robot_name]
#     joint_names = robot_states.joint_names.tolist()
#     joint_pos = robot_states.joint_pos[env_id].detach().cpu().numpy()
#     joint_vel = robot_states.joint_vel[env_id].detach().cpu().numpy()

#     robot_pos = robot_states.root_state[env_id,:3].detach()
#     robot_rot = robot_states.root_state[env_id,3:7].detach()
#     dof_pos = {}
#     dof_vel = {}
#     for i, name in enumerate(joint_names):
#         dof_pos[name] = joint_pos[i]
#         dof_vel[name] = joint_vel[i]



#     snapshot_data["robots"][robot_name] = {
#         "pos": robot_pos,
#         "rot": robot_rot,
#         "dof_pos": dof_pos,
#         "dof_vel": dof_vel,
#     }


#     # Extrahuje data objektů (např. dveře)
#     for obj_name, obj_state in full_states.objects.items():
#         joint_names = obj_state.joint_names.tolist()
#         joint_pos = obj_state.joint_pos[env_id].detach().cpu().numpy()
#         joint_vel = obj_state.joint_vel[env_id].detach().cpu().numpy()
#         dof_pos = {}
#         dof_vel = {}
#         for i, name in enumerate(joint_names):
#             dof_pos[name] = joint_pos[i]
#             dof_vel[name] = joint_vel[i]


#         snapshot_data["objects"][obj_name] = {
#             "pos": obj_state.root_state[env_id,:3].detach(),
#             "rot": obj_state.root_state[env_id,3:7].detach(),
#             "dof_vel": dof_vel,
#             "dof_pos": dof_pos,
#         }
#     # 3. Kontrola limitu snapshotů (Mazání nejstaršího)
#     # Získáme seznam všech .pkl souborů v adresáři
#     list_of_files = sorted(stage_dir.glob("*.pkl"), key=os.path.getctime)

#     while len(list_of_files) >= MAX_SNAPSHOTS:
#         oldest_file = list_of_files.pop(0) # První je nejstarší
#         try:
#             os.remove(oldest_file)
#         except OSError as e:
#             print(f"Error deleting old snapshot: {e}")

#     # 4. Uložení nového snapshotu
#     # Název souboru obsahuje timestamp pro unikátnost
#     timestamp = int(time() * 1000)
#     filename = stage_dir / f"snapshot_{timestamp}_{env_id}.pkl"

#     with open(filename, 'wb') as f:
#         pickle.dump(snapshot_data, f)



# def load_snapshot_chairman(stage: int) -> dict | None:
#     """
#     Načte náhodný snapshot pro danou stage.
#     Vrací slovník se strukturou { "robots": {...}, "objects": {...} },
#     který je kompatibilní s handler.set_states().
#     """
#     stage_dir = SNAPSHOT_DIR / f"stage_{stage}"

#     # 1. Kontrola existence adresáře
#     if not stage_dir.exists():
#         return None

#     # 2. Získání seznamu všech snapshotů (.pkl soubory)
#     # glob vrací iterátor, převedeme na list
#     list_of_files = list(stage_dir.glob("*.pkl"))

#     if not list_of_files:
#         return None # Adresář existuje, ale je prázdný

#     # 3. Náhodný výběr jednoho souboru (Staged Reset logika)
#     random_file = random.choice(list_of_files)

#     # 4. Načtení dat
#     try:
#         with open(random_file, 'rb') as f:
#             snapshot_data = pickle.load(f)

#         # Data jsou již uložena jako {"robots": ..., "objects": ...} a hodnoty jsou numpy array/dict,
#         # což je přesně to, co handler.set_states obvykle zpracovává.
#         return snapshot_data

#     except Exception as e:
#         print(f"Chyba při načítání snapshotu {random_file}: {e}")
#         return None


def stage0_init(robot_name: str):
    if robot_name == "g1_slider":
        state = {
            "robots": {
                "g1_slider": {
                    "dof_pos": {
                        "baseslide_joint": 0.0,
                        "baseslide_joint2": -1.5,
                        "baserot_joint": 0.0,
                        #"waist_yaw_joint": 0.0,
                        #"waist_roll_joint": 0.0,
                        #"waist_pitch_joint": 0.0,
                        "left_shoulder_pitch_joint": 0.0,
                        "left_shoulder_roll_joint": 0.0,
                        "left_shoulder_yaw_joint": 0.0,
                        "left_elbow_joint": 0.0,
                        "left_wrist_roll_joint": 0.0,
                        "left_wrist_pitch_joint": 0.0,
                        "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
                        "right_wrist_roll_joint": 0.0,
                        "right_wrist_pitch_joint": 0.0,
                        "right_wrist_yaw_joint": 0.0,
                        "left_hand_thumb_0_joint": 0.0,
                        "left_hand_thumb_1_joint": 0.0,
                        "left_hand_thumb_2_joint": 0.0,
                        "left_hand_middle_0_joint": 0.0,
                        "left_hand_middle_1_joint": 0.0,
                        "left_hand_index_0_joint": 0.0,
                        "left_hand_index_1_joint": 0.0,
                        "right_hand_thumb_0_joint": 0.0,
                        "right_hand_thumb_1_joint": 0.0,
                        "right_hand_thumb_2_joint": 0.0,
                        "right_hand_middle_0_joint": 0.0,
                        "right_hand_middle_1_joint": 0.0,
                        "right_hand_index_0_joint": 0.0,
                        "right_hand_index_1_joint": 0.0
                    },
                    "pos": torch.tensor([
                        0.0,
                        0.0,
                        0.8
                    ]),
                    "rot": torch.tensor([
                        1.0,
                        0.0,
                        0.0,
                        0.0
                    ])
                },
            },
            "objects": {
                "chair": {
                        "pos": torch.tensor([
                            0.0,
                            0.0,
                            0.1
                        ]),
                        "rot": torch.tensor([
                            1.0,
                            0.0,
                            0.0,
                            0.0
                        ]),
                        "dof_pos":{
                            "floor_slide_x": 0.75,
                            "floor_slide_y": 0.0,
                            "floor_rotate_z": 1.57



                        }

                    },
                "room": {
                        "pos": torch.tensor([
                            0.0,
                            0.0,
                            0.0
                        ]),
                        "rot": torch.tensor([
                            0.0,
                            0.0,
                            0.0,
                            1.0
                        ])
                }
            }
        }
    elif robot_name == "g1_with_hands":
        state = {
            "robots": {
                "g1_with_hands": {
                    "pos" : torch.tensor([-2.5,0.0,0.8]),
                    "rot" : torch.tensor([1.0,0.0,0.0,0.0]),
                    "dof_pos": {
                        "left_hip_pitch_joint": -0.1,
                        "left_hip_roll_joint": 0.0,
                        "left_hip_yaw_joint": 0.0,
                        "left_knee_joint": 0.3,
                        "left_ankle_pitch_joint": -0.2,
                        "left_ankle_roll_joint": 0.0,
                        "right_hip_pitch_joint": -0.1,
                        "right_hip_roll_joint": 0.0,
                        "right_hip_yaw_joint": 0.0,
                        "right_knee_joint": 0.3,
                        "right_ankle_pitch_joint": -0.2,
                        "right_ankle_roll_joint": 0.0,
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.0,
                        "left_shoulder_pitch_joint": 0.0,
                        "left_shoulder_roll_joint": 0.0,
                        "left_shoulder_yaw_joint": 0.0,
                        "left_elbow_joint": 0.0,
                        "left_wrist_roll_joint": 0.0,
                        "left_wrist_pitch_joint": 0.0,
                        "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
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
                        "right_hand_index_1_joint": 0.0,

                    }
                }
            },
            "objects": {
                    "chair": {
                            "pos": torch.tensor([
                                0.0,
                                0.0,
                                0.03
                            ]),
                            "rot": torch.tensor([
                                1.0,
                                0.0,
                                0.0,
                                0.0
                            ]),
                            "dof_pos":{
                                "floor_slide_x": 0.75,
                                "floor_slide_y": 0.0,
                                "floor_rotate_z": 1.57



                            }

                        },
            },
        }

    return state

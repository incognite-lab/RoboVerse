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
try:
    from metasim.sim import BaseSimHandler
except:
    pass

STAGE_TIMEOUTS = {
    0: 400,   # dojit k zidli
    1: 120,   # spustit ruce dolu a zustat stat
    2: 500,   # couvat se zidli na target
    3: 120,   # zvednout ruce nahoru a zustat stat
}
VELOCITY_THRESHOLD = 0.2
HEIGHT_THRESHOLD = 0.4
ANGULAR_VELOCITY_THRESHOLD = 0.35
# stage 0
DISTANCE_TO_CHAIR_X_TARGET = 0.75
DISTANCE_TO_CHAIR_X_TOL = 0.08
DISTANCE_TO_CHAIR_Y_TOL = 0.20
FACING_FORWARD_DOT_THRESHOLD = 0.92


# ruce
ARM_UP_THRESHOLD = 0.20
ARM_DOWN_THRESHOLD = 0.24

# stage 1 / 3 stani na miste
ANCHOR_POS_THRESHOLD = 0.05
ANCHOR_YAW_THRESHOLD = 0.20

# stage 2
BACK_TARGET_X = -1.60
BACK_TARGET_Y = 0.00
BACK_TARGET_POS_TOL = 0.2
CHAIR_MOVED_MIN_X = 0.30
CHAIR_STILL_THRESHOLD = 0.1
STAGE0_ARM_MIN_ANGLE = -1.3
# =========================================================
# SNAPSHOT CONFIG
# =========================================================

# Pokud True, při startu se načtou snapshoty z disku do RAM bufferu.
ENABLE_DISK_SNAPSHOT_LOAD = False

# Pokud True, nové snapshoty se budou průběžně zapisovat i na disk.
ENABLE_DISK_SNAPSHOT_SAVE = False

SNAPSHOT_DIR = Path("config_run/snapshots_chair/")
MAX_SNAPSHOTS = 10
# Pokud True, všechny envy vždy startují od stage 0
# a snapshot curriculum se zcela ignoruje.
FORCE_START_FROM_STAGE0 = True
RAM_SNAPSHOT_BUFFER = {1: [], 2: [], 3: []}
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
    RAM_SNAPSHOT_BUFFER = {1: [], 2: [], 3: []}

    if not ENABLE_DISK_SNAPSHOT_LOAD:
        BUFFER_INITIALIZED = True
        print("RAM Snapshot Buffer inicializován bez načítání z disku.")
        return

    print("Inicializuji RAM Snapshot Buffer z disku...")
    for stage in range(1, 4):
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
    counts = [len(RAM_SNAPSHOT_BUFFER[s]) for s in range(1, 4)]
    print(f"RAM Buffer načten. Počty snapshotů pro stages 1-3: {counts}")


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


# =========================================================
# HELPERS
# =========================================================

def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _get_pelvis_idx(states: list[EnvState], handler: BaseSimHandler) -> int:
    return states.robots[handler.robot.name].body_names.index("pelvis")


def _get_robot_xy(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    pelvis_idx = _get_pelvis_idx(states, handler)
    return states.robots[handler.robot.name].body_state[idx, pelvis_idx, :2]


def _get_robot_pos(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    pelvis_idx = _get_pelvis_idx(states, handler)
    return states.robots[handler.robot.name].body_state[idx, pelvis_idx, :3]


def _get_robot_lin_vel(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    pelvis_idx = _get_pelvis_idx(states, handler)
    return states.robots[handler.robot.name].body_state[idx, pelvis_idx, 7:10]


def _get_robot_ang_vel(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    pelvis_idx = _get_pelvis_idx(states, handler)
    return states.robots[handler.robot.name].body_state[idx, pelvis_idx, 10:13]


def _get_robot_yaw(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    pelvis_idx = _get_pelvis_idx(states, handler)
    q = states.robots[handler.robot.name].body_state[idx, pelvis_idx, 3:7]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw


def _get_robot_forward_xy(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    pelvis_idx = _get_pelvis_idx(states, handler)
    q = states.robots[handler.robot.name].body_state[idx, pelvis_idx, 3:7]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    forward_x = 1 - 2 * (y**2 + z**2)
    forward_y = 2 * (x * y + w * z)
    forward = torch.stack([forward_x, forward_y], dim=-1)
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-6)
    return forward


def _get_chair_base_idx(states: list[EnvState]) -> int:
    return states.objects["chair"].body_names.index("base_link")


def _get_chair_pos(states: list[EnvState], idx: torch.Tensor) -> torch.Tensor:
    chair_idx = _get_chair_base_idx(states)
    return states.objects["chair"].body_state[idx, chair_idx, :3]


def _get_chair_xy(states: list[EnvState], idx: torch.Tensor) -> torch.Tensor:
    chair_idx = _get_chair_base_idx(states)
    return states.objects["chair"].body_state[idx, chair_idx, :2]


def _get_chair_lin_vel(states: list[EnvState], idx: torch.Tensor) -> torch.Tensor:
    chair_idx = _get_chair_base_idx(states)
    return states.objects["chair"].body_state[idx, chair_idx, 7:10]


def _get_shoulder_indices(states: list[EnvState], handler: BaseSimHandler):
    joint_names = states.robots[handler.robot.name].joint_names
    joint_names = joint_names.tolist()
    left_idx = joint_names.index("left_shoulder_pitch_joint")
    right_idx = joint_names.index("right_shoulder_pitch_joint")
    return left_idx, right_idx


def _arm_pose_error(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor,
                    left_target: float, right_target: float) -> torch.Tensor:
    robot = states.robots[handler.robot.name]
    left_idx, right_idx = _get_shoulder_indices(states, handler)

    q_left = robot.joint_pos[idx, left_idx]
    q_right = robot.joint_pos[idx, right_idx]

    err = 0.5 * (torch.abs(q_left - left_target) + torch.abs(q_right - right_target))
    return err


def _arms_up_ok(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    err = _arm_pose_error(
        states, handler, idx,
        left_target=-1.86,
        right_target=-1.86,
    )
    return err < ARM_UP_THRESHOLD


def _arms_down_ok(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    err = _arm_pose_error(
        states, handler, idx,
        left_target=-1.32,
        right_target=-1.32,
    )
    return err < ARM_DOWN_THRESHOLD


def _robot_is_still(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    lin_vel = _get_robot_lin_vel(states, handler, idx)
    ang_vel = _get_robot_ang_vel(states, handler, idx)

    lin_speed = torch.norm(lin_vel[:, :2], dim=-1)
    yaw_speed = torch.abs(ang_vel[:, 2])

    return (lin_speed < VELOCITY_THRESHOLD) & (yaw_speed < ANGULAR_VELOCITY_THRESHOLD)


def _chair_is_still(states: list[EnvState], idx: torch.Tensor) -> torch.Tensor:
    chair_lin_vel = _get_chair_lin_vel(states, idx)
    chair_speed = torch.norm(chair_lin_vel[:, :2], dim=-1)
    return chair_speed < CHAIR_STILL_THRESHOLD


def _chair_moved_back_enough(states: list[EnvState], idx: torch.Tensor) -> torch.Tensor:
    chair_xy = _get_chair_xy(states, idx)
    # stage0_init ma chair x = 0.75
    moved_x = 0.75 - chair_xy[:, 0]
    return moved_x > CHAIR_MOVED_MIN_X


def _robot_near_back_target(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    robot_xy = _get_robot_xy(states, handler, idx)
    target = torch.tensor([BACK_TARGET_X, BACK_TARGET_Y], device=robot_xy.device)
    dist = torch.norm(robot_xy - target.unsqueeze(0), dim=-1)
    return dist < BACK_TARGET_POS_TOL


def _robot_facing_forward(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    forward = _get_robot_forward_xy(states, handler, idx)
    desired = torch.tensor([1.0, 0.0], device=forward.device).unsqueeze(0).repeat(forward.shape[0], 1)
    dot = torch.sum(forward * desired, dim=-1)
    return dot > FACING_FORWARD_DOT_THRESHOLD


def _robot_near_chair_pregrasp(states: list[EnvState], handler: BaseSimHandler, idx: torch.Tensor) -> torch.Tensor:
    robot_xy = _get_robot_xy(states, handler, idx)
    chair_xy = _get_chair_xy(states, idx)

    dx = torch.abs(robot_xy[:, 0] - chair_xy[:, 0])
    dy = torch.abs(robot_xy[:, 1] - chair_xy[:, 1])

    return (
        (dx >= DISTANCE_TO_CHAIR_X_TARGET - DISTANCE_TO_CHAIR_X_TOL) &
        (dx <= DISTANCE_TO_CHAIR_X_TARGET + DISTANCE_TO_CHAIR_X_TOL) &
        (dy <= DISTANCE_TO_CHAIR_Y_TOL)
    )

# =========================================================
# COMMON CHECKER
# =========================================================

def common_chairman_checker(states: list[EnvState], handler: BaseSimHandler,
                            idx: torch.Tensor, stage_id: int) -> torch.BoolTensor:
    """
    Common fail checker:
    - robot fell
    - timeout in current stage
    """
    num_envs = states.robots[handler.robot.name].joint_pos.shape[0]
    device = idx.device

    if not hasattr(handler.task, "stage_steps"):
        handler.task.stage_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        handler.task.recorded_stage = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    changed_mask = handler.task.recorded_stage[idx] != stage_id
    reset_idx = idx[changed_mask]
    if len(reset_idx) > 0:
        handler.task.stage_steps[reset_idx] = 0

    handler.task.recorded_stage[idx] = stage_id
    handler.task.stage_steps[idx] += 1

    pelvis_idx = _get_pelvis_idx(states, handler)
    pelvis_z = states.robots[handler.robot.name].body_state[idx, pelvis_idx, 2]
    is_fallen = pelvis_z < HEIGHT_THRESHOLD
    if not handler.scenario.dagger == 2:
        limit = STAGE_TIMEOUTS.get(stage_id, 9999)
        is_timeout = handler.task.stage_steps[idx] > limit
    else:
        is_timeout = torch.zeros_like(is_fallen, dtype=torch.bool, device=device)
    return is_fallen | is_timeout

def _arms_dropped_too_low_stage0(states: list[EnvState], handler: BaseSimHandler,
                                 idx: torch.Tensor) -> torch.Tensor:
    """
    Vrací True tam, kde alespoň jedna ruka ve stage 0 klesla pod povolený limit.
    """
    robot = states.robots[handler.robot.name]
    left_idx, right_idx = _get_shoulder_indices(states, handler)

    q_left = robot.joint_pos[idx, left_idx]
    q_right = robot.joint_pos[idx, right_idx]

    return (q_left > STAGE0_ARM_MIN_ANGLE) | (q_right > STAGE0_ARM_MIN_ANGLE)
# =========================================================
# STAGE CHECKERS
# =========================================================

def stege0_chacker(states: list[EnvState], handler: BaseSimHandler,
                   mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Stage 0:
    - dojit k zidli
    - zastavit
    - ruce nahore
    - celne dopredu
    """
    num_envs = mask.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)

    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return terminated, success

    fail_common = common_chairman_checker(states, handler, idx, stage_id=0) #| _arms_dropped_too_low_stage0(states, handler, idx)

    near_chair = _robot_near_chair_pregrasp(states, handler, idx)
    arms_up = _arms_up_ok(states, handler, idx)
    chair_still = _chair_is_still(states, idx)

    success_cond = near_chair & arms_up & chair_still

    terminated[idx] = fail_common | success_cond
    success[idx] = success_cond & (~fail_common)
    return terminated, success


def stege1_chacker(states: list[EnvState], handler: BaseSimHandler,
                   mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Stage 1:
    - zustat na miste
    - ruce dole
    """
    num_envs = mask.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)

    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return terminated, success

    fail_common = common_chairman_checker(states, handler, idx, stage_id=1)

    # anchor logika
    if not hasattr(handler.task, "stage1_anchor_xy"):
        device = mask.device
        n = mask.shape[0]
        handler.task.stage1_anchor_xy = torch.zeros(n, 2, device=device)
        handler.task.stage1_anchor_yaw = torch.zeros(n, device=device)
        handler.task.stage1_anchor_valid = torch.zeros(n, dtype=torch.bool, device=device)

    robot_xy = _get_robot_xy(states, handler, idx)
    robot_yaw = _get_robot_yaw(states, handler, idx)

    invalid = ~handler.task.stage1_anchor_valid[idx]
    if invalid.any():
        set_idx = idx[invalid]
        handler.task.stage1_anchor_xy[set_idx] = robot_xy[invalid]
        handler.task.stage1_anchor_yaw[set_idx] = robot_yaw[invalid]
        handler.task.stage1_anchor_valid[set_idx] = True

    anchor_xy = handler.task.stage1_anchor_xy[idx]
    anchor_yaw = handler.task.stage1_anchor_yaw[idx]

    pos_err = torch.norm(robot_xy - anchor_xy, dim=-1)
    yaw_err = torch.abs(_wrap_to_pi(robot_yaw - anchor_yaw))

    #stay_ok = (pos_err < ANCHOR_POS_THRESHOLD) & (yaw_err < ANCHOR_YAW_THRESHOLD)
    robot_still = _robot_is_still(states, handler, idx)
    arms_down = _arms_down_ok(states, handler, idx)

    success_cond =  robot_still & arms_down #& stay_ok

    terminated[idx] = fail_common | success_cond
    success[idx] = success_cond & (~fail_common)
    return terminated, success


def stege2_chacker(states: list[EnvState], handler: BaseSimHandler,
                   mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Stage 2:
    - ruce dole
    - couvat se zidli
    - dojit na target pozici
    - zastavit
    """
    num_envs = mask.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)

    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return terminated, success

    fail_common = common_chairman_checker(states, handler, idx, stage_id=2)

    robot_near_target = _robot_near_back_target(states, handler, idx)
    robot_still = _robot_is_still(states, handler, idx)
    arms_down = _arms_down_ok(states, handler, idx)
    facing_forward = _robot_facing_forward(states, handler, idx)

    chair_moved = _chair_moved_back_enough(states, idx)
    chair_still = _chair_is_still(states, idx)

    success_cond = (
        robot_near_target &
        robot_still &
        arms_down &
        #facing_forward &
        chair_moved
        #chair_still
    )

    terminated[idx] = fail_common | success_cond
    success[idx] = success_cond & (~fail_common)
    return terminated, success


def stege3_chacker(states: list[EnvState], handler: BaseSimHandler,
                   mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Stage 3:
    - zustat na miste
    - ruce nahore
    - zidle zustane odtazena a v klidu
    """
    num_envs = mask.shape[0]
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=mask.device)

    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return terminated, success

    fail_common = common_chairman_checker(states, handler, idx, stage_id=3)

    if not hasattr(handler.task, "stage3_anchor_xy"):
        device = mask.device
        n = mask.shape[0]
        handler.task.stage3_anchor_xy = torch.zeros(n, 2, device=device)
        handler.task.stage3_anchor_yaw = torch.zeros(n, device=device)
        handler.task.stage3_anchor_valid = torch.zeros(n, dtype=torch.bool, device=device)

    robot_xy = _get_robot_xy(states, handler, idx)
    robot_yaw = _get_robot_yaw(states, handler, idx)

    invalid = ~handler.task.stage3_anchor_valid[idx]
    if invalid.any():
        set_idx = idx[invalid]
        handler.task.stage3_anchor_xy[set_idx] = robot_xy[invalid]
        handler.task.stage3_anchor_yaw[set_idx] = robot_yaw[invalid]
        handler.task.stage3_anchor_valid[set_idx] = True

    anchor_xy = handler.task.stage3_anchor_xy[idx]
    anchor_yaw = handler.task.stage3_anchor_yaw[idx]

    pos_err = torch.norm(robot_xy - anchor_xy, dim=-1)
    yaw_err = torch.abs(_wrap_to_pi(robot_yaw - anchor_yaw))

    stay_ok = (pos_err < ANCHOR_POS_THRESHOLD) & (yaw_err < ANCHOR_YAW_THRESHOLD)
    robot_still = _robot_is_still(states, handler, idx)
    arms_up = _arms_up_ok(states, handler, idx)

    chair_moved = _chair_moved_back_enough(states, idx)
    chair_still = _chair_is_still(states, idx)

    success_cond = stay_ok & robot_still & arms_up & chair_moved# & chair_still

    terminated[idx] = fail_common | success_cond
    success[idx] = success_cond & (~fail_common)
    return terminated, success

def reset_chairman(handler: BaseSimHandler, env_ids: list[int] | None = None):
    global BUFFER_INITIALIZED

    # 1. Inicializace RAM bufferu
    if not BUFFER_INITIALIZED:
        init_ram_buffer()

    states = [stage0_init(handler.robot.name)] * handler.num_envs
    if env_ids is None:
        env_ids = list(range(handler.num_envs))

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
        #print(f"[reset_chairman] FORCE_START_FROM_STAGE0=True -> reset envs {env_ids} vždy do stage 0")

        for env in env_ids:
            for i in range(len(handler.task.reward_functions)):
                handler.task.reward_functions[i].actual_stage[env] = 0
                handler.task.reward_functions[i].completed_stages[env] = 0

            states[env] = stage0_init(handler.robot.name)
            #print(f"[reset_chairman] env {env} -> stage 0")

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
    for i in range(1, 4):
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
            continue
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
        if obj_name == "room":
            formatted_data["objects"][obj_name] = {
                "pos": torch.as_tensor(obj_data["pos"], dtype=torch.float32),
                "rot": torch.as_tensor(obj_data["rot"], dtype=torch.float32),
            }
            continue
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
    if robot_name == "g1_slider_simple":
        state = {
            "robots": {
                "g1_slider_simple": {
                    "dof_pos": {
                        "baseslide_joint": 0.0,
                        "baseslide_joint2": -1.5,
                        "baserot_joint": 0.3,

                        "left_shoulder_pitch_joint": random.uniform(-1.57, -1.0), #-1.2, #random.uniform()
                        "right_shoulder_pitch_joint": random.uniform(-1.57, -1.0), #-1.2, #random.uniform()
                    },
                    "pos": torch.tensor([0.0, 0.0, 0.8]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
            },
            "objects": {
                "chair": {
                    "pos": torch.tensor([0.0, 0.0, 0.1]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "floor_slide_x": 0.75, #random.uniform(0.65,0.75), #0.75,
                        "floor_slide_y": random.uniform(-0.1, 0.1), #0.0
                        "floor_rotate_z": 1.57,
                    },
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
            },

        }
        return state

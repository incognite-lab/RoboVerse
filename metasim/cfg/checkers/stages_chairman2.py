from __future__ import annotations

import pickle
import random
from pathlib import Path
import torch
import threading

from metasim.utils.humanoid_robot_util import neck_height_tensor
from metasim.types import EnvState
from metasim.utils.chair_navigation import (
    CHAIR_FINAL_DISTANCE,
    CHAIR_FINAL_TOLERANCE,
    chair_back_direction_xy,
    forward_direction_xy,
)
try:
    from metasim.sim import BaseSimHandler
except ImportError:
    pass

# All distances are metres, velocities m/s and joint angles radians.
STAGE_TIMEOUTS = {0: 300, 1: 400, 2: 300, 3: 300, 4: 400, 5: 300}
STAGE_TIMEOUT_REFERENCE_DT = 0.02
VELOCITY_THRESHOLD = 0.2
HEIGHT_THRESHOLD = 0.4
FACING_CHAIR_THRESHOLD = 0.90
ROBOT_DRIFT_THRESHOLD = 0.5
CHAIR_DRIFT_THRESHOLD = 0.05
CHAIR_PULL_DISTANCE_THRESHOLD = 1.0
CHAIR_PULL_TOLERANCE = 0.10
PALM_FRONT_OFFSET = 0.10
PALM_POSITION_TOLERANCE = 0.05
PALM_LIFT_HEIGHT = 0.10
PALM_LIFT_XY_TOLERANCE = 0.15
STAGE1_HOLD_STEPS = 5
STAGE4_HOLD_STEPS = 5

# Editable example poses for G1. Every listed joint must meet its tolerance;
# legs are deliberately omitted so that the walking controller can balance.
STAGE0_JOINT_TARGETS = {
    "left_shoulder_pitch_joint": 1.51,
    "left_shoulder_roll_joint": 0.93,
    "left_shoulder_yaw_joint": 1.15,
    "left_elbow_joint": -0.59,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 1.51,
    "right_shoulder_roll_joint": -0.93,
    "right_shoulder_yaw_joint": -1.15,
    "right_elbow_joint": -0.59,
    "right_wrist_roll_joint": 0.0,

}
STAGE2_JOINT_TARGETS = {
    "left_shoulder_pitch_joint": -1.66,
    "left_shoulder_roll_joint": 0.23,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.45,
    "left_wrist_roll_joint": 1.35,
    "right_shoulder_pitch_joint": -1.66,
    "right_shoulder_roll_joint": -0.23,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.45,
    "right_wrist_roll_joint": -1.35,
}
JOINT_POSITION_TOLERANCE = 0.2
# Optional overrides, e.g. {"left_elbow_joint": 0.10}.
STAGE0_JOINT_TOLERANCES = {}
STAGE2_JOINT_TOLERANCES = {}


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
ENABLE_DISK_SNAPSHOT_LOAD = True

# Pokud True, nové snapshoty se budou průběžně zapisovat i na disk.
ENABLE_DISK_SNAPSHOT_SAVE = True

# These stage meanings differ from stages_chairman; never reuse its snapshots.
SNAPSHOT_DIR = Path("config_run/snapshots_chairman2/")
MAX_SNAPSHOTS = 100
# Pokud True, všechny envy vždy startují od stage 0
# a snapshot curriculum se zcela ignoruje.
FORCE_START_FROM_STAGE0 = False
RAM_SNAPSHOT_BUFFER = {1: [], 2: [], 3: [], 4: [], 5: []}
BUFFER_INITIALIZED = False
SNAPSHOT_BUFFER_VERSION = 0
UNSAVED_COUNT = 0
SYNC_THRESHOLD = 30  # Každých 50 uložených snapshotů se jeden zapíše trvale na disk
LOCK = threading.Lock()

def init_ram_buffer():
    """Inicializuje RAM snapshot buffer. Volitelně načte snapshoty z disku."""
    global BUFFER_INITIALIZED, RAM_SNAPSHOT_BUFFER, SNAPSHOT_BUFFER_VERSION

    if BUFFER_INITIALIZED:
        return

    # Vždy začneme s čistým RAM bufferem
    RAM_SNAPSHOT_BUFFER = {1: [], 2: [], 3: [], 4: [], 5: []}

    if not ENABLE_DISK_SNAPSHOT_LOAD:
        BUFFER_INITIALIZED = True
        SNAPSHOT_BUFFER_VERSION += 1
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
    SNAPSHOT_BUFFER_VERSION += 1
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

def _body(states, handler, idx):
    robot = states.robots[handler.robot.name]
    chair = states.objects["chair"]
    return (robot.body_state[idx, robot.body_names.index("pelvis")],
            chair.body_state[idx, chair.body_names.index("base_link")])


def _capture_stage_reference(states, handler, idx):
    """Record independent anchors for each environment on stage entry/reset."""
    robot, chair = _body(states, handler, idx)
    n = states.robots[handler.robot.name].joint_pos.shape[0]
    for name, value in (("chairman_robot_anchor", robot[:, :2]),
                        ("chairman_chair_anchor", chair[:, :3]),
                        ("chairman_pull_direction", chair_back_direction_xy(chair[:, 3:7]))):
        if not hasattr(handler.task, name):
            setattr(handler.task, name, value.new_zeros((n, value.shape[1])))
        getattr(handler.task, name)[idx] = value


def check_movement_chair(states, handler, idx):
    _, chair = _body(states, handler, idx)
    return torch.linalg.vector_norm(
        chair[:, :3] - handler.task.chairman_chair_anchor[idx], dim=-1
    ) > CHAIR_DRIFT_THRESHOLD


def _robot_moved(states, handler, idx):
    robot, _ = _body(states, handler, idx)
    return torch.linalg.vector_norm(
        robot[:, :2] - handler.task.chairman_robot_anchor[idx], dim=-1
    ) > ROBOT_DRIFT_THRESHOLD


def _joint_pose_matches(states, handler, idx, targets, tolerances):
    if not targets:
        raise ValueError("A joint pose must contain at least one target joint")
    robot = states.robots[handler.robot.name]
    names = list(robot.joint_names)
    missing = set(targets) - set(names)
    if missing:
        raise ValueError(f"Target joints missing from robot: {sorted(missing)}")
    desired = robot.joint_pos.new_tensor(list(targets.values()))
    tolerance = robot.joint_pos.new_tensor([
        tolerances.get(name, JOINT_POSITION_TOLERANCE) for name in targets
    ])
    if not torch.isfinite(desired).all() or not torch.isfinite(tolerance).all() or (tolerance < 0).any():
        raise ValueError("Joint targets and tolerances must be finite; tolerances must be nonnegative")
    actual = robot.joint_pos[idx][:, [names.index(name) for name in targets]]
    return (torch.abs(actual - desired) <= tolerance).all(dim=-1)


def _palm_targets(states, handler, idx):
    robot = states.robots[handler.robot.name]
    chair = states.objects["chair"]
    palms = torch.stack([
        robot.body_state[idx, robot.body_names.index(f"{side}_hand_palm_link"), :3]
        for side in ("left", "right")
    ], dim=1)
    targets = torch.stack([
        chair.body_state[idx, chair.body_names.index(f"target_hand_{side}"), :3]
        for side in ("left", "right")
    ], dim=1)
    return palms, targets


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
        _capture_stage_reference(states, handler, reset_idx)
        for name in ("stage1_success_steps", "stage4_success_steps"):
            counter = getattr(handler.task, name, None)
            if counter is not None:
                counter[reset_idx] = 0

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

def _check_stage(states, handler, mask, stage_id):
    """Return full-size (terminated, success) masks; failures take precedence."""
    terminated = torch.zeros_like(mask)
    success = torch.zeros_like(mask)
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return terminated, success

    fail = common_chairman_checker(states, handler, idx, stage_id)
    robot, chair = _body(states, handler, idx)
    if stage_id in (0, 2, 3):
        fail |= _robot_moved(states, handler, idx)
    if stage_id in (1, 2):
        fail |= check_movement_chair(states, handler, idx)

    if stage_id in (0, 2):
        targets = STAGE0_JOINT_TARGETS if stage_id == 0 else STAGE2_JOINT_TARGETS
        tolerances = STAGE0_JOINT_TOLERANCES if stage_id == 0 else STAGE2_JOINT_TOLERANCES
        reached = _joint_pose_matches(states, handler, idx, targets, tolerances)
    elif stage_id == 1:
        reached = _walking_success(states, handler, idx, fail)
    elif stage_id == 3:
        # Offset towards the robot (chair local +Y), independent of world yaw.
        palms, targets = _palm_targets(states, handler, idx)
        targets[:, :, :2] += PALM_FRONT_OFFSET * chair_back_direction_xy(chair[:, 3:7])[:, None, :]
        reached = (torch.linalg.vector_norm(palms - targets, dim=-1) <= PALM_POSITION_TOLERANCE).all(dim=-1)
    elif stage_id == 4:
        displacement = chair[:, :2] - handler.task.chairman_chair_anchor[idx, :2]
        direction = handler.task.chairman_pull_direction[idx]
        distance = (displacement * direction).sum(dim=-1)
        lateral = torch.linalg.vector_norm(displacement - distance[:, None] * direction, dim=-1)
        reached = (
            ((distance - CHAIR_PULL_DISTANCE_THRESHOLD).abs() <= CHAIR_PULL_TOLERANCE)
            & (lateral <= CHAIR_PULL_TOLERANCE)
            & (torch.linalg.vector_norm(robot[:, 7:10], dim=-1) < VELOCITY_THRESHOLD)
            & (torch.linalg.vector_norm(chair[:, 7:10], dim=-1) < VELOCITY_THRESHOLD)
        )
        reached = _held_condition(handler, "stage4_success_steps", idx,
                                  reached & ~fail, STAGE4_HOLD_STEPS)
    else:  # Stage 5: both palms above their backrest targets.
        palms, targets = _palm_targets(states, handler, idx)
        reached = (
            (palms[:, :, 2] >= targets[:, :, 2] + PALM_LIFT_HEIGHT)
            & (torch.linalg.vector_norm(palms[:, :, :2] - targets[:, :, :2], dim=-1)
               <= PALM_LIFT_XY_TOLERANCE)
        ).all(dim=-1)

    success[idx] = reached & ~fail
    terminated[idx] = fail | reached
    return terminated, success


def _walking_success(states, handler, idx, term_common):
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
    success_now &= ~term_common
    success_cond = _held_condition(
        handler, "stage1_success_steps", idx, success_now, STAGE1_HOLD_STEPS
    )
    return success_cond


def stege0_chacker(states: EnvState, handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Stage 0: walking arm pose in place."""
    return _check_stage(states, handler, mask, 0)

def stege1_chacker(states: EnvState, handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Stage 1: walk to the chair and stop."""
    return _check_stage(states, handler, mask, 1)

def stege2_chacker(states: EnvState, handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Stage 2: forward arm pose in place."""
    return _check_stage(states, handler, mask, 2)

def stege3_chacker(states: EnvState, handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Stage 3: palms 10 cm in front of the backrest, in place."""
    return _check_stage(states, handler, mask, 3)

def stege4_chacker(states: EnvState, handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Stage 4: pull the chair back one metre and stop both bodies."""
    return _check_stage(states, handler, mask, 4)

def stege5_chacker(states: EnvState, handler: BaseSimHandler, mask: torch.BoolTensor) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Stage 5: lift both palms above the backrest."""
    return _check_stage(states, handler, mask, 5)

def _repeat_packed_state(packed, count: int):
    """Create a writable reset batch from a cached one-row GPU template."""
    return {
        obj_name: {
            key: value.expand((count,) + value.shape[1:]).clone()
            for key, value in entity.items()
        }
        for obj_name, entity in packed.items()
    }


def _cached_stage0_batch(handler: BaseSimHandler, env_ids: torch.Tensor):
    """Return cached stage 0 tensors, ready for future domain randomization."""
    cache_key = (handler.robot.name, str(handler.device))
    cache = getattr(handler, "_chairman_stage0_state_cache", None)
    if cache is None or cache[0] != cache_key:
        cache = (cache_key, handler.pack_state_batch([stage0_init(handler.robot.name)]))
        handler._chairman_stage0_state_cache = cache

    batch = _repeat_packed_state(cache[1], env_ids.numel())
    randomizer = getattr(handler.task, "randomize_chairman_reset_batch", None)
    if callable(randomizer):
        # A future randomizer can modify root poses, qpos and qvel in place;
        # all tensors are already on the simulator device.
        randomizer(batch, env_ids=env_ids)
    return batch


def _snapshot_tensor_banks(handler: BaseSimHandler, max_stage: int):
    """Lazily convert loaded snapshot dictionaries to reusable GPU tensors."""
    cache = getattr(handler, "_chairman_snapshot_tensor_cache", None)
    if cache is not None and cache[0] == SNAPSHOT_BUFFER_VERSION:
        return cache[1]

    banks = {
        stage: handler.pack_state_batch(RAM_SNAPSHOT_BUFFER[stage])
        for stage in range(1, max_stage + 1)
        if RAM_SNAPSHOT_BUFFER[stage]
    }
    handler._chairman_snapshot_tensor_cache = (SNAPSHOT_BUFFER_VERSION, banks)
    return banks


def _copy_packed_rows(destination, source, row_ids, source_ids):
    """Scatter selected snapshot rows into a reset batch on the GPU."""
    for obj_name, destination_entity in destination.items():
        source_entity = source.get(obj_name)
        if source_entity is None:
            continue
        for key, destination_value in destination_entity.items():
            source_value = source_entity.get(key)
            if source_value is not None:
                destination_value[row_ids] = source_value.index_select(0, source_ids)


def _reset_chairman_legacy(
    handler: BaseSimHandler,
    env_ids: torch.Tensor,
    current_stages: torch.Tensor,
    completed_stages: torch.Tensor,
    reset_to_stage0: bool,
    requested_stage: int | None,
):
    """Compatibility path for simulators without the Genesis tensor API."""
    cpu_ids = env_ids.detach().cpu().tolist()
    stage0_state = stage0_init(handler.robot.name)
    states = [stage0_state] * handler.num_envs
    max_available_stage = 0
    if not reset_to_stage0:
        for stage in range(1, 6):
            if RAM_SNAPSHOT_BUFFER[stage]:
                max_available_stage = stage
            else:
                break

    selected_stages = []
    for env_id in cpu_ids:
        if requested_stage is not None:
            stage = requested_stage
        else:
            stage = 0 if reset_to_stage0 else random.randint(0, max_available_stage)
        state = load_snapshot_chairman(stage) if stage > 0 else None
        if state is None:
            stage = 0
            state = stage0_state
        selected_stages.append(stage)
        states[env_id] = state

    selected_stages_tensor = torch.as_tensor(
        selected_stages, dtype=torch.long, device=handler.device
    )
    current_stages.index_copy_(0, env_ids, selected_stages_tensor)
    completed_stages.index_fill_(0, env_ids, 0)
    handler.set_states(states=states, env_ids=cpu_ids)
    return handler.get_states()


def reset_chairman(
    handler: BaseSimHandler,
    env_ids: list[int] | torch.Tensor | None = None,
):
    global BUFFER_INITIALIZED

    if not BUFFER_INITIALIZED:
        init_ram_buffer()

    if env_ids is None:
        env_ids = torch.arange(handler.num_envs, dtype=torch.long, device=handler.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=handler.device)
    else:
        env_ids = env_ids.to(device=handler.device, dtype=torch.long)
    env_ids = env_ids.flatten()
    reset_count = env_ids.numel()
    if reset_count == 0:
        return

    for counter_name in (
        "stage0_success_steps", "stage1_success_steps", "stage2_success_steps",
        "stage3_success_steps", "stage4_success_steps",
    ):
        counter = getattr(handler.task, counter_name, None)
        if counter is not None:
            counter.index_fill_(0, env_ids, 0)

    reward_functions = handler.task.reward_functions
    current_stages = reward_functions[0].actual_stage
    if current_stages is None:
        current_stages = torch.zeros(handler.num_envs, dtype=torch.long, device=handler.device)
        completed_stages = torch.zeros_like(current_stages)
    else:
        completed_stages = reward_functions[0].completed_stages

    # Rewards share these tensors.  Only one indexed write is then needed per
    # reset instead of one small GPU operation per reward and environment.
    for reward_fn in reward_functions:
        reward_fn.actual_stage = current_stages
        reward_fn.completed_stages = completed_stages

    use_snapshot_curriculum = bool(getattr(handler.task, "use_snapshot_curriculum", True))
    requested_stage = getattr(handler.task, "eval_start_stage", None)
    if requested_stage is not None:
        if isinstance(requested_stage, bool) or not isinstance(requested_stage, int):
            raise ValueError(
                f"eval_start_stage must be an integer from 0 to 5, got {requested_stage!r}"
            )
        if not 0 <= requested_stage <= 5:
            raise ValueError(
                f"eval_start_stage must be between 0 and 5, got {requested_stage}"
            )
        if requested_stage > 0 and not RAM_SNAPSHOT_BUFFER[requested_stage]:
            stage_dir = SNAPSHOT_DIR / f"stage_{requested_stage}"
            raise RuntimeError(
                f"Cannot start evaluation from stage {requested_stage}: no snapshot is available. "
                f"Expected snapshots in {stage_dir}."
            )

    reset_to_stage0 = (
        requested_stage == 0
        if requested_stage is not None
        else (
            FORCE_START_FROM_STAGE0
            or bool(getattr(handler.task, "reset_to_stage0", False))
            or not use_snapshot_curriculum
        )
    )

    if not hasattr(handler, "pack_state_batch") or not hasattr(handler, "set_packed_state_batch"):
        states = _reset_chairman_legacy(
            handler,
            env_ids,
            current_stages,
            completed_stages,
            reset_to_stage0,
            requested_stage,
        )
        if hasattr(handler.task, "recorded_stage"):
            handler.task.recorded_stage.index_fill_(0, env_ids, -1)
        for reward_fn in reward_functions:
            if hasattr(reward_fn, "reset"):
                reward_fn.reset(env_ids=env_ids, states=states)
        return

    reset_batch = _cached_stage0_batch(handler, env_ids)

    if reset_to_stage0:
        new_stages = torch.zeros(reset_count, dtype=torch.long, device=handler.device)
    else:
        if requested_stage is not None:
            max_available_stage = requested_stage
        else:
            max_available_stage = 0
            for stage in range(1, 6):
                if RAM_SNAPSHOT_BUFFER[stage]:
                    max_available_stage = stage
                else:
                    break

        if requested_stage is None:
            new_stages = torch.randint(
                0, max_available_stage + 1, (reset_count,), device=handler.device
            )
        else:
            new_stages = torch.full(
                (reset_count,), requested_stage, dtype=torch.long, device=handler.device
            )
        snapshot_banks = _snapshot_tensor_banks(handler, max_available_stage)
        for stage in range(1, max_available_stage + 1):
            row_ids = (new_stages == stage).nonzero(as_tuple=False).flatten()
            bank = snapshot_banks.get(stage)
            if row_ids.numel() == 0 or bank is None:
                continue
            first_entity = next(iter(bank.values()))
            bank_size = next(iter(first_entity.values())).shape[0]
            source_ids = torch.randint(
                0, bank_size, (row_ids.numel(),), device=handler.device
            )
            _copy_packed_rows(reset_batch, bank, row_ids, source_ids)

    current_stages.index_copy_(0, env_ids, new_stages)
    completed_stages.index_fill_(0, env_ids, 0)
    if hasattr(handler.task, "recorded_stage"):
        handler.task.recorded_stage.index_fill_(0, env_ids, -1)

    handler.set_packed_state_batch(reset_batch, env_ids=env_ids)
    states = handler.get_states()
    for reward_fn in reward_functions:
        if hasattr(reward_fn, "reset"):
            reward_fn.reset(env_ids=env_ids, states=states)



def _store_snapshot(stage: int, snapshot_data: dict) -> int:
    """Insert one already CPU-resident snapshot into the curriculum buffer."""
    global UNSAVED_COUNT, SNAPSHOT_BUFFER_VERSION
    with LOCK:
        if len(RAM_SNAPSHOT_BUFFER[stage]) < MAX_SNAPSHOTS:
            RAM_SNAPSHOT_BUFFER[stage].append(snapshot_data)
            idx = len(RAM_SNAPSHOT_BUFFER[stage]) - 1
        else:
            idx = random.randint(0, MAX_SNAPSHOTS - 1)
            RAM_SNAPSHOT_BUFFER[stage][idx] = snapshot_data

        SNAPSHOT_BUFFER_VERSION += 1
        UNSAVED_COUNT += 1
        trigger_sync = False
        if ENABLE_DISK_SNAPSHOT_SAVE and UNSAVED_COUNT >= SYNC_THRESHOLD:
            trigger_sync = True
            UNSAVED_COUNT = 0

    # Volitelný zápis na disk
    if trigger_sync:
        thread = threading.Thread(target=_sync_to_disk_worker, args=(stage, snapshot_data, idx))
        thread.start()
    return idx


def _update_snapshot_tensor_cache(handler, stage: int, index: int, snapshot_data: dict, old_version: int):
    """Incrementally mirror one CPU reservoir update into an existing GPU bank."""
    cache = getattr(handler, "_chairman_snapshot_tensor_cache", None)
    if cache is None or cache[0] != old_version:
        return

    banks = cache[1]
    packed_row = handler.pack_state_batch([snapshot_data])
    bank = banks.get(stage)
    if bank is None:
        banks[stage] = packed_row
    else:
        for obj_name, row_entity in packed_row.items():
            if obj_name not in bank:
                bank[obj_name] = row_entity
                continue
            for key, row_value in row_entity.items():
                bank_value = bank[obj_name].get(key)
                if bank_value is None:
                    bank[obj_name][key] = row_value
                elif index == bank_value.shape[0]:
                    bank[obj_name][key] = torch.cat((bank_value, row_value), dim=0)
                else:
                    bank_value[index] = row_value[0]
    handler._chairman_snapshot_tensor_cache = (SNAPSHOT_BUFFER_VERSION, banks)


def save_snapshots_chairman(
    handler: BaseSimHandler,
    env_ids: torch.Tensor,
    stages: torch.Tensor,
) -> list[tuple[int, int]]:
    """Capture selected environments with one batched transfer per state field.

    Snapshot dictionaries intentionally live on CPU because resets and optional
    pickle persistence consume them there.  Batching avoids dozens of CUDA
    synchronizations for every successful environment.
    """
    if FORCE_START_FROM_STAGE0 or not bool(
        getattr(handler.task, "use_snapshot_curriculum", True)
    ):
        return []
    env_ids = env_ids.to(device=handler.device, dtype=torch.long).flatten()
    stages = stages.to(device=handler.device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return []
    env_ids_cpu = env_ids.detach().cpu().tolist()
    stages_cpu = stages.detach().cpu().tolist()
    full_states = handler.get_states()

    robot_name = handler.robot.name
    robot_state = full_states.robots[robot_name]
    robot_joint_names = robot_state.joint_names.tolist()
    robot_root = robot_state.root_state.index_select(0, env_ids).detach().cpu()
    robot_q = robot_state.joint_pos.index_select(0, env_ids).detach().cpu().numpy()
    robot_dq = robot_state.joint_vel.index_select(0, env_ids).detach().cpu().numpy()

    object_batches = {}
    for obj_name, obj_state in full_states.objects.items():
        object_batches[obj_name] = (
            obj_state.joint_names.tolist(),
            obj_state.root_state.index_select(0, env_ids).detach().cpu(),
            obj_state.joint_pos.index_select(0, env_ids).detach().cpu().numpy(),
            obj_state.joint_vel.index_select(0, env_ids).detach().cpu().numpy(),
        )

    saved = []
    for row, (env_id, stage) in enumerate(zip(env_ids_cpu, stages_cpu)):
        stage = int(stage)
        if stage not in RAM_SNAPSHOT_BUFFER:
            continue
        snapshot_data = {"robots": {}, "objects": {}}
        snapshot_data["robots"][robot_name] = {
            "pos": robot_root[row, :3].clone(),
            "rot": robot_root[row, 3:7].clone(),
            "dof_pos": dict(zip(robot_joint_names, robot_q[row])),
            "dof_vel": dict(zip(robot_joint_names, robot_dq[row])),
        }
        for obj_name, (joint_names, root, joint_pos, joint_vel) in object_batches.items():
            obj_data = {
                "pos": root[row, :3].clone(),
                "rot": root[row, 3:7].clone(),
                "dof_pos": dict(zip(joint_names, joint_pos[row])),
                "dof_vel": dict(zip(joint_names, joint_vel[row])),
            }
            snapshot_data["objects"][obj_name] = obj_data
        old_version = SNAPSHOT_BUFFER_VERSION
        snapshot_index = _store_snapshot(stage, snapshot_data)
        if hasattr(handler, "pack_state_batch"):
            _update_snapshot_tensor_cache(
                handler, stage, snapshot_index, snapshot_data, old_version
            )
        saved.append((int(env_id), stage))
    return saved


def save_snapshot_chairman(handler: BaseSimHandler, env_id: int, stage: int) -> None:
    """Backward-compatible single-environment snapshot API."""
    save_snapshots_chairman(
        handler,
        torch.tensor([env_id], dtype=torch.long, device=handler.device),
        torch.tensor([stage], dtype=torch.long, device=handler.device),
    )

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
    elif robot_name == "g1_without_hands":
        state = {
            "robots": {
                "g1_without_hands": {
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
                        "left_shoulder_pitch_joint": 0.28,
                        "left_shoulder_roll_joint": 0.35,
                        "left_shoulder_yaw_joint": 0.0,
                        "left_elbow_joint": 0.77,
                        "left_wrist_roll_joint": 0.0,
                        "left_wrist_pitch_joint": 0.0,
                        "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.28,
                        "right_shoulder_roll_joint": -0.35,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.77,
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

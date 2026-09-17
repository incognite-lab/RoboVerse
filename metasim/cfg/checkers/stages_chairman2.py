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

# Shared criteria also drive reward shaping; all thresholds use SI units.
from metasim.utils import chairman2_geometry as geometry
from metasim.utils.chairman2_geometry import STAGE0_JOINT_TARGETS, STAGE1_JOINT_TARGETS
NUM_STAGES = geometry.NUM_STAGES
STAGE_TIMEOUT_REFERENCE_DT = 0.02
STAGE_TIMEOUTS = {0: 750, 1: 300, 2: 400, 3: 1000, 4: 300}
HOLD_SECONDS = (0.25, 0.25, 0.25, 0.40, 0.25)
CONTACT_GRACE_SECONDS = 0.10


def _capture_stage_reference(states, handler, idx):
    robot = states.robots[handler.robot.name]
    base = robot.body_state[idx, robot.body_names.index('pelvis')]
    chair = states.objects['chair']
    chair_body = chair.body_state[idx, chair.body_names.index('base_link')]
    n = robot.joint_pos.shape[0]
    direction = chair_back_direction_xy(chair_body[:, 3:7])
    for name, value in (('chairman_robot_anchor', base[:, :2]),
                        ('chairman_chair_anchor', chair_body[:, :3]),
                        ('chairman_pull_direction', direction),
                        ('chairman_chair_heading', direction)):
        if not hasattr(handler.task, name):
            setattr(handler.task, name, value.new_zeros((n, value.shape[1])))
        getattr(handler.task, name)[idx] = value


def begin_stage(states, handler, ids, stages):
    """Anchor a new stage before its first action, including partial resets."""
    task = handler.task
    _capture_stage_reference(states, handler, ids)
    for name in ('stage_steps', 'success_hold_steps', 'contact_loss_steps'):
        if not hasattr(task, name):
            setattr(task, name, torch.zeros_like(stages))
        getattr(task, name)[ids] = 0
    if not hasattr(task, 'recorded_stage'):
        task.recorded_stage = torch.full_like(stages, -1)
    task.recorded_stage[ids] = stages[ids]


def success_conditions(m):
    """Instantaneous conjunctions; the checker additionally requires a hold."""
    still = (m['robot_speed'] <= geometry.STILL_SPEED) & (m['robot_yaw_speed'] <= geometry.STILL_YAW_SPEED)
    chair_still = (m['chair_speed'] <= geometry.STILL_SPEED) & (m['chair_yaw_speed'] <= geometry.STILL_YAW_SPEED)
    facing = m['heading'] <= geometry.HEADING_TOLERANCE
    anchored = (m['robot_drift'] <= geometry.ROBOT_DRIFT_TOLERANCE) & (m['chair_drift'] <= geometry.CHAIR_DRIFT_TOLERANCE)
    chair_unturned = m['chair_yaw'] <= geometry.HEADING_TOLERANCE
    hands_ready = (m['palm_angle'] <= geometry.PALM_ANGLE_TOLERANCE).all(-1) & (m['elbow_angle'] <= geometry.ELBOW_ANGLE_TOLERANCE).all(-1)
    contact = m['contact'].all(-1) & (m['contact_force'] <= geometry.CONTACT_FORCE_MAX).all(-1)
    quiet_hands = m['hand_slip'].amax(-1) <= 0.08
    return torch.stack((
        (m['approach_error'] <= geometry.POSITION_TOLERANCE) & (m['pose0'] <= geometry.JOINT_TOLERANCE).all(-1)
            & facing & still & chair_still & (m['chair_drift'] <= geometry.CHAIR_DRIFT_TOLERANCE) & chair_unturned,
        (m['pose1'] <= geometry.JOINT_TOLERANCE).all(-1) & anchored & still & chair_still & facing
            & chair_unturned & (m['arm_speed'] <= 0.20),
        hands_ready & contact & anchored & still & chair_still & facing & quiet_hands & chair_unturned,
        (m['pull_error'] <= geometry.PULL_TOLERANCE) & (m['lateral'] <= geometry.PULL_TOLERANCE)
            & hands_ready & contact & still & chair_still & facing & chair_unturned & quiet_hands,
        (m['lift_height'] >= 0.10).all(-1) & (m['lift_xy'] <= 0.10).all(-1) & ~m['any_contact'].any(-1)
            & anchored & still & chair_still & facing & chair_unturned & quiet_hands,
    ), dim=-1)


def evaluate_stages(states, handler, stages):
    """One physical measurement pass, per-row timers and independent anchors."""
    import math
    task = handler.task
    n, device = stages.shape[0], stages.device
    if not hasattr(task, 'stage_steps'):
        task.stage_steps = torch.zeros(n, dtype=torch.long, device=device)
        task.recorded_stage = torch.full_like(stages, -1)
    for name in ('success_hold_steps', 'contact_loss_steps'):
        if not hasattr(task, name):
            setattr(task, name, torch.zeros_like(stages))
    changed = task.recorded_stage != stages
    ids = changed.nonzero(as_tuple=True)[0]
    if ids.numel():
        _capture_stage_reference(states, handler, ids)
        task.stage_steps[ids] = 0
        task.success_hold_steps[ids] = 0
        task.contact_loss_steps[ids] = 0
    task.recorded_stage.copy_(stages)
    task.stage_steps += 1
    dt = (handler.scenario.sim_params.dt or 0.002) * handler.scenario.decimation
    m = geometry.measure(states, handler.robot.name, task)
    task.chairman2_metrics = m
    safe_stage = stages.clamp(0, NUM_STAGES-1)
    limits = stages.new_tensor([max(1, math.ceil(STAGE_TIMEOUTS[s]*STAGE_TIMEOUT_REFERENCE_DT/dt)) for s in range(NUM_STAGES)])
    failed = (stages < 0) | (stages >= NUM_STAGES) | ~m['finite'] | (task.stage_steps > limits[safe_stage])
    failed |= (neck_height_tensor(states, handler.robot.name) < 0.4) | (m['upright'] < 0.5)
    stationary_stage = (stages == 1) | (stages == 2) | (stages == 4)
    # Tolerance errors receive shaping; large deviations terminate the attempt.
    failed |= stationary_stage & (m['robot_drift'] > 0.25)
    failed |= (stages != 3) & (m['chair_drift'] > 0.15)
    failed |= (stages == 3) & ((m['lateral'] > 0.25) | (m['chair_yaw'] > math.radians(25)))
    lost = (stages == 3) & ~m['contact'].all(-1)
    task.contact_loss_steps = torch.where(lost, task.contact_loss_steps+1, torch.zeros_like(stages))
    failed |= (stages == 3) & (task.contact_loss_steps > max(1, math.ceil(CONTACT_GRACE_SECONDS/dt)))
    reached = success_conditions(m).gather(1, safe_stage[:, None]).squeeze(1) & ~failed
    task.success_hold_steps = torch.where(reached, task.success_hold_steps+1, torch.zeros_like(stages))
    holds = stages.new_tensor([max(1, math.ceil(seconds/dt)) for seconds in HOLD_SECONDS])
    succeeded = (task.success_hold_steps >= holds[safe_stage]) & ~failed
    return failed, succeeded


# =========================================================
# SNAPSHOT CONFIG
# =========================================================

# Pokud True, při startu se načtou snapshoty z disku do RAM bufferu.
ENABLE_DISK_SNAPSHOT_LOAD = True

# Pokud True, nové snapshoty se budou průběžně zapisovat i na disk.
ENABLE_DISK_SNAPSHOT_SAVE = True

# These stage meanings differ from stages_chairman; never reuse its snapshots.
SNAPSHOT_DIR = Path("config_run/snapshots_chairman2_five_stage_v1/")
MAX_SNAPSHOTS = 100
# Pokud True, všechny envy vždy startují od stage 0
# a snapshot curriculum se zcela ignoruje.
FORCE_START_FROM_STAGE0 = False
RAM_SNAPSHOT_BUFFER = {stage: [] for stage in range(1, NUM_STAGES)}
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
    RAM_SNAPSHOT_BUFFER = {stage: [] for stage in range(1, NUM_STAGES)}

    if not ENABLE_DISK_SNAPSHOT_LOAD:
        BUFFER_INITIALIZED = True
        SNAPSHOT_BUFFER_VERSION += 1
        print("RAM Snapshot Buffer inicializován bez načítání z disku.")
        return

    print("Inicializuji RAM Snapshot Buffer z disku...")
    for stage in range(1, NUM_STAGES):
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
    counts = [len(RAM_SNAPSHOT_BUFFER[s]) for s in range(1, NUM_STAGES)]
    print(f"RAM Buffer načten. Počty snapshotů pro stages 1-4: {counts}")


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
        banks = cache[1]
        # Curriculum can unlock more stages without changing the reservoir.
        for stage in range(1, max_stage + 1):
            if stage not in banks and RAM_SNAPSHOT_BUFFER[stage]:
                banks[stage] = handler.pack_state_batch(RAM_SNAPSHOT_BUFFER[stage])
        return banks

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
        for stage in range(1, NUM_STAGES):
            if RAM_SNAPSHOT_BUFFER[stage]:
                max_available_stage = stage
            else:
                break

    cap = getattr(handler.task, "curriculum_max_stage", None)
    if cap is not None:
        max_available_stage = min(max_available_stage, int(cap))

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
        "success_hold_steps", "contact_loss_steps", "stage2_success_steps",
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
    training_stage = getattr(handler.task, "train_stage", None)
    if training_stage is not None:
        requested_stage = training_stage
    stage_option = "train_stage" if training_stage is not None else "eval_start_stage"
    if requested_stage is not None:
        if isinstance(requested_stage, bool) or not isinstance(requested_stage, int):
            raise ValueError(
                f"{stage_option} must be an integer from 0 to 4, got {requested_stage!r}"
            )
        if not 0 <= requested_stage < NUM_STAGES:
            raise ValueError(
                f"{stage_option} must be between 0 and 4, got {requested_stage}"
            )
        if requested_stage > 0 and not RAM_SNAPSHOT_BUFFER[requested_stage]:
            stage_dir = SNAPSHOT_DIR / f"stage_{requested_stage}"
            raise RuntimeError(
                f"Cannot start from stage {requested_stage} ({stage_option}): no snapshot is available. "
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
        begin_stage(states, handler, env_ids, current_stages)
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
            for stage in range(1, NUM_STAGES):
                if RAM_SNAPSHOT_BUFFER[stage]:
                    max_available_stage = stage
                else:
                    break

        if requested_stage is None:
            cap = getattr(handler.task, "curriculum_max_stage", None)
            if cap is not None:
                max_available_stage = min(max_available_stage, int(cap))
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
    begin_stage(states, handler, env_ids, current_stages)
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
    bank = banks.get(stage)
    if bank is None:
        # A stage omitted by the curriculum may already have many CPU rows.
        # Do not create a one-row bank with a reservoir index larger than zero.
        banks[stage] = handler.pack_state_batch(RAM_SNAPSHOT_BUFFER[stage])
    else:
        packed_row = handler.pack_state_batch([snapshot_data])
        expected_size = len(RAM_SNAPSHOT_BUFFER[stage])
        sizes = {value.shape[0] for entity in bank.values() for value in entity.values()}
        append = sizes == {expected_size - 1} and index == expected_size - 1
        replace = sizes == {expected_size} and 0 <= index < expected_size
        same_fields = bank.keys() == packed_row.keys() and all(
            bank[name].keys() == entity.keys() for name, entity in packed_row.items()
        )
        if not same_fields or not (append or replace):
            banks[stage] = handler.pack_state_batch(RAM_SNAPSHOT_BUFFER[stage])
            handler._chairman_snapshot_tensor_cache = (SNAPSHOT_BUFFER_VERSION, banks)
            return
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

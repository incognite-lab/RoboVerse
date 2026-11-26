"""Opt-in recurrent DAgger collector, sequence trainer and evaluations."""
from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from loguru import logger as log

from .buffer import SequenceBuffer, train_sequence_step
from .checkpoint import (
    IMU_LAYOUT, atomic_save, control_dt, evaluation_path, load_student, model_kwargs,
    output_directory, restore_optimizer, save_checkpoint,
)
from .inputs import IMU_BODY_NAME, RolloutState, StudentInputs
from .student_net import ACTION_GROUP_SIZES, RecurrentVisionStudent
from .evaluation import EvaluationBeta, evaluate_policy, evaluation_resets
from .finetuning import VisualFineTuning
from .stage_beta import StageSawtoothBeta, stage_measurement
from .stage_probe import StageProbeProcess

try:
    from ..visual_augmentation import VisualAugmentation, augmentation_settings, DEFAULT_SEED
    from ..student_vision import image_size_from_config
    from ..dagger_vp.beta_schedule import beta_at_step, dagger_target_steps
    from ..dagger_vp.metrics import DaggerMetrics, HardwareMonitor
    from ..separate_dagger import _device, _student_checkpoints
    from ..separate_ppo_trainer import load_policy_router
    from ..separate_evaluation import save_stage_plots
    from ..stage_diagnostics import save_stage_diagnostics
except ImportError:  # python config_run/main_separate.py ...
    from visual_augmentation import VisualAugmentation, augmentation_settings, DEFAULT_SEED
    from student_vision import image_size_from_config
    from dagger_vp.beta_schedule import beta_at_step, dagger_target_steps
    from dagger_vp.metrics import DaggerMetrics, HardwareMonitor
    from separate_dagger import _device, _student_checkpoints
    from separate_ppo_trainer import load_policy_router
    from separate_evaluation import save_stage_plots
    from stage_diagnostics import save_stage_diagnostics


def build_stage_evaluation_env(config):
    """Independent small scene: periodic probes must not reset training episodes."""
    try:
        from ..main_separate import build_env
    except ImportError:
        from main_separate import build_env
    settings = deepcopy(config)
    settings.update(num_envs=int(config.get('dagger_stage_eval_num_envs',32)),
                    train_or_eval='train_dagger_new',headless=True,eval_start_stage=0,
                    dagger_target_stage=int(config.get('dagger_target_stage',4)),
                    reset_to_stage0=False,use_snapshot_curriculum=True,snapshot_save_probability=0.,
                    dagger_defer_camera_rendering=True,log_termination_reasons=False)
    return build_env(settings)


def validate_action_layout(env):
    expected = [name for name in env.robot_joint_names
                if name.startswith("waist_")]
    for side in ("left", "right"):
        expected += [name for name in env.robot_joint_names
                     if name.startswith(f"{side}_") and "_hand_" not in name
                     and not any(word in name for word in ("hip_", "knee_", "ankle_"))]
    for side in ("left", "right"):
        expected += [name for name in env.robot_joint_names if name.startswith(f"{side}_hand_")]
    expected += list(env.LOCOMOTION_COMMAND_NAMES)
    if len(expected) != 34 or list(env.action_names) != expected:
        raise ValueError("New student requires the standard 31 upper-body targets + 3 walking commands")
    for name, size in ACTION_GROUP_SIZES:
        if len(env.action_groups[name]) != size:
            raise ValueError(f"Unexpected {name} action layout")


def camera_config(config):
    """Record image geometry and rendering appearance for compatibility checks."""
    keys = ('name', 'width', 'height', 'fov_vertical', 'fov_horizontal', 'clipping_range',
            'mount_to', 'mount_link', 'mount_pos', 'mount_quat', 'pos', 'look_at',
            'gaussian_splat_path', 'gaussian_position', 'gaussian_rotation_xyzw',
            'gaussian_scale', 'gaussian_z_up', 'render_sim_geometry', 'tone_mapper', 'render_mode')
    return deepcopy({name: {key: spec.get('params', {}).get(key) for key in keys}
                     for name, spec in config.get('cameras', {}).items()})


def validate_camera_config(metadata, config, *, strict=True):
    if 'camera_config' in metadata and metadata['camera_config'] != camera_config(config):
        message = 'Camera calibration/scene differs from the student checkpoint'
        if strict:
            raise ValueError(message)
        log.warning('{}; continuing evaluation with the current camera/scene configuration',message)


def train_dagger(config, base, env):
    device = _device()
    validate_action_layout(env)
    stage_beta = config.get('beta_schedule') == 'stage_sawtooth'
    target_stage = config.get('dagger_target_stage', 4 if stage_beta else None)
    if target_stage is not None and (isinstance(target_stage,bool) or not isinstance(target_stage,int)
                                     or not 0 <= target_stage <= 5):
        raise ValueError('dagger_target_stage must be an integer from 0 to 5')
    task = getattr(base.env.handler,'task',None)
    if task is not None and target_stage is not None:
        task.dagger_target_stage = target_stage
        task.curriculum_max_stage = target_stage
    sequence_length = int(config.get("dagger_new_sequence_length", 16))
    burn_in = int(config.get("dagger_new_burn_in", 8))
    batch_size = int(config.get("dagger_new_batch_size", min(8, max(1, int(config.get("dagger_batch_size", 512))
                                                                  // max(1, sequence_length)))))
    capacity = int(config.get("dagger_buffer_steps", 3000))
    store_count = int(config.get("dagger_store_per_step", min(32, env.num_envs)))
    train_every = int(config.get("dagger_train_every", 20))
    updates = int(config.get("dagger_updates_per_train", 10))
    log_every = int(config.get("dagger_log_every", 100))
    grad_clip = float(config.get("dagger_new_grad_clip", 1.0))
    if min(sequence_length, batch_size, train_every, updates, log_every) <= 0 or burn_in < 0:
        raise ValueError("Sequence/training counts must be positive and burn-in nonnegative")
    if capacity < sequence_length + burn_in:
        raise ValueError(
            f"dagger_buffer_steps={capacity} must be >= dagger_new_burn_in={burn_in} "
            f"+ dagger_new_sequence_length={sequence_length} = {burn_in + sequence_length}"
        )
    if store_count <= 0:
        raise ValueError(f"dagger_store_per_step must be positive, got {store_count}")
    if store_count > env.num_envs:
        log.warning("dagger_store_per_step={} exceeds num_envs={}; storing all {} environments instead",
                    store_count, env.num_envs, env.num_envs)
        store_count = env.num_envs
    if not np.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("dagger_new_grad_clip must be positive and finite")
    seed = int(config.get("dagger_new_seed", 0))
    torch.manual_seed(seed)
    random.seed(seed)
    expert_path = config.get("expert_model_path") or config.get("load_model_path")
    if not expert_path:
        raise ValueError("train_dagger_new requires expert_model_path (the six-policy teacher bundle)")
    teacher, _ = load_policy_router(env, expert_path, device=device,
                                    checkpoint=config.get("load_model_checkpoint"))
    resume = config.get("train_or_eval") == "load_and_train_dagger_new"
    start_step = 0
    if resume:
        if not config.get("load_student_path"):
            raise ValueError("load_and_train_dagger_new requires load_student_path")
        student, metadata, start_step = load_student(config["load_student_path"], env, base, device)
        if student.architecture_version < 2:
            raise ValueError("Legacy recurrent v1 weights may be saturated. Start train_dagger_new with "
                             "a fresh v2 model and optimizer; v1 checkpoints remain evaluable.")
        if ('dagger_new_pretrained_visual' in config
                and bool(config['dagger_new_pretrained_visual']) != bool(metadata.get('pretrained_visual', False))):
            raise ValueError("Cannot change pretrained ResNet initialization on resume; start a new student instead")
        camera_name = metadata["camera_name"]
        if 'target_stage' in metadata and metadata['target_stage'] != target_stage:
            raise ValueError('Cannot change dagger_target_stage on resume')
    else:
        camera_name = str(config.get("dagger_camera_name", "camera0"))
        kwargs = model_kwargs(config, len(env.robot_joint_names))
        student = RecurrentVisionStudent(
            **kwargs, pretrained_visual=bool(config.get("dagger_new_pretrained_visual", False))
        ).to(device)
        # Fixed physical scales, rather than normalization changing underneath
        # an active recurrent rollout. Saved buffers are reused in evaluation.
        std = torch.ones(student.proprio_dim, device=device)
        std[student.num_joints:2 * student.num_joints] = 20.0
        std[2 * student.num_joints:2 * student.num_joints + 3] = 4.0
        student.set_proprio_normalization(torch.zeros_like(std), std)
        metadata = dict(model_kwargs=kwargs, joint_names=list(env.robot_joint_names),
                        action_names=list(env.action_names), imu_layout=IMU_LAYOUT, imu_body_name=IMU_BODY_NAME,
                        camera_name=camera_name, control_dt=control_dt(base),
                        pretrained_visual=bool(config.get('dagger_new_pretrained_visual', False)))
    metadata['target_stage'] = target_stage
    visual_training = VisualFineTuning(student, config, pretrained=metadata.get('pretrained_visual', False),
                                       saved=metadata.get('visual_finetuning'))
    optimizer = visual_training.make_optimizer(float(config.get('dagger_new_weight_decay', 1e-4)))
    if resume:
        restored_step = restore_optimizer(config["load_student_path"], optimizer, device, student=student)
        if restored_step != start_step:
            raise ValueError("Student and optimizer checkpoint steps differ")
        log.info("Resuming recurrent weights/optimizer at {}; simulator, replay and GRU start fresh", start_step)
        visual_training.apply(optimizer)
    log.info("ResNet pretrained={}, normalization={}, fine-tuning enabled={}, active={}, completed updates={}, "
             "warmup={} updates, LR policy={} / ResNet={}",
             visual_training.pretrained, student.visual_normalization, visual_training.enabled, visual_training.active,
             visual_training.completed_updates, visual_training.warmup_updates,
             visual_training.policy_lr, visual_training.visual_lr if visual_training.active else 0.)
    augmentation_enabled = config.get('dagger_visual_augmentation',False)
    if not isinstance(augmentation_enabled,bool):
        raise ValueError('dagger_visual_augmentation must be true or false')
    saved_augmentation = metadata.get('visual_augmentation',{})
    rollout_augmentation = replay_augmentation = None
    if augmentation_enabled:
        seed = int(config.get('dagger_augmentation_seed',DEFAULT_SEED))
        rollout_augmentation = VisualAugmentation(seed,saved_augmentation.get('rollout_rng'))
        replay_augmentation = VisualAugmentation(seed+1,saved_augmentation.get('replay_rng'))
        if saved_augmentation.get('settings') not in (None,augmentation_settings()):
            log.warning('Visual augmentation ranges changed; using constants from visual_augmentation.py')
        log.info('Visual augmentation enabled: clean replay, episode styles during rollout, '
                 'sequence styles during learning; settings={}',augmentation_settings())
    end_step = dagger_target_steps(config, start_step)
    observation = env.reset()
    state = RolloutState(student, env.num_envs)
    adapter = StudentInputs(base, env, device, camera_name, metadata["joint_names"],
                            image_size=image_size_from_config(config))
    first = adapter.read(state.previous_actions)
    image_shape = list(first[0].shape[1:])
    if resume and image_shape != metadata["image_shape"]:
        raise ValueError("Resume camera image shape differs from training")
    metadata["image_shape"] = image_shape
    metadata["image_preprocessing"] = "center_crop_to_aspect_area_resize_rgb_v2"
    validate_camera_config(metadata, config)
    metadata["camera_config"] = camera_config(config)
    minimum_ratio = float(config.get("dagger_min_train_sample_ratio", 1.))
    if not np.isfinite(minimum_ratio) or minimum_ratio < 0:
        raise ValueError("dagger_min_train_sample_ratio must be finite and nonnegative")
    updates = max(updates, math.ceil(train_every * store_count * minimum_ratio / (batch_size * sequence_length)))
    log.info("Learner uses {} updates per collection block; trained/new positions={:.3f}",
             updates, updates * batch_size * sequence_length / (train_every * store_count))
    # Keep slots fixed throughout the run; resampling IDs each step would
    # create sequences made from unrelated environments.
    selected = torch.randperm(env.num_envs, device=device)[:store_count]
    selected_numpy = selected.cpu().numpy()
    buffer = SequenceBuffer(
        capacity, store_count, image_shape, student.num_joints,
        (student.num_layers, student.hidden_size), device=device,
        storage_device=config.get("dagger_new_buffer_device", "cpu"),
        pin_memory=bool(config.get("dagger_buffer_pin_memory", False)),
        demonstrations_per_stage=int(config.get("dagger_demonstrations_per_stage", 0)),
        demonstration_length=sequence_length + burn_in,
        demonstration_fraction=float(config.get("dagger_demonstration_fraction", .5)),
        demonstration_burn_in=burn_in,
    )
    anchor_fraction = float(config.get("dagger_teacher_anchor_fraction", 0.))
    if not 0 <= anchor_fraction < 1:
        raise ValueError("dagger_teacher_anchor_fraction must be in [0,1)")
    anchors = selected[:max(1, math.ceil(store_count * anchor_fraction))] if anchor_fraction else selected[:0]
    gated_beta = config.get("beta_schedule", "evaluation") == "evaluation"
    if (not stage_beta and metadata.get('beta_controller', {}).get('mode') == StageSawtoothBeta.mode):
        raise ValueError('Cannot change beta controller on resume; start a new run')
    if stage_beta:
        controller = StageSawtoothBeta(config, metadata.get('beta_controller'), start_step)
    else:
        controller = EvaluationBeta(config, metadata.get('beta_controller'), start_step) if gated_beta else None
    def current_beta(step):
        return controller.beta_at_step(step) if controller is not None else beta_at_step(config, step)
    eval_every = int(config.get("dagger_eval_every", 500 if stage_beta else 5000 if gated_beta else 0))
    if (gated_beta or stage_beta) and eval_every <= 0:
        raise ValueError("Evaluation-gated beta requires dagger_eval_every > 0")
    if stage_beta and int(config.get('dagger_eval_episodes',32)) < controller.min_attempts:
        raise ValueError('dagger_eval_episodes must be >= dagger_beta_min_stage_attempts')
    if stage_beta:
        prefix_every = int(config.get('dagger_prefix_eval_every',2500))
        if (int(config.get('dagger_stage_eval_num_envs',32)) <= 0 or prefix_every < 0
                or prefix_every and prefix_every % eval_every):
            raise ValueError('Stage evaluation env count must be positive; prefix interval must be 0 or a multiple of dagger_eval_every')
    save_dir = output_directory(config)
    save_dir.mkdir(parents=True, exist_ok=True)
    archive_path = save_dir / "teacher_demonstrations.pth"
    if resume and buffer.demonstrations_per_stage:
        previous_archive = (Path(config["load_student_path"]).parent
                            / metadata.get('demonstration_archive', 'teacher_demonstrations.pth'))
        if not previous_archive.is_file():
            raise FileNotFoundError(f"Resume requires preserved teacher demonstrations: {previous_archive}")
        archive = torch.load(previous_archive, map_location='cpu', weights_only=True)
        if archive['camera_config'] != camera_config(config):
            raise ValueError("Demonstration archive camera calibration differs")
        buffer.load_archive(archive['replay'])
    log_root = Path(config.get("tensorboard_log", str(save_dir / "tensorboard"))) / "recurrent"
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(log_root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")))
    writer.add_text("Run/config", json.dumps(config, indent=2, default=str), start_step)
    writer.add_text('Run/visual_augmentation',json.dumps(dict(enabled=augmentation_enabled,
                    settings=augmentation_settings()),indent=2),start_step)
    writer.add_text("Run/recurrent", "Replay uses stored rollout hidden states plus burn-in (truncated BPTT). "
                    "Only episode resets zero memory. Resume restarts simulator/replay. "
                    "Mixed-teacher training success is not pure-student evaluation.", start_step)
    metrics = DaggerMetrics(env.get_current_stages(), env.NUM_POLICY_STAGES, control_dt(base))
    hardware = HardwareMonitor(device, interval=config.get("dagger_gpu_poll_seconds", 10))
    save_frequency = int(config.get("model_save_freq", 5000))
    next_step = start_step
    health = {}
    zero_gradient_checks = 0
    starting_updates = visual_training.completed_updates
    finished_normally = False
    probe_worker = None

    def stage_probe():
        nonlocal probe_worker
        if probe_worker is None:
            probe_worker = StageProbeProcess(config,metadata)
        return probe_worker

    def save_training(path, interrupted=False):
        replay_path = Path(str(path) + '.demonstrations.pth') if interrupted else archive_path
        metadata['visual_finetuning'] = visual_training.state_dict()
        metadata['visual_augmentation'] = dict(enabled=augmentation_enabled,settings=augmentation_settings())
        if augmentation_enabled:
            metadata['visual_augmentation'].update(rollout_rng=rollout_augmentation.state_dict(),
                                                   replay_rng=replay_augmentation.state_dict())
        if buffer.demonstrations_per_stage:
            metadata['demonstration_archive'] = replay_path.name
        if controller is not None:
            # Evaluation may fail after env.step advanced next_step, before
            # the normal end-of-step beta update. Recovery must stay aligned.
            controller.beta_at_step(next_step)
            metadata['beta_controller'] = controller.state_dict()
        save_checkpoint(path, student, optimizer, next_step, metadata)
        if buffer.demonstrations_per_stage:
            atomic_save(dict(camera_config=camera_config(config), replay=buffer.archive_state()), replay_path)

    def independent_evaluation(expert=False):
        if stage_beta:
            result = stage_probe().evaluate(student,start_stage=0,stop_stage=None,
                episodes=int(config.get('dagger_eval_episodes',32)),
                max_steps=int(config.get('dagger_eval_max_steps',6000))*math.ceil(
                    int(config.get('dagger_eval_episodes',32))/int(config.get('dagger_stage_eval_num_envs',32))),
                expert=expert)
        else:
            with evaluation_resets(env, start_stage=0):
                result = evaluate_policy(student,adapter,env,
                    min(env.num_envs,int(config.get('dagger_eval_episodes',32))),
                    int(config.get('dagger_eval_max_steps',6000)),teacher=teacher if expert else None)
        prefix = 'Evaluation/teacher' if expert else 'Evaluation/student'
        for key, value in result.items():
            if value is not None:
                writer.add_scalar(f'{prefix}/{key}', value, next_step)
        with (save_dir / 'training_evaluations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(step=next_step, policy=prefix, **result)) + '\n')
        log.info("{} success={}/{}, capped={}", prefix, result['success_count'],
                 result['episodes_finished'], result['capped_attempts'])
        # Evaluation ran in this simulator. Explicit resets prevent replay
        # windows or GRU memory from silently bridging unrelated trajectories.
        if stage_beta:
            return result,observation,state
        training_observation = env.reset()
        metrics.restart_rollout(env.get_current_stages())
        return result, training_observation, RolloutState(student, env.num_envs)

    def stage_evaluation(expert=False):
        measurements = {}
        episodes = int(config.get('dagger_eval_episodes',32))
        # Fixed per-slot quotas may require several episodes on small devices.
        max_steps = int(config.get('dagger_eval_max_steps',6000))*math.ceil(episodes/int(config.get('dagger_stage_eval_num_envs',32)))
        for stage in controller.stages:
            try:
                row = stage_probe().evaluate(student,start_stage=stage,stop_stage=stage,
                    episodes=episodes,max_steps=max_steps,expert=expert)
                measurement = stage_measurement(row,stage)
            except RuntimeError as error:
                if 'no snapshot is available' not in str(error):
                    raise
                measurement = dict(successes=0,failures=0,attempts=0,censored=0,
                                   capped_attempts=0,success_rate=None)
                log.warning('Stage {} has no evaluation snapshot; reference/comparison is unknown',stage)
            measurements[stage] = measurement
            policy = 'teacher' if expert else 'student'
            for key,value in measurement.items():
                if value is not None:
                    writer.add_scalar(f'Evaluation/{policy}_stage_{stage}/{key}',value,next_step)
            with (save_dir/'training_evaluations.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=next_step,policy=policy,stage=stage,**measurement))+'\n')
            log.info('Stage {} {}: successes={}/{}, rate={}',stage,policy,
                     measurement['successes'],measurement['attempts'],measurement['success_rate'])
        return measurements,observation,state

    def capture_stage_reference():
        independent_evaluation(expert=True)
        reference,observation,state = stage_evaluation(expert=True)
        controller.set_reference(reference,next_step)
        atomic_save(dict(step=next_step,target_stage=target_stage,reference=reference),
                    save_dir/'teacher_stage_reference.pth')
        (save_dir/'teacher_stage_reference.json').write_text(json.dumps(reference,indent=2))
        unknown = [s for s in controller.stages if not controller.reliable(reference.get(s))]
        if unknown:
            log.warning('Insufficient teacher samples for stages {}; these stages will not trigger beta corrections',unknown)
        return observation,state

    def write_metrics():
        metrics.write(writer, next_step, buffer, current_beta(next_step),
                      optimizer.param_groups[0]['lr'], hardware, end_step)
        writer.add_scalar('Recurrent/buffer_total_mib', buffer.storage_bytes / 2**20, next_step)
        writer.add_scalar('Replay/demonstration_mib', buffer.demonstration_bytes / 2**20, next_step)
        writer.add_scalar('Student/trained_to_stored_ratio', metrics.trained_samples / max(1, metrics.stored), next_step)
        for name, value in visual_training.metrics().items():
            writer.add_scalar(f'Student/{name}', value, next_step)
        writer.add_scalar('Augmentation/enabled',int(augmentation_enabled),next_step)
        if rollout_augmentation is not None and rollout_augmentation.episode_parameters is not None:
            writer.add_scalar('Augmentation/rollout_env_fraction',
                              float(rollout_augmentation.episode_parameters.enabled.float().mean()),next_step)
        if stage_beta:
            for name,value in controller.metrics(next_step).items():
                writer.add_scalar(f'BetaControl/{name}',value,next_step)
        for name, value in health.items():
            writer.add_scalar(f'Health/{name}', value, next_step)
        for stage in range(6):
            writer.add_scalar(f'Replay/stage_{stage}_teacher_windows',
                              len(buffer.demonstrations.get(stage, [])), next_step)

    log.info("Recurrent DAgger: {}..{}, sequence={} + burn-in={}, batch={} sequences, replay={:.1f} MiB on {}",
             start_step, end_step, sequence_length, burn_in, batch_size, buffer.storage_bytes / 2**20,
             buffer.storage_device)
    try:
        if stage_beta and controller.reference is None and next_step >= controller.warmup_end:
            observation,state = capture_stage_reference()
        if not stage_beta and config.get('dagger_teacher_baseline', False):
            baseline, observation, state = independent_evaluation(expert=True)
            if baseline['success_rate'] < float(config.get('dagger_teacher_min_success', .01)):
                raise RuntimeError("Teacher baseline failed; inspect training_evaluations.jsonl before distillation")
        for step in range(start_step, end_step):
            beta = current_beta(step)
            with metrics.measure("camera_and_inputs"):
                values = adapter.read(state.previous_actions)
            with metrics.measure("inference_and_actions"):
                student.eval()
                with torch.no_grad():
                    expert = teacher.predict(observation, deterministic=True)
                    images = values[0].float().div_(255).contiguous()
                    if rollout_augmentation is not None:
                        images = rollout_augmentation.rollout(images,state.reset_mask,inplace=True)
                    prediction, hidden = student(images, *values[1:],
                                                 hidden=state.hidden, reset_mask=state.reset_mask)
                proposed = prediction.cpu().numpy()
                if not np.isfinite(proposed).all() or not np.isfinite(expert).all():
                    raise FloatingPointError("Non-finite student/teacher action; simulator was not stepped")
                metrics.record_predictions(proposed, expert, env.action_space.low, env.action_space.high)
                execute_expert = torch.rand(env.num_envs, device=device) < beta
                execute_expert[anchors] = True
                mask = execute_expert.cpu().numpy()
                actions = np.clip(np.where(mask[:, None], expert, proposed),
                                  env.action_space.low, env.action_space.high)
            with metrics.measure("buffer_store"):
                buffer.add(tuple(value.index_select(0, selected) for value in values),
                           torch.as_tensor(expert, device=device).index_select(0, selected),
                           state.hidden.index_select(1, selected), state.reset_mask.index_select(0, selected))
                get_stages = getattr(env, 'get_current_stages_torch', None)
                stages = (get_stages().index_select(0, selected) if get_stages is not None
                          else env.get_current_stages()[selected_numpy])
                buffer.record_demonstrations(stages, execute_expert.index_select(0, selected))
            with metrics.measure("env_step"):
                observation, rewards, dones, infos = env.step(actions)
                state.after_step(hidden, actions, dones)
            metrics.observe(dones, infos, store_count, float(mask.mean()))
            if (step + 1) % train_every == 0 and buffer.filled_steps >= sequence_length + burn_in:
                with metrics.measure("student_updates"):
                    losses = []
                    for update in range(updates):
                        inspect = update == updates - 1
                        student.collect_diagnostics = inspect
                        losses.append(train_sequence_step(student, optimizer, buffer, batch_size,
                                                          sequence_length, burn_in, grad_clip,
                                                          diagnostics=health if inspect else None,
                                                          augmentation=replay_augmentation))
                        if visual_training.after_update(optimizer):
                            log.info("Unfreezing pretrained ResNet after {} gradient updates; visual LR={}",
                                     visual_training.completed_updates, visual_training.visual_lr)
                    student.collect_diagnostics = False
                    zero_gradient_checks = (zero_gradient_checks + 1 if
                                            health['visual_gradient_norm'] == 0 and health['gru_gradient_norm'] == 0
                                            and losses[-1] > 1e-8 else 0)
                    if zero_gradient_checks >= 3:
                        raise RuntimeError("Encoder and GRU gradients vanished for three learning blocks; stopping collapsed training")
                metrics.record_updates(losses, batch_size * sequence_length)
                writer.add_scalar("DAgger/MSE_Loss", float(np.mean(losses)), step + 1)
            next_step = step + 1
            if stage_beta and controller.reference is None and next_step >= controller.warmup_end:
                observation,state = capture_stage_reference()
            if (stage_beta and controller.reference is not None and next_step >= controller.warmup_end
                    and (next_step-controller.warmup_end) % eval_every == 0):
                measured,observation,state = stage_evaluation()
                bad = controller.observe(measured,next_step)
                if bad:
                    log.info('Stage beta correction: stages={}, beta={:.3f}, pause_until={}',
                             bad,controller.beta,controller.pause_until)
                prefix_every = int(config.get('dagger_prefix_eval_every',2500))
                if prefix_every > 0 and (next_step-controller.warmup_end) % prefix_every == 0:
                    _,observation,state = independent_evaluation()
            elif not stage_beta and eval_every > 0 and next_step % eval_every == 0:
                result, observation, state = independent_evaluation()
                if controller is not None:
                    controller.observe(result, next_step)
            beta_next = current_beta(next_step)
            if metrics.steps % log_every == 0:
                write_metrics()
                log.info("Recurrent DAgger {}/{} beta={:.4f}, episodes={}, success={}",
                         next_step, end_step, beta_next, metrics.episodes, metrics.successful_episodes)
            if save_frequency > 0 and next_step % save_frequency == 0:
                save_training(save_dir / f"student_model_step_{step}.pth")
        finished_normally = True
    finally:
        try:
            if finished_normally:
                save_training(save_dir / 'student_model_final.pth')
            elif next_step > start_step or visual_training.completed_updates > starting_updates:
                recovery_path = save_dir / f'student_model_interrupted_step_{next_step}.pth'
                save_training(recovery_path, interrupted=True)
                log.warning('Training interrupted; recovery checkpoint saved to {}', recovery_path)
            if metrics.steps > metrics.last_steps:
                write_metrics()
        finally:
            hardware.close()
            writer.close()
            if probe_worker is not None:
                probe_worker.close()


def evaluate_checkpoint(config, base, env, path, device, video=False, *, stage_records=None):
    student, metadata, checkpoint_step = load_student(path, env, base, device)
    target_stage = metadata.get('target_stage')
    configured_target = config.get('dagger_target_stage')
    if configured_target is not None and target_stage is not None and configured_target != target_stage:
        raise ValueError('Evaluation target stage differs from the student checkpoint')
    if target_stage is not None:
        task = getattr(base.env.handler, 'task', None)
        if task is not None:
            task.dagger_target_stage = target_stage
            task.curriculum_max_stage = target_stage
    validate_camera_config(metadata, config,strict=False)
    adapter = StudentInputs(base, env, device, metadata["camera_name"], metadata["joint_names"],
                            image_size=tuple(metadata["image_shape"][-2:]))
    video_writer = None
    frame_callback = None
    if video:
        import cv2
        video_path = Path(config.get("video_save_path", "./config_run/output/eval_videos/recurrent.mp4"))
        video_path.parent.mkdir(parents=True, exist_ok=True)
        _, height, width = metadata["image_shape"]
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       float(config.get("video_fps", 30)), (width, height))
        if not video_writer.isOpened():
            video_writer.release()
            raise RuntimeError(f"Cannot write video to {video_path}")
        def frame_callback(images):
            frame = cv2.cvtColor(images[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
            for _ in range(max(1, int(config.get("video_slowdown", 1)))):
                video_writer.write(frame)
    try:
        with evaluation_resets(env, int(config.get('eval_start_stage') or 0)):
            result = evaluate_policy(
                student, adapter, env, int(config.get('eval_episodes', 1 if video else env.num_envs)),
                int(config.get('eval_max_steps', 2000) if video else config.get('eval_total_step_cap', 6000)),
                frame_callback=frame_callback,
                count_timeouts=bool(config.get('eval_count_timeouts_as_failures', True)),
                stage_records=stage_records)
    finally:
        if video_writer is not None:
            video_writer.release()
    row = dict(checkpoint=Path(path).name, x_step=checkpoint_step, **result)
    log.info("Recurrent evaluation {}: success={}/{}, capped={}", path, result['success_count'],
             result['episodes_finished'], result['capped_attempts'])
    return row


def eval_dagger(config, base, env, video=False):
    validate_action_layout(env)
    root = evaluation_path(config.get("load_model_path", "./config_run/output/dagger_separate_stable_v2"))
    checkpoints = _student_checkpoints(root)
    if video:
        # A directory selects its final/latest checkpoint for video.
        checkpoints = checkpoints[-1:]
    report = []
    for _, path in checkpoints:
        stage_records = [] if not video else None
        row = evaluate_checkpoint(config, base, env, path, _device(), video, stage_records=stage_records)
        report.append((row, stage_records))
    rows = sorted((row for row, _ in report), key=lambda row: row["x_step"])
    output = Path(config.get("eval_output_dir", root.parent if root.is_file() else root))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "eval_dagger_new_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not video:
        save_stage_diagnostics(report, output, "eval_dagger_new")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib is unavailable; recurrent evaluation CSV was saved")
            return rows
        save_stage_plots(rows, output, "eval_dagger_new", "DAgger vector step")
        for key, label in (("mean_reward", "Average reward"), ("success_rate", "Success rate"),
                           ("mean_length", "Average episode length")):
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot([row["x_step"] for row in rows], [row[key] for row in rows], marker="o")
            axis.set(xlabel="DAgger vector step", ylabel=label, title="Recurrent student")
            axis.grid(True, alpha=.3)
            figure.tight_layout()
            figure.savefig(output / f"eval_dagger_new_{key}.png")
            plt.close(figure)
    return rows


def run_dagger(config, base, env):
    mode = config.get("train_or_eval")
    if mode in ("train_dagger_new", "load_and_train_dagger_new"):
        return train_dagger(config, base, env)
    if mode in ("eval_dagger_new", "eval_dagger_new_video"):
        return eval_dagger(config, base, env, video=mode == "eval_dagger_new_video")
    raise ValueError(f"Unsupported recurrent DAgger mode: {mode}")

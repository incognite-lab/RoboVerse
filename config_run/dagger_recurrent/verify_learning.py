"""Measure actual learning, gradients, and independent closed-loop performance.

Run from the repository root (the simulator requires GPU access):
  python -m config_run.dagger_recurrent.verify_learning --output config_run/output/dagger_verification

This is a bounded diagnostic, not evidence that a short run solves the entire
task. JSON explicitly separates imitation checks from closed-loop success.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .buffer import SequenceBuffer, train_sequence_step
from .checkpoint import control_dt, model_kwargs, save_checkpoint
from .evaluation import evaluate_policy, evaluation_resets
from .inputs import IMU_BODY_NAME, RolloutState, StudentInputs
from .student_net import RecurrentVisionStudent
from .finetuning import VisualFineTuning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='chairman_separate/train_dagger')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--eval-steps', type=int, default=6000)
    parser.add_argument('--image-width', type=int, default=640)
    parser.add_argument('--image-height', type=int, default=360)
    parser.add_argument('--synthetic', action='store_true', help='No simulator; tests only optimization')
    parser.add_argument('--no-pretrained', action='store_true', help='Verify random weights + GroupNorm')
    parser.add_argument('--visual-warmup-updates', type=int, default=10,
                        help='Short verification warmup; production config normally uses 1000')
    args = parser.parse_args()
    if (args.num_envs < 2 or args.steps < 24 or args.updates < 1 or args.eval_steps < 1
            or args.visual_warmup_updates < 0):
        parser.error('Need >=2 environments, >=24 steps, positive updates/eval steps')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = dict(device=str(device), synthetic=args.synthetic, arguments=vars(args).copy(),
                  full_task_success_proven=False)
    report['arguments']['output'] = str(args.output)
    started = time.perf_counter()
    def write_report():
        report['elapsed_seconds'] = time.perf_counter()-started
        (args.output/'results.json').write_text(json.dumps(report, indent=2))
    env = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from config_run.main_separate import build_env, load_config_from_yaml
        from config_run.separate_ppo_trainer import load_policy_router
        from .runner import camera_config
        from .checkpoint import IMU_LAYOUT
        config = load_config_from_yaml(args.config)
        if args.no_pretrained:
            config['dagger_new_pretrained_visual'] = False
        config['dagger_new_visual_warmup_updates'] = args.visual_warmup_updates
        config.update(num_envs=args.num_envs, headless=True, dagger_defer_camera_rendering=True,
                      dagger_image_width=args.image_width, dagger_image_height=args.image_height,
                      reset_to_stage0=False, use_snapshot_curriculum=True,
                      log_termination_reasons=False, snapshot_save_probability=0.,
                      dagger_new_visual_batch_size=2, dagger_new_checkpoint_visual=True)
        (args.output/'config.json').write_text(json.dumps(config, indent=2))
        if not args.synthetic:
            # Match production startup: Genesis initializes its CUDA runtime
            # before PyTorch allocates model tensors on the device.
            from loguru import logger
            logger.remove()
            logger.add(sys.stderr, level='WARNING')
            base, env = build_env(deepcopy(config))
        pretrained = bool(config.get('dagger_new_pretrained_visual', False))
        finetune = config.get('dagger_new_finetune_visual', True)
        if pretrained and finetune and args.updates <= args.visual_warmup_updates:
            raise ValueError('Verification needs more updates than warmup to measure backbone fine-tuning')
        model = RecurrentVisionStudent(**model_kwargs(config, 43), pretrained_visual=pretrained).to(device)
        std = torch.ones(model.proprio_dim, device=device)
        std[43:86] = 20.
        std[86:89] = 4.
        model.set_proprio_normalization(torch.zeros_like(std), std)
        model.collect_diagnostics = True
        shape = (3, args.image_height, args.image_width)
        # Production sequence length and burn-in; independent held-out env 0.
        length, burn = 16, 8
        capacity = min(args.steps, 128)
        archive_options = dict(demonstrations_per_stage=16, demonstration_length=length+burn,
                               demonstration_fraction=.5, demonstration_burn_in=burn)
        hidden_shape = (model.num_layers, model.hidden_size)
        train = SequenceBuffer(capacity, args.num_envs-1, shape, 43, hidden_shape, device=device,
                               **archive_options)
        heldout = SequenceBuffer(capacity, 1, shape, 43, hidden_shape, device=device,
                                 **archive_options)
        if args.synthetic:
            for step in range(args.steps):
                images = torch.randint(0, 256, (args.num_envs, *shape), dtype=torch.uint8, device=device)
                q = torch.randn(args.num_envs, 43, device=device)*.3
                values = (images, q, torch.zeros_like(q), torch.zeros(args.num_envs, 6, device=device),
                          torch.zeros(args.num_envs, 34, device=device))
                target = q[:, :34]*.5
                target[:, 0] = (images[:, 0].float().mean((1,2))/255-.5)
                for buffer, sl in ((train, slice(1, None)), (heldout, slice(0, 1))):
                    buffer.add(tuple(v[sl] for v in values), target[sl], model.initial_hidden(args.num_envs)[:, sl],
                               torch.ones(args.num_envs, dtype=torch.bool, device=device)[sl])
        else:
            teacher, manifest = load_policy_router(env, config['expert_model_path'], device=device)
            report['teacher_global_timesteps'] = manifest['global_timesteps']
            adapter = StudentInputs(base, env, device, 'camera0', image_size=(args.image_height,args.image_width))
            with evaluation_resets(env, start_stage=0):
                report['teacher_evaluation'] = evaluate_policy(model, adapter, env, args.num_envs,
                                                               args.eval_steps, teacher=teacher)
            write_report()
            print('Teacher evaluation:', json.dumps(report['teacher_evaluation']), flush=True)
            observation = env.reset()
            state = RolloutState(model, args.num_envs)
            seen_stages = set()
            for step in range(args.steps):
                values = adapter.read(state.previous_actions)
                with torch.no_grad():
                    expert = teacher.predict(observation, deterministic=True)
                    _, hidden = model(values[0].float()/255, *values[1:], hidden=state.hidden,
                                      reset_mask=state.reset_mask)
                target = torch.as_tensor(expert, device=device)
                stages = env.get_current_stages()
                seen_stages.update(stages.tolist())
                for buffer, sl in ((train, slice(1, None)), (heldout, slice(0, 1))):
                    buffer.add(tuple(v[sl] for v in values), target[sl], state.hidden[:, sl], state.reset_mask[sl])
                    buffer.record_demonstrations(stages[sl], torch.ones(args.num_envs, dtype=torch.bool, device=device)[sl])
                observation, _, dones, _ = env.step(expert)
                state.after_step(hidden, expert, dones)
            report['collected_stages'] = sorted(seen_stages)
            report['train_archive_windows'] = {k: len(v) for k,v in train.demonstrations.items()}
            report['heldout_archive_windows'] = {k: len(v) for k,v in heldout.demonstrations.items()}
        # Evaluate EXACTLY the same held-out windows before and after training.
        train_batch = train.sample(2, length, burn)
        validation_batch = heldout.sample(8, length, burn)
        def mse(batch):
            values, target, hidden, resets = batch
            model.eval()
            with torch.no_grad():
                _, warmed = model(*(v[:, :burn] for v in values), hidden=hidden, reset_mask=resets[:, :burn])
                pred, _ = model(*(v[:, burn:] for v in values), hidden=warmed, reset_mask=resets[:, burn:])
                return float((pred-target[:, burn:]).square().mean())
        report['train_mse_before'] = mse(train_batch)
        report['heldout_mse_before'] = mse(validation_batch)
        groups = dict(visual=model.visual_encoder, proprio=model.proprio_encoder, gru=model.gru)
        before = {name: next(module.parameters()).detach().clone() for name,module in groups.items()}
        visual_training = VisualFineTuning(model, config, pretrained=pretrained)
        optimizer = visual_training.make_optimizer(float(config.get('dagger_new_weight_decay', 1e-4)))
        losses, health = [], {}
        for update in range(args.updates):
            loss = train_sequence_step(model, optimizer, train, 2, length, burn, 1., diagnostics=health)
            visual_training.after_update(optimizer)
            losses.append(loss)
            if (update+1) % 10 == 0:
                print('Update', update+1, 'MSE', loss, 'health', json.dumps(health), flush=True)
        report.update(train_mse_after=mse(train_batch), heldout_mse_after=mse(validation_batch),
                      final_health=health, losses=losses,
                      visual_finetuning=visual_training.state_dict(),
                      parameter_max_changes={name: float((next(module.parameters()).detach()-before[name]).abs().max())
                                             for name,module in groups.items()})
        trained_groups = groups if finetune else {k:v for k,v in groups.items() if k != 'visual'}
        frozen_visual_ok = (finetune or (report['parameter_max_changes']['visual'] == 0
                                        and health['visual_gradient_norm'] == 0))
        report['optimization_checks_passed'] = bool(
            report['train_mse_after'] < report['train_mse_before'] * .5
            and all(report['parameter_max_changes'][name] > 0 for name in trained_groups)
            and all(health[f'{name}_gradient_norm'] > 0 for name in trained_groups)
            and frozen_visual_ok
            and health['gru_input_saturation_fraction'] < .1)
        report['generalization_checks_passed'] = report['heldout_mse_after'] < report['heldout_mse_before']
        write_report()
        if env is not None:
            metadata = dict(model_kwargs=model_kwargs(config,43), joint_names=list(env.robot_joint_names),
                            action_names=list(env.action_names), imu_layout=IMU_LAYOUT, imu_body_name=IMU_BODY_NAME,
                            camera_name='camera0', camera_config=camera_config(config), control_dt=control_dt(base),
                            pretrained_visual=pretrained, visual_finetuning=visual_training.state_dict(),
                            image_shape=list(shape), image_preprocessing='center_crop_to_aspect_area_resize_rgb_v2')
            save_checkpoint(args.output/'student_after_smoke.pth', model, optimizer, 0, metadata)
            with evaluation_resets(env, start_stage=0):
                report['student_evaluation'] = evaluate_policy(model, adapter, env, args.num_envs, args.eval_steps)
            report['full_task_success_proven'] = report['student_evaluation']['success_rate'] >= .8
        write_report()
        print(json.dumps({k:v for k,v in report.items() if k != 'losses'}, indent=2), flush=True)
        if not report['optimization_checks_passed']:
            raise SystemExit('Optimization verification failed; inspect results.json')
    except BaseException as exc:
        report['error'] = str(exc)
        write_report()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()

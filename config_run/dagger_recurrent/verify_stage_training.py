"""Bounded GPU integration check of the production stage-based DAgger runner.

python -m config_run.dagger_recurrent.verify_stage_training --output /tmp/stage_check
Uses 2 training + 2 evaluation envs, actual teacher and 640x360 images.
This checks integration; it does not demonstrate a trained student's success.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eval-steps', type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config_run.main_separate import build_env, load_config_from_yaml
    from .runner import run_dagger
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr,level='INFO')
    config = load_config_from_yaml('chairman_separate/train_dagger')
    config.update(num_envs=2,headless=True,total_timesteps=32,
        model_save_path=str(args.output),tensorboard_log=str(args.output/'logs'),
        dagger_stage_eval_num_envs=2,dagger_eval_episodes=2,dagger_beta_min_stage_attempts=2,
        dagger_teacher_warmup_steps=2,dagger_eval_every=1000,dagger_prefix_eval_every=0,
        dagger_eval_max_steps=args.eval_steps,dagger_buffer_steps=32,dagger_store_per_step=2,
        dagger_new_batch_size=1,dagger_train_every=8,dagger_updates_per_train=1,
        dagger_min_train_sample_ratio=0,dagger_new_visual_warmup_updates=1,
        dagger_new_visual_batch_size=2,dagger_log_every=8,model_save_freq=0,
        log_termination_reasons=False,snapshot_save_probability=0.)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    base,env = build_env(config)
    try:
        run_dagger(config,base,env)
    finally:
        env.close()
    import torch
    payload = torch.load(args.output/'recurrent/student_model_final.pth',map_location='cpu',weights_only=True)
    metadata = payload['metadata']
    result = dict(next_step=payload['next_step'],target_stage=metadata['target_stage'],
                  beta_controller=metadata['beta_controller'],visual_finetuning=metadata['visual_finetuning'],
                  full_task_success_proven=False)
    assert result['next_step'] == 32
    assert result['target_stage'] == 4
    assert result['visual_finetuning']['completed_updates'] == 2
    assert set(result['beta_controller']['reference']) == set(range(5))
    assert abs(result['beta_controller']['beta']-.97) < 1e-6
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

"""Train and evaluate six simultaneous ChairMan body-part PPO policies."""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
from loguru import logger as log
from metasim.cfg.scenario import ScenarioCfg
from metasim.wrapper.gym_vec_env import MetaSimVecEnv
try:
    from .main_multi import load_config_from_yaml, get_cameras_from_config, get_sensors_from_config
    from .SB3_chairman_separate import StableBaseline3VecEnv, POLICY_NAMES
    from .separate_ppo_trainer import SeparatePPOTrainer, load_policy_router
    from .utils import ObsSaver
except ImportError:
    from main_multi import load_config_from_yaml, get_cameras_from_config, get_sensors_from_config
    from SB3_chairman_separate import StableBaseline3VecEnv, POLICY_NAMES
    from separate_ppo_trainer import SeparatePPOTrainer, load_policy_router
    from utils import ObsSaver

def build_env(config):
    if config.get("task") != "chairmanseparate":
        raise ValueError("main_separate.py requires task: chairmanseparate")
    scenario = ScenarioCfg(
        task=config["task"], robots=config["robots"], sim=config.get("sim", "genesis"),
        num_envs=int(config.get("num_envs", 1)), headless=bool(config.get("headless", False)),
        try_add_table=bool(config.get("try_add_table", config.get("add_table", True))),
        sensors=get_sensors_from_config(config.get("sensors", {})),
        cameras=get_cameras_from_config(config.get("cameras", {})),
        force=bool(config.get("force", False)),
        force_x_min=float(config.get("force_x_min", 0)), force_x_max=float(config.get("force_x_max", 0)),
        force_y_min=float(config.get("force_y_min", 0)), force_y_max=float(config.get("force_y_max", 0)),
    )
    scenario.env_spacing = float(config.get("env_spacing", 4.0))
    scenario.robots[0].fix_base_link = bool(config.get("fix_base_link", False))
    scenario.task.decimation = int(config.get("decimation", 1))
    scenario.task.reset_to_stage0 = bool(config.get("reset_to_stage0", True))
    scenario.task.use_snapshot_curriculum = bool(config.get("use_snapshot_curriculum", False))
    scenario.task.snapshot_save_probability = float(config.get("snapshot_save_probability", 0.0))
    scenario.task.eval_start_stage = config.get("eval_start_stage")
    scenario.task.log_termination_reasons = bool(config.get("log_termination_reasons", True))
    scenario.task.log_reward_components = True
    scenario.task.train_stage = None
    scenario.task.curriculum_max_stage = None
    scenario.task.separate_policy_specs = config.get("policies", {})
    scenario.task.verbose_motion_diagnostics = bool(config.get("verbose_motion_diagnostics", False))
    scenario.task.visualize_reach_waypoints = bool(config.get("visualize_reach_waypoints", False))
    scenario.task.visualize_center_of_mass = bool(config.get("visualize_center_of_mass", False))
    #scenario.dagger = 1
    if scenario.robots[0].name != "g1_with_hands" or scenario.robots[0].fix_base_link:
        raise ValueError("Separate ChairMan requires mobile g1_with_hands and fix_base_link: false")
    base = MetaSimVecEnv(scenario, task_name=config["task"],
                         num_envs=int(config.get("num_envs", 1)), sim=config.get("sim", "genesis"))
    return base, StableBaseline3VecEnv(base)

def evaluate(config, env, video=False):
    router, manifest = load_policy_router(
        env, config.get("load_model_path"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint=config.get("load_model_checkpoint"))
    log.info("Loaded six-policy bundle at {} transitions", manifest.get("global_timesteps", 0))
    max_steps = int(config.get("eval_max_steps", 6000))
    episodes_target = int(config.get("eval_episodes", env.num_envs))
    obs = env.reset()
    returns = np.zeros(env.num_envs, dtype=np.float64)
    lengths = np.zeros(env.num_envs, dtype=np.int64)
    finished_returns, successes = [], []
    saver = None
    if video:
        path = config.get("video_save_path", "./config_run/output/chairman_separate.mp4")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        saver = ObsSaver(video_path=path)
    for step in range(max_steps):
        action = router.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        returns += np.asarray(reward); lengths += 1
        if saver is not None:
            states = env.env.env.handler.get_states()
            for _ in range(int(config.get("video_slowdown", 1))): saver.add(states)
        for index in np.flatnonzero(done):
            finished_returns.append(float(returns[index]))
            successes.append(bool(infos[index].get("is_success", False)))
            returns[index] = 0; lengths[index] = 0
            if len(finished_returns) >= episodes_target: break
        if len(finished_returns) >= episodes_target: break
    if saver is not None:
        saver.save(); log.info("Saved evaluation video to {}", config.get("video_save_path"))
    mean_return = float(np.mean(finished_returns)) if finished_returns else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0
    log.info("Evaluation: episodes={}, mean reward={:.4f}, success={:.2%}",
             len(finished_returns), mean_return, success_rate)

def main():
    config_name = sys.argv[1] if len(sys.argv) == 2 else "chairman_separate/train_ppo"
    if len(sys.argv) > 2: raise SystemExit("Usage: python config_run/main_separate.py [config/name]")
    config = load_config_from_yaml(config_name)
    log.info("Loaded config {} for policies {}", config_name, POLICY_NAMES)
    _, env = build_env(config)
    mode = config.get("train_or_eval", "train")
    try:
        if mode in ("train", "load_and_train"):
            trainer = SeparatePPOTrainer(
                env, config,
                resume_path=config.get("load_model_path") if mode == "load_and_train" else None,
                resume_checkpoint=config.get("load_model_checkpoint"))
            trainer.learn()
        elif mode == "eval":
            evaluate(config, env, video=False)
        elif mode == "eval_video":
            evaluate(config, env, video=True)
        else:
            raise ValueError(f"Unsupported separate-policy mode: {mode}")
    finally:
        env.close()

if __name__ == "__main__":
    main()

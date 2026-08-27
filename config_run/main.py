import sys
import os
import yaml
import torch
import random

from loguru import logger as log

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg, NyxGaussianSplatCameraCfg, GyroSensorCfg, CommandCfg
from metasim.constants import PhysicStateType, SimType
from metasim.wrapper.gym_vec_env import MetaSimVecEnv
from stable_baselines3 import PPO
from callbacks import TensorboardMetricsCallback, SaveModelCallback,RewardPlotCallback,EvalCallback
import numpy as np
from metasim.cfg.lights import DistantLightCfg, CylinderLightCfg

from utils import ObsSaver
import time




def load_config_from_yaml(config_name: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_name (str): Name of the YAML config file

    Returns:
        dict: The loaded config dictionary
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "configs", f"{config_name}.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config
def get_lights_from_config(lights_config: dict):
    lights = []
    for light_config in lights_config.values():
        light_type = light_config.get("type")
        params = light_config.get("params", {})

        if light_type == "DistantLightCfg":
            if "direction" in params and isinstance(params["direction"], str):
                params["direction"] = tuple(map(float, params["direction"].strip("()[]").split(",")))
            if "color" in params and isinstance(params["color"], str):
                params["color"] = tuple(map(float, params["color"].strip("()[]").split(",")))
            light = DistantLightCfg(**params)

        elif light_type == "CylinderLightCfg":
            # Parsování specifických parametrů pro CylinderLightCfg
            if "pos" in params and isinstance(params["pos"], str):
                params["pos"] = tuple(map(float, params["pos"].strip("()[]").split(",")))
            if "rot" in params and isinstance(params["rot"], str):
                params["rot"] = tuple(map(float, params["rot"].strip("()[]").split(",")))
            if "color" in params and isinstance(params["color"], str):
                params["color"] = tuple(map(float, params["color"].strip("()[]").split(",")))
            light = CylinderLightCfg(**params)

        else:
            log.warning(f"Unknown light type: {light_type}, skipping...")
            continue

        lights.append(light)
    return lights
def get_sensors_from_config(sensors_config: dict):
    sensors = []
    #sensor_config = {'type': 'GyroSensorCfg', 'params': {'name': 'gyro0', 'pos': '(0.0, 0.0, 0.0)', 'mount_to': 'g1_with_hands', 'mount_link': 'torso_link'}}
    for sensor_config in sensors_config.values():
        sensor_type = sensor_config.get("type")
        params = sensor_config.get("params", {})
        if sensor_type == "GyroSensorCfg":
            sensor = GyroSensorCfg(**params)
            sensor.pos = tuple(map(float, sensor.pos.strip("()").split(",")))
        elif sensor_type == "CommandCfg":
            sensor = CommandCfg(**params)
        else:
            log.warning(f"Unknown sensor type: {sensor_type}, skipping...")
            continue
        sensors.append(sensor)
    return sensors
def get_cameras_from_config(cameras: dict):
    camera_list = []
    for camera_config in cameras.values():
        camera_type = camera_config.get("type")
        params = camera_config.get("params", {})
        if camera_type == "PinholeCameraCfg":
            camera = PinholeCameraCfg(**params)
            #camera.pos = tuple(map(float, camera.pos.strip("()").split(",")))
            #camera.look_at = tuple(map(float, camera.look_at.strip("()").split(",")))
        elif camera_type == "NyxGaussianSplatCameraCfg":
            camera = NyxGaussianSplatCameraCfg(**params)
        else:
            log.warning(f"Unknown camera type: {camera_type}, skipping...")
            continue
        camera_list.append(camera)
    return camera_list





def main():
    if len(sys.argv) < 2:
        #config_name = "g1_door_open_train"
        #config_name = "g1_door_open_eval"
        #config_name = "g1_reach_IK"
        #config_name = "g1_walk_new_train"
        #config_name = "g1_reach_pos_ori_train"
        #config_name = "g1_reach_pos_ori_eval"
        #config_name = "g1_stand_eval"
        #config_name = "g1_stand_train"
        #config_name = "g1_walk_new_eval"
        #config_name = "g1_door_IK"
        #config_name = "g1_door_open_stand_train"
        #config_name = "g1_door_stand_IK"
        #config_name = "chairman/eval_ppo_video"
        #config_name = "g1_ChairMan"
        config_name = "chairman_multi/train_ppo"
        # log.error("Please provide the config file path, e.g. python train_sb3.py configs/isaacgym.yaml")
        # exit(1)
    elif len(sys.argv) == 2:
        config_name = sys.argv[1]
    else:
        log.error("Too many arguments provided. Please provide only the config file path.")
        exit(1)
    config = load_config_from_yaml(config_name)
    log.info(f"Loaded config: {config_name}")

    scenario = ScenarioCfg(
        task=config.get("task"),
        robots = config.get("robots"),
        try_add_table=config.get("try_add_table", True),
        sim=config.get("sim"),
        num_envs=config.get("num_envs", 1),
        headless=config.get("headless", False),
        sensors = get_sensors_from_config(config.get("sensors", {})),
        cameras= get_cameras_from_config(config.get("cameras", {})),
        #lights = get_lights_from_config(config.get("lights", {})),
        force = config.get("force", False),
        force_x_min = config.get("force_x_min", 0.0),
        force_x_max = config.get("force_x_max", 0.0),
        force_y_min = config.get("force_y_min", 0.0),
        force_y_max = config.get("force_y_max", 0.0),

        )
    scenario.env_spacing = config.get("env_spacing", 2.0)
    scenario.robots[0].fix_base_link = config.get("fix_base_link", False)
    scenario.task.decimation = config.get("decimation", 1)


    #TODO import correct StableBaseline3VecEnv
    if config.get("task") == "stand":
        from SB3_stand_env import StableBaseline3VecEnv
        scenario.robots[0].urdf_path = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"
        scenario.robots[0].fix_base_link = False
    elif config.get("task") == "reachpos":
        from SB3_reach_pos_env import StableBaseline3VecEnv
    elif config.get("task") == "reachposori":
        from SB3_reach_pos_ori_env import StableBaseline3VecEnv
    elif config.get("task") == "walk":
        from SB3_walk_env import StableBaseline3VecEnv
        scenario.robots[0].urdf_path = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"
        scenario.robots[0].fix_base_link = False
    elif config.get("task") == "door":
        from SB3_door_opening import StableBaseline3VecEnv
    elif config.get("task") == "walk_new":
        from SB3_walk_new_env import StableBaseline3VecEnv
        scenario.robots[0].urdf_path = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"
        scenario.robots[0].fix_base_link = False
    elif config.get("task") == "door_stand":
        from SB3_door_stand_env import StableBaseline3VecEnv
        scenario.robots[0].urdf_path = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"
    elif config.get("task") in ["chairman", "chairmangrpo", "chairmangaussian", "chairmangaussiangrpo"]:
        from SB3_chairman_env import StableBaseline3VecEnv
        if scenario.robots[0].fix_base_link == False and "gaussian" not in config.get("task"):
            # The full G1 walking setup selects its Genesis-compatible URDF.
            # Do not replace it with the explicit ``world``-root variant.
            if scenario.robots[0].name != "g1_with_hands":
                scenario.robots[0].urdf_path = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"
            scenario.robots[0].fix_base_link = False
    elif config.get("task") in [
        "chairmansimple",
        "chairmansimplegrpo",
        "chairmansimplegaussian",
        "chairmansimplegaussiangrpo",
    ]:
        from SB3_chairman_simple_env import StableBaseline3VecEnv
        if scenario.robots[0].fix_base_link == False and "gaussian" not in config.get("task"):
            scenario.robots[0].urdf_path = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"
            scenario.robots[0].fix_base_link = False



    #-----------------------------------------------
    if config.get("net_arch_pivf", False):
        policy_kwargs = dict({"net_arch":{"pi": config.get("net_arch_pi", [128, 128, 128]),
                                         "vf": config.get("net_arch_vf", [128, 128, 128])},
                              "log_std_init": config.get("log_std_init", 0.0)})
    else:
        policy_kwargs = dict({"net_arch": config.get("net_arch", [128, 128, 128]),
                              "log_std_init": config.get("log_std_init", 0.0)})
    #-----------------------------------------------
    def lr_schedule(initial_value: float, final_value: float):
        """set linear or constant learning rate schedule"""
        def func(progress_remaining: float) -> float:
            if config.get("learning_schedule", "constant") == "linear":
                return final_value + (initial_value - final_value) * progress_remaining
            else:
                return initial_value
        return func

    if config.get("train_or_eval") == "IK":
        metasim_env = MetaSimVecEnv(scenario, task_name=config.get("task"), num_envs=config.get("num_envs", 1), sim=config.get("sim"))
        env = StableBaseline3VecEnv(metasim_env)
        from SB3_reach_pos_ori_env import ik_solver
        os.makedirs(os.path.dirname(config.get("video_save_path")), exist_ok=True)
        observation = ObsSaver(video_path=config.get("video_save_path"))
        slow = config.get("video_slowdown", 3)
        # inference
        obs = env.reset()
        target_object_name = config.get("target_object_name", None)
        for _ in range(3):
            #joint_positions = ik_solver(scenario.robots[0], target_object_name, env)
            for step in range(config.get("eval_episodes", 1000)):
                #time.sleep(0.2)
                if step % 100 == 0:
                    joint_positions = ik_solver(scenario.robots[0], target_object_name, env)
                pos = [pos for pos in joint_positions.values()]

                obs, rewards, dones, infos = env.step([pos])
                states = metasim_env.env.handler.get_states()
                for _ in range(slow):
                    observation.add(states)

                print(f"Step reward: {rewards} at step {step}")

                if dones.any():
                    log.info(f"Episode finished after {step + 1} steps")
            env.reset()
        observation.save()
        log.info(f"🎬 Video saved to {config.get('video_path')}")
        env.close()

        quit()



    elif config.get("train_or_eval") == "train":
        # PPO configuration
        #_Eval env
        metasim_env = MetaSimVecEnv(scenario, task_name=config.get("task"), num_envs=config.get("num_envs", 1), sim=config.get("sim"))
        env = StableBaseline3VecEnv(metasim_env)

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 128),
            batch_size=config.get("batch_size", 256),
            n_epochs=config.get("n_epochs", 4),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            ent_coef=config.get("ent_coef", 0.0),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            tensorboard_log=config.get("tensorboard_log", "./ppo_tensorboard/"),
            policy_kwargs=policy_kwargs,
        )
        model.learn(total_timesteps=config.get("total_timesteps", 1_000_000),
                    callback=[
                    SaveModelCallback(save_path=config.get("model_save_path"), save_freq=config.get("model_save_freq", 1_000_000),task_name=config.get("task")),
                    TensorboardMetricsCallback(
                        log_dir=config.get("tensorboard_log", "./ppo_tensorboard/"),
                        log_interval=config.get("tensorboard_log_interval", 1_000_000),
                        max_stage=config.get(
                            "max_stage",
                            4
                            if "simple" in config.get("task", "")
                            else (6 if "chairman" in config.get("task", "") else 3),
                        ),
                        verbose=1,
                    ),
                    # EvalCallback(
                    # eval_env=eval_env,
                    # eval_freq=config.get("eval_freq", 10_000_000),
                    # n_eval_episodes=config.get("n_eval_episodes", 5),
                    # log_dir=config.get("eval_log_dir", "./eval_logs"),
                    # save_best=True,
                    # best_model_dir=config.get("best_model_dir", "./best_models"),
                    # eval_max_steps=config.get("eval_max_steps", 1000)
                    #     )
                    ],
                    progress_bar=True,)

        #Save the model
        task_name = scenario.task.__class__.__name__[:-3]
        model.save(config.get("model_save_path", f"ppo_{task_name}_{config.get('sim', 'unknown_sim')}"))
        log.info("Model saved. Ending the training and closing the environment.")
        env.close()
        quit()

    elif config.get("train_or_eval") == "eval":
        import re
        import csv
        import matplotlib.pyplot as plt

        metasim_env = MetaSimVecEnv(
            scenario,
            task_name=config.get("task"),
            num_envs=config.get("num_envs", 1),
            sim=config.get("sim")
        )
        env = StableBaseline3VecEnv(metasim_env)

        # Oprava pro numpy při načítání modelů
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.numeric"] = np.core.numeric

        model_dir = config.get("load_model_path")

        if not os.path.isdir(model_dir):
            log.error(f"Provided load_model_path is not a directory: {model_dir}")
            env.close()
            exit(1)
        scenario.dagger = 2
        # ---------------------------------------------------------
        # Pomocná funkce: pozice base_linku židle v XY rovině
        # ---------------------------------------------------------
        def get_chair_base_xy():
            states = metasim_env.env.handler.get_states()
            chair = states.objects["chair"]
            chair_base_idx = chair.body_names.index("base_link")
            chair_pos_xy = chair.body_state[:, chair_base_idx, :2]
            return chair_pos_xy.detach().cpu().numpy().copy()

        # ---------------------------------------------------------
        # 1) Najít a seřadit checkpointy model_XXXX.zip
        # ---------------------------------------------------------
        model_files = []
        pattern = re.compile(r"model_(\d+)\.zip$")

        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match:
                step_count = int(match.group(1))
                model_files.append((step_count, filename))

        model_files.sort(key=lambda x: x[0])

        if not model_files:
            log.warning("No 'model_{step}.zip' files found in directory.")
            env.close()
            exit(0)

        log.info(f"Found {len(model_files)} PPO checkpoints. Starting evaluation...")

        # ---------------------------------------------------------
        # 2) Nastavení evaluace
        # ---------------------------------------------------------
        requested_eval_episodes = config.get("eval_episodes", env.num_envs)

        if requested_eval_episodes > env.num_envs:
            log.warning(
                f"eval_episodes={requested_eval_episodes} is larger than env.num_envs={env.num_envs}. "
                f"Using only {env.num_envs} episodes, because this evaluation mode allows each env to count only once."
            )

        # Každý env může odpovídat maximálně jedné epizodě.
        n_eval_episodes = min(requested_eval_episodes, env.num_envs)

        max_total_steps = config.get("eval_total_step_cap", 5000)
        count_timeouts_as_failures = config.get("eval_count_timeouts_as_failures", True)

        # stages 0, 1, 2, 3
        num_eval_stages = config.get("num_eval_stages", 4)

        # ---------------------------------------------------------
        # 3) Data pro grafy a CSV
        # ---------------------------------------------------------
        eval_steps = []

        avg_rewards = []
        std_rewards = []
        var_rewards = []

        success_rates = []

        avg_lengths = []
        std_lengths = []
        var_lengths = []

        avg_chair_displacements = []
        std_chair_displacements = []

        # Průměrný počet kroků strávený ve stages
        avg_stage_steps_per_model = []

        # Kolik envů skončilo v jednotlivých stage
        end_stage_counts_per_model = []

        # Počet úspěšných envů/epizod
        success_counts_per_model = []

        csv_rows = []
        csv_episode_rows = []

        # ---------------------------------------------------------
        # 4) Hlavní eval smyčka přes checkpointy
        # ---------------------------------------------------------
        for step_count, filename in model_files:
            full_path = os.path.join(model_dir, filename)
            log.info(f"Evaluating PPO checkpoint: {filename} | step={step_count}")

            try:
                model = PPO.load(
                    full_path,
                    env=env,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            except Exception as e:
                log.error(f"Failed to load model {filename}: {e}")
                continue

            obs = env.reset()

            episode_rewards = []
            episode_successes = []
            episode_lengths = []
            episode_stage_counts = []
            episode_end_stages = []
            episode_chair_displacements = []

            current_rewards = np.zeros(env.num_envs, dtype=np.float64)
            current_lengths = np.zeros(env.num_envs, dtype=np.int64)

            current_stage_counts = np.zeros(
                (env.num_envs, num_eval_stages),
                dtype=np.float64
            )

            last_stage_before_step = np.zeros(env.num_envs, dtype=np.int64)

            # Počáteční pozice židle pro každý env
            initial_chair_xy = get_chair_base_xy()

            # Poslední známá pozice židle před env.step().
            # Používá se při done, protože wrapper může po done env interně resetovat.
            last_chair_xy_before_step = initial_chair_xy.copy()

            # ---------------------------------------------------------
            # 1 env = 1 evaluační epizoda
            # ---------------------------------------------------------
            active_eval_envs = np.zeros(env.num_envs, dtype=bool)
            active_eval_envs[:n_eval_episodes] = True

            finished_envs = np.zeros(env.num_envs, dtype=bool)

            total_steps = 0

            while (
                np.sum(finished_envs & active_eval_envs) < n_eval_episodes
                and total_steps < max_total_steps
            ):
                running_envs = active_eval_envs & (~finished_envs)

                # -----------------------------------------------------
                # Uložíme pozici židle před krokem.
                # -----------------------------------------------------
                current_chair_xy = get_chair_base_xy()
                last_chair_xy_before_step[running_envs] = current_chair_xy[running_envs]

                actions, _ = model.predict(obs, deterministic=True)

                actions = np.asarray(actions)

                # Pro envy, které už byly vyhodnoceny, posíláme nulovou akci.
                # Tyto envy se ale už nepočítají do statistik.
                if np.any(~running_envs):
                    actions[~running_envs] = 0.0

                # ---------------------------------------------------------
                # Stage čteme PŘED env.step().
                # Tento krok tedy odpovídá stage, ve které agent právě vybírá akci.
                # ---------------------------------------------------------
                try:
                    actual_stage = (
                        metasim_env.env.handler.task.reward_functions[0]
                        .actual_stage
                    )

                    if actual_stage is None:
                        stages_before_step = np.zeros(env.num_envs, dtype=np.int64)
                    else:
                        stages_before_step = (
                            actual_stage
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int64)
                        )

                except Exception as e:
                    log.warning(f"Could not read actual_stage during PPO eval: {e}")
                    stages_before_step = np.zeros(env.num_envs, dtype=np.int64)

                stages_before_step = np.clip(
                    stages_before_step,
                    0,
                    num_eval_stages - 1
                )

                for i in range(env.num_envs):
                    if running_envs[i]:
                        current_stage_counts[i, stages_before_step[i]] += 1
                        last_stage_before_step[i] = stages_before_step[i]

                obs, rewards, dones, infos = env.step(actions)

                rewards = np.asarray(rewards, dtype=np.float64)
                dones = np.asarray(dones, dtype=bool)

                current_rewards[running_envs] += rewards[running_envs]
                current_lengths[running_envs] += 1

                total_steps += 1

                for i in range(env.num_envs):
                    # Započítáváme pouze první dokončení daného envu.
                    if running_envs[i] and dones[i]:
                        is_success = infos[i].get("is_success", False)

                        chair_displacement = float(
                            np.linalg.norm(
                                last_chair_xy_before_step[i] - initial_chair_xy[i]
                            )
                        )

                        end_stage = int(last_stage_before_step[i])
                        end_stage = int(np.clip(end_stage, 0, num_eval_stages - 1))

                        ep_reward = float(current_rewards[i])
                        ep_length = int(current_lengths[i])
                        ep_success = 1 if is_success else 0

                        episode_rewards.append(ep_reward)
                        episode_lengths.append(ep_length)
                        episode_successes.append(ep_success)
                        episode_stage_counts.append(current_stage_counts[i].copy())
                        episode_end_stages.append(end_stage)
                        episode_chair_displacements.append(chair_displacement)

                        finished_envs[i] = True

                        csv_episode_rows.append({
                            "checkpoint": filename,
                            "x_step": step_count,
                            "episode_index": len(episode_rewards),
                            "env_id": i,
                            "episode_reward": ep_reward,
                            "episode_length": ep_length,
                            "success": ep_success,
                            "end_stage": end_stage,
                            "chair_displacement_xy": chair_displacement,
                        })

                        log.info(
                            f"Eval env {i} finished once | "
                            f"episode={len(episode_rewards)}/{n_eval_episodes} | "
                            f"length={ep_length} | "
                            f"reward={ep_reward:.4f} | "
                            f"chair_displacement_xy={chair_displacement:.4f} m | "
                            f"end_stage={end_stage} | "
                            f"success={is_success}"
                        )

                        current_rewards[i] = 0.0
                        current_lengths[i] = 0
                        current_stage_counts[i, :] = 0.0

                # DŮLEŽITÉ:
                # Nevoláme obs = env.reset() při np.any(dones).
                # Hotové envy ignorujeme pomocí finished_envs.

            # ---------------------------------------------------------
            # Timeout epizody započítat jako neúspěšné
            # ---------------------------------------------------------
            episodes_finished = len(episode_rewards)

            if episodes_finished < n_eval_episodes:
                log.warning(
                    f"Checkpoint {filename}: only {episodes_finished}/{n_eval_episodes} "
                    f"eval envs finished before max_total_steps={max_total_steps}."
                )

                if count_timeouts_as_failures:
                    running_envs = active_eval_envs & (~finished_envs)
                    timeout_chair_xy = get_chair_base_xy()

                    for i in range(env.num_envs):
                        if running_envs[i]:
                            chair_displacement = float(
                                np.linalg.norm(
                                    timeout_chair_xy[i] - initial_chair_xy[i]
                                )
                            )

                            end_stage = int(last_stage_before_step[i])
                            end_stage = int(np.clip(end_stage, 0, num_eval_stages - 1))

                            ep_reward = float(current_rewards[i])
                            ep_length = int(current_lengths[i])
                            ep_success = 0

                            episode_rewards.append(ep_reward)
                            episode_lengths.append(ep_length)
                            episode_successes.append(ep_success)
                            episode_stage_counts.append(current_stage_counts[i].copy())
                            episode_end_stages.append(end_stage)
                            episode_chair_displacements.append(chair_displacement)

                            finished_envs[i] = True

                            csv_episode_rows.append({
                                "checkpoint": filename,
                                "x_step": step_count,
                                "episode_index": len(episode_rewards),
                                "env_id": i,
                                "episode_reward": ep_reward,
                                "episode_length": ep_length,
                                "success": ep_success,
                                "end_stage": end_stage,
                                "chair_displacement_xy": chair_displacement,
                            })

                            log.info(
                                f"Eval env {i} timed out | "
                                f"episode={len(episode_rewards)}/{n_eval_episodes} | "
                                f"length={ep_length} | "
                                f"reward={ep_reward:.4f} | "
                                f"chair_displacement_xy={chair_displacement:.4f} m | "
                                f"end_stage={end_stage} | "
                                f"success=False"
                            )

                            if len(episode_rewards) >= n_eval_episodes:
                                break

            # ---------------------------------------------------------
            # Statistiky pro checkpoint
            # ---------------------------------------------------------
            if len(episode_rewards) == 0:
                mean_reward = 0.0
                std_reward = 0.0
                var_reward = 0.0

                success_rate = 0.0
                success_count = 0

                mean_length = 0.0
                std_length = 0.0
                var_length = 0.0

                mean_stage_counts = np.zeros(num_eval_stages, dtype=np.float64)
                end_stage_counts = np.zeros(num_eval_stages, dtype=np.int64)

                mean_chair_displacement = 0.0
                std_chair_displacement = 0.0

                log.warning(f"No episodes evaluated for checkpoint {filename}")

            else:
                rewards_np = np.array(episode_rewards, dtype=np.float64)
                lengths_np = np.array(episode_lengths, dtype=np.float64)
                successes_np = np.array(episode_successes, dtype=np.float64)

                mean_reward = float(np.mean(rewards_np))
                std_reward = float(np.std(rewards_np))
                var_reward = float(np.var(rewards_np))

                success_rate = float(np.mean(successes_np))
                success_count = int(np.sum(successes_np))

                mean_length = float(np.mean(lengths_np))
                std_length = float(np.std(lengths_np))
                var_length = float(np.var(lengths_np))

                if len(episode_stage_counts) > 0:
                    mean_stage_counts = np.mean(
                        np.array(episode_stage_counts, dtype=np.float64),
                        axis=0
                    )
                else:
                    mean_stage_counts = np.zeros(num_eval_stages, dtype=np.float64)

                end_stage_counts = np.zeros(num_eval_stages, dtype=np.int64)
                for end_stage in episode_end_stages:
                    end_stage = int(np.clip(end_stage, 0, num_eval_stages - 1))
                    end_stage_counts[end_stage] += 1

                mean_chair_displacement = float(np.mean(episode_chair_displacements))
                std_chair_displacement = float(np.std(episode_chair_displacements))

            eval_steps.append(step_count)

            avg_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            var_rewards.append(var_reward)

            success_rates.append(success_rate)

            avg_lengths.append(mean_length)
            std_lengths.append(std_length)
            var_lengths.append(var_length)

            avg_chair_displacements.append(mean_chair_displacement)
            std_chair_displacements.append(std_chair_displacement)

            avg_stage_steps_per_model.append(mean_stage_counts)
            end_stage_counts_per_model.append(end_stage_counts.copy())
            success_counts_per_model.append(success_count)

            csv_row = {
                "checkpoint": filename,
                "x_step": step_count,

                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "var_reward": var_reward,

                "success_rate": success_rate,
                "success_count": success_count,

                "mean_length": mean_length,
                "std_length": std_length,
                "var_length": var_length,

                "mean_chair_displacement_xy": mean_chair_displacement,
                "std_chair_displacement_xy": std_chair_displacement,

                "episodes_evaluated": len(episode_rewards),
                "requested_eval_episodes": n_eval_episodes,
            }

            for stage_id in range(num_eval_stages):
                csv_row[f"mean_stage_{stage_id}_steps"] = float(mean_stage_counts[stage_id])

            for stage_id in range(num_eval_stages):
                csv_row[f"end_stage_{stage_id}_count"] = int(end_stage_counts[stage_id])

            csv_rows.append(csv_row)

            log.info(
                f" -> Reward: mean={mean_reward:.4f}, std={std_reward:.4f}, var={var_reward:.4f} | "
                f"Length: mean={mean_length:.2f}, std={std_length:.2f}, var={var_length:.2f} | "
                f"Chair displacement XY: mean={mean_chair_displacement:.4f} m, std={std_chair_displacement:.4f} m | "
                f"Success: {success_rate:.2%} ({success_count}/{len(episode_rewards)}) | "
                f"Avg Stage Steps: {mean_stage_counts} | "
                f"End Stage Counts: {end_stage_counts}"
            )

        env.close()

        # ---------------------------------------------------------
        # 5) Uložení CSV souborů
        # ---------------------------------------------------------
        summary_csv_path = os.path.join(model_dir, "eval_ppo_summary.csv")

        summary_fieldnames = [
            "checkpoint",
            "x_step",

            "mean_reward",
            "std_reward",
            "var_reward",

            "success_rate",
            "success_count",

            "mean_length",
            "std_length",
            "var_length",

            "mean_chair_displacement_xy",
            "std_chair_displacement_xy",

            "episodes_evaluated",
            "requested_eval_episodes",
        ]

        for stage_id in range(num_eval_stages):
            summary_fieldnames.append(f"mean_stage_{stage_id}_steps")

        for stage_id in range(num_eval_stages):
            summary_fieldnames.append(f"end_stage_{stage_id}_count")

        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        episodes_csv_path = os.path.join(model_dir, "eval_ppo_episodes.csv")

        episode_fieldnames = [
            "checkpoint",
            "x_step",
            "episode_index",
            "env_id",
            "episode_reward",
            "episode_length",
            "success",
            "end_stage",
            "chair_displacement_xy",
        ]

        with open(episodes_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=episode_fieldnames)
            writer.writeheader()
            writer.writerows(csv_episode_rows)

        log.info(f"Summary CSV saved to: {summary_csv_path}")
        log.info(f"Episode CSV saved to: {episodes_csv_path}")

        # ---------------------------------------------------------
        # 6) Převod na numpy pro grafy
        # ---------------------------------------------------------
        eval_steps_np = np.array(eval_steps, dtype=np.float64)

        avg_rewards_np = np.array(avg_rewards, dtype=np.float64)
        std_rewards_np = np.array(std_rewards, dtype=np.float64)

        avg_lengths_np = np.array(avg_lengths, dtype=np.float64)
        std_lengths_np = np.array(std_lengths, dtype=np.float64)

        avg_chair_disp_np = np.array(avg_chair_displacements, dtype=np.float64)
        std_chair_disp_np = np.array(std_chair_displacements, dtype=np.float64)

        success_rates_np = np.array(success_rates, dtype=np.float64)

        end_stage_matrix = np.array(
            end_stage_counts_per_model,
            dtype=np.float64
        )

        success_counts_np = np.array(
            success_counts_per_model,
            dtype=np.float64
        )

        # ---------------------------------------------------------
        # 7) Graf rewardu s rozptylem
        # Název a styl sjednocený s DAgger grafem.
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))

        plt.plot(
            eval_steps_np,
            avg_rewards_np,
            marker="o",
            linestyle="-",
            label="Mean reward"
        )

        plt.fill_between(
            eval_steps_np,
            avg_rewards_np - std_rewards_np,
            avg_rewards_np + std_rewards_np,
            alpha=0.2,
            label="±1 std"
        )

        plt.title(f'PPO Evaluation - Average Reward ({config.get("task")})')
        plt.xlabel("Training Steps")
        plt.ylabel("Average Reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_ppo_reward_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 8) Graf success rate
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))

        plt.plot(
            eval_steps_np,
            success_rates_np,
            marker="o",
            linestyle="-",
            label="Success rate"
        )

        plt.title(f'PPO Evaluation - Success Rate ({config.get("task")})')
        plt.xlabel("Training Steps")
        plt.ylabel("Success Rate")
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_ppo_success_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 9) Graf délky epizody s rozptylem
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))

        plt.plot(
            eval_steps_np,
            avg_lengths_np,
            marker="o",
            linestyle="-",
            label="Mean episode length"
        )

        plt.fill_between(
            eval_steps_np,
            avg_lengths_np - std_lengths_np,
            avg_lengths_np + std_lengths_np,
            alpha=0.2,
            label="±1 std"
        )

        plt.title(f'PPO Evaluation - Average Episode Length ({config.get("task")})')
        plt.xlabel("Training Steps")
        plt.ylabel("Average Steps per Episode")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_ppo_length_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 10) Graf posunutí židle s rozptylem
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))

        plt.plot(
            eval_steps_np,
            avg_chair_disp_np,
            marker="o",
            linestyle="-",
            label="Mean chair displacement"
        )

        plt.fill_between(
            eval_steps_np,
            avg_chair_disp_np - std_chair_disp_np,
            avg_chair_disp_np + std_chair_disp_np,
            alpha=0.2,
            label="±1 std"
        )

        plt.title(f'PPO Evaluation - Chair Displacement ({config.get("task")})')
        plt.xlabel("Training Steps")
        plt.ylabel("Chair displacement in XY plane [m]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_ppo_chair_displacement_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 11) Stacked bar graf stage kroků + označení success rate
        # Styl sjednocený s DAgger grafem.
        # ---------------------------------------------------------
        if len(avg_stage_steps_per_model) > 0:
            stage_matrix = np.array(avg_stage_steps_per_model, dtype=np.float64)

            plt.figure(figsize=(max(12, len(eval_steps) * 0.7), 6))

            x = np.arange(len(eval_steps))
            bottom = np.zeros(len(eval_steps), dtype=np.float64)

            for stage_id in range(num_eval_stages):
                plt.bar(
                    x,
                    stage_matrix[:, stage_id],
                    bottom=bottom,
                    label=f"Stage {stage_id}"
                )
                bottom += stage_matrix[:, stage_id]

            max_bar_height = float(np.max(bottom)) if len(bottom) > 0 else 0.0
            text_offset = max(5.0, 0.03 * max_bar_height)

            for i, success_rate in enumerate(success_rates):
                success_percent = success_rate * 100.0

                if success_rate > 0.0:
                    success_label = f"✓ {success_percent:.0f}%"
                else:
                    success_label = "✗ 0%"

                plt.text(
                    x[i],
                    bottom[i] + text_offset,
                    success_label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0
                )

            plt.title(
                f'PPO Evaluation - Episode Length Split by Stage ({config.get("task")})'
            )
            plt.xlabel('Training Steps / Checkpoint')
            plt.ylabel('Average Steps per Episode')

            plt.xticks(
                x,
                [str(step) for step in eval_steps],
                rotation=45,
                ha="right"
            )

            plt.ylim(0, max_bar_height + 4 * text_offset)

            plt.legend(title="Stage")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "eval_ppo_stage_stacked_bar.png"))
            plt.close()

        # ---------------------------------------------------------
        # 12) Graf: kolik envů skončilo v jednotlivých stage
        #     + kolik envů splnilo celou úlohu
        # ---------------------------------------------------------
        if len(end_stage_counts_per_model) > 0:
            plt.figure(figsize=(max(12, len(eval_steps) * 0.7), 6))

            for stage_id in range(num_eval_stages):
                plt.plot(
                    eval_steps_np,
                    end_stage_matrix[:, stage_id],
                    marker="o",
                    linestyle="-",
                    label=f"Ended in stage {stage_id}"
                )

            plt.plot(
                eval_steps_np,
                success_counts_np,
                marker="o",
                linestyle="--",
                linewidth=2.5,
                label="Task success"
            )

            plt.title(
                f'PPO Evaluation - Number of Environments Ending in Each Stage ({config.get("task")})'
            )
            plt.xlabel("Training Steps / Checkpoint")
            plt.ylabel("Number of environments")
            plt.ylim(0, n_eval_episodes + 1)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "eval_ppo_end_stage_counts_plot.png"))
            plt.close()

        log.info(
            f"PPO evaluation complete. Plots and CSV files saved to: {model_dir}. "
            f"Saved plots: "
            f"eval_ppo_reward_plot.png, "
            f"eval_ppo_success_plot.png, "
            f"eval_ppo_length_plot.png, "
            f"eval_ppo_chair_displacement_plot.png, "
            f"eval_ppo_stage_stacked_bar.png, "
            f"eval_ppo_end_stage_counts_plot.png"
        )

        quit()
    elif config.get("train_or_eval") == "eval1":
        import re
        import matplotlib.pyplot as plt

        metasim_env = MetaSimVecEnv(
            scenario,
            task_name=config.get("task"),
            num_envs=config.get("num_envs", 1),
            sim=config.get("sim")
        )
        env = StableBaseline3VecEnv(metasim_env)

        # Oprava pro numpy
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.numeric'] = np.core.numeric

        model_dir = config.get("load_model_path")
        if not os.path.isdir(model_dir):
            log.error(f"Provided load_model_path is not a directory: {model_dir}")
            exit(1)

        # 1. Získání a seřazení modelů podle počtu kroků
        model_files = []
        pattern = re.compile(r"model_(\d+)\.zip")

        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match:
                step_count = int(match.group(1))
                model_files.append((step_count, filename))

        model_files.sort(key=lambda x: x[0])

        if not model_files:
            log.warning("No 'model_{step}.zip' files found in directory.")
            exit(0)

        log.info(f"Found {len(model_files)} models. Starting evaluation...")

        # Data pro grafy
        eval_steps = []
        avg_rewards = []
        success_rates = []
        avg_lengths = []

        # Data pro stacked bar graf stage kroků
        # Např. pro stages 0..3 nastav num_eval_stages: 4
        # Pro stages 0..6 nastav num_eval_stages: 7
        num_eval_stages = config.get("num_eval_stages", 4)
        avg_stage_steps_per_model = []

        # Počet epizod pro evaluaci jednoho modelu
        n_eval_episodes = config.get("eval_episodes", 20)

        # 2. Hlavní smyčka přes všechny modely
        for step_count, filename in model_files:
            full_path = os.path.join(model_dir, filename)
            log.info(f"Evaluating model: {filename} (Step: {step_count})")

            # Načtení modelu
            try:
                model = PPO.load(
                    full_path,
                    env=env,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            except Exception as e:
                log.error(f"Failed to load model {filename}: {e}")
                continue

            # Evaluace jednoho modelu
            episode_rewards = []
            episode_successes = []
            episode_lengths = []

            # Pro každou dokončenou epizodu uložíme počty kroků ve stages
            # Každý prvek bude vektor délky num_eval_stages:
            # [steps_stage_0, steps_stage_1, ...]
            episode_stage_counts = []

            # Reset prostředí
            obs = env.reset()

            # Pomocné proměnné pro akumulaci v běžících epizodách
            current_rewards = np.zeros(env.num_envs, dtype=np.float64)
            current_lengths = np.zeros(env.num_envs, dtype=np.float64)

            # Čítač kroků ve stage pro každé paralelní prostředí
            current_stage_counts = np.zeros(
                (env.num_envs, num_eval_stages),
                dtype=np.float64
            )

            # Běžíme dokud nemáme dostatek dokončených epizod
            while len(episode_rewards) < n_eval_episodes:
                actions, _ = model.predict(obs, deterministic=True)

                # ---------------------------------------------------------
                # Zjistíme stage PŘED krokem prostředí.
                #
                # Je to důležité, protože env.step() může při done rovnou
                # resetovat prostředí a stage by se po kroku mohla změnit.
                # Tento jeden krok tedy započítáme do stage, ve které byla
                # politika před provedením akce.
                # ---------------------------------------------------------
                try:
                    actual_stage = (
                        metasim_env.env.handler.task.reward_functions[0]
                        .actual_stage
                    )

                    if actual_stage is None:
                        stages_before_step = np.zeros(env.num_envs, dtype=np.int64)
                    else:
                        stages_before_step = (
                            actual_stage
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int64)
                        )

                except Exception as e:
                    log.warning(f"Could not read actual_stage during eval: {e}")
                    stages_before_step = np.zeros(env.num_envs, dtype=np.int64)

                stages_before_step = np.clip(
                    stages_before_step,
                    0,
                    num_eval_stages - 1
                )

                for i in range(env.num_envs):
                    current_stage_counts[i, stages_before_step[i]] += 1

                obs, rewards, dones, infos = env.step(actions)

                rewards = np.asarray(rewards)
                dones = np.asarray(dones)

                current_rewards += rewards
                current_lengths += 1

                # Zpracování dokončených epizod
                for i in range(env.num_envs):
                    if dones[i]:
                        # Uložení celkové odměny
                        episode_rewards.append(float(current_rewards[i]))
                        current_rewards[i] = 0.0

                        # Uložení délky epizody
                        episode_lengths.append(float(current_lengths[i]))
                        current_lengths[i] = 0.0

                        # Uložení počtu kroků ve stages pro tuto epizodu
                        episode_stage_counts.append(current_stage_counts[i].copy())
                        current_stage_counts[i, :] = 0.0

                        # Zjištění success rate
                        is_success = infos[i].get("is_success", False)
                        episode_successes.append(1 if is_success else 0)

                        if len(episode_rewards) >= n_eval_episodes:
                            break

            # Výpočet statistik pro tento model
            mean_reward = float(np.mean(episode_rewards))
            success_rate = float(np.mean(episode_successes))
            mean_length = float(np.mean(episode_lengths))

            # Průměrný počet kroků ve stages pro tento model
            if len(episode_stage_counts) > 0:
                mean_stage_counts = np.mean(
                    np.array(episode_stage_counts, dtype=np.float64),
                    axis=0
                )
            else:
                mean_stage_counts = np.zeros(num_eval_stages, dtype=np.float64)

            log.info(
                f" -> Mean Reward: {mean_reward:.2f}, "
                f"Success: {success_rate:.2%}, "
                f"Avg Length: {mean_length:.1f}, "
                f"Avg Stage Steps: {mean_stage_counts}"
            )

            eval_steps.append(step_count)
            avg_rewards.append(mean_reward)
            success_rates.append(success_rate)
            avg_lengths.append(mean_length)
            avg_stage_steps_per_model.append(mean_stage_counts)

        env.close()

        # 3. Vykreslení a uložení grafů

        # Graf 1: Average Reward
        plt.figure(figsize=(10, 5))
        plt.plot(eval_steps, avg_rewards, marker='o', linestyle='-', color='b')
        plt.title(f'Training Progress - Average Reward ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_reward_plot.png"))
        plt.close()

        # Graf 2: Success Rate
        plt.figure(figsize=(10, 5))
        plt.plot(eval_steps, success_rates, marker='o', linestyle='-', color='g')
        plt.title(f'Training Progress - Success Rate ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Success Rate')
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_success_plot.png"))
        plt.close()

        # Graf 3: Average Episode Length
        plt.figure(figsize=(10, 5))
        plt.plot(eval_steps, avg_lengths, marker='o', linestyle='-', color='r')
        plt.title(f'Training Progress - Average Episode Length ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Steps per Episode')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_length_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # Graf 4: stacked bar graf + označení success rate
        #
        # Každý sloupec = jeden model/checkpoint.
        # Celková výška sloupce = průměrná délka epizody.
        # Barevné části = kolik kroků průměrně politika strávila v dané stage.
        # Text nad sloupcem = success rate daného modelu.
        # ---------------------------------------------------------
        if len(avg_stage_steps_per_model) > 0:
            stage_matrix = np.array(avg_stage_steps_per_model, dtype=np.float64)

            plt.figure(figsize=(max(12, len(eval_steps) * 0.7), 6))

            x = np.arange(len(eval_steps))
            bottom = np.zeros(len(eval_steps), dtype=np.float64)

            for stage_id in range(num_eval_stages):
                plt.bar(
                    x,
                    stage_matrix[:, stage_id],
                    bottom=bottom,
                    label=f"Stage {stage_id}"
                )
                bottom += stage_matrix[:, stage_id]

            # -----------------------------------------------------
            # Označení success rate nad každým sloupcem
            # -----------------------------------------------------
            max_bar_height = float(np.max(bottom)) if len(bottom) > 0 else 0.0
            text_offset = max(5.0, 0.03 * max_bar_height)

            for i, success_rate in enumerate(success_rates):
                success_percent = success_rate * 100.0

                if success_rate > 0.0:
                    success_label = f"✓ {success_percent:.0f}%"
                else:
                    success_label = "✗ 0%"

                plt.text(
                    x[i],
                    bottom[i] + text_offset,
                    success_label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0
                )

            plt.title(
                f'Training Progress - Episode Length Split by Stage ({config.get("task")})'
            )
            plt.xlabel('Training Steps / Model Checkpoint')
            plt.ylabel('Average Steps per Episode')

            plt.xticks(
                x,
                [str(step) for step in eval_steps],
                rotation=45,
                ha="right"
            )

            # Aby se text nad sloupci neořízl
            plt.ylim(0, max_bar_height + 4 * text_offset)

            plt.legend(title="Stage")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "eval_stage_stacked_bar.png"))
            plt.close()

        log.info(
            f"Evaluation complete. Plots saved to {model_dir}. "
            f"Stage stacked bar saved as eval_stage_stacked_bar.png"
        )
        quit()
    # elif config.get("train_or_eval") == "eval":
    #     import re
    #     import matplotlib.pyplot as plt

    #     metasim_env = MetaSimVecEnv(scenario, task_name=config.get("task"), num_envs=config.get("num_envs", 1), sim=config.get("sim"))
    #     env = StableBaseline3VecEnv(metasim_env)

    #     # Oprava pro numpy
    #     sys.modules['numpy._core'] = np.core
    #     sys.modules['numpy._core.numeric'] = np.core.numeric

    #     model_dir = config.get("load_model_path")
    #     if not os.path.isdir(model_dir):
    #         log.error(f"Provided load_model_path is not a directory: {model_dir}")
    #         exit(1)

    #     # 1. Získání a seřazení modelů podle počtu kroků
    #     model_files = []
    #     pattern = re.compile(r"model_(\d+)\.zip")

    #     for filename in os.listdir(model_dir):
    #         match = pattern.match(filename)
    #         if match:
    #             step_count = int(match.group(1))
    #             model_files.append((step_count, filename))

    #     # Seřadíme podle kroku (step_count)
    #     model_files.sort(key=lambda x: x[0])

    #     if not model_files:
    #         log.warning("No 'model_{step}.zip' files found in directory.")
    #         exit(0)

    #     log.info(f"Found {len(model_files)} models. Starting evaluation...")

    #     # Data pro grafy
    #     eval_steps = []
    #     avg_rewards = []
    #     success_rates = []
    #     avg_lengths = []  # <--- NOVÉ: Seznam pro průměrné délky epizod

    #     # Počet epizod pro evaluaci jednoho modelu
    #     n_eval_episodes = config.get("eval_episodes", 20)

    #     # 2. Hlavní smyčka přes všechny modely
    #     for step_count, filename in model_files:
    #         full_path = os.path.join(model_dir, filename)
    #         log.info(f"Evaluating model: {filename} (Step: {step_count})")

    #         # Načtení modelu
    #         try:
    #             model = PPO.load(full_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    #         except Exception as e:
    #             log.error(f"Failed to load model {filename}: {e}")
    #             continue

    #         # Evaluace jednoho modelu
    #         episode_rewards = []
    #         episode_successes = []
    #         episode_lengths = []  # <--- NOVÉ: Ukládání délek pro aktuální model

    #         # Reset prostředí
    #         obs = env.reset()

    #         # Pomocné proměnné pro akumulaci v běžících epizodách
    #         current_rewards = np.zeros(env.num_envs)
    #         current_lengths = np.zeros(env.num_envs)  # <--- NOVÉ: Čítač kroků pro každé prostředí

    #         # Běžíme dokud nemáme dostatek dokončených epizod
    #         while len(episode_rewards) < n_eval_episodes:
    #             actions, _ = model.predict(obs, deterministic=True)
    #             obs, rewards, dones, infos = env.step(actions)

    #             current_rewards += rewards
    #             current_lengths += 1  # <--- NOVÉ: Zvýšení počtu kroků

    #             # Zpracování dokončených epizod
    #             for i in range(env.num_envs):
    #                 if dones[i]:
    #                     # Uložení celkové odměny
    #                     episode_rewards.append(current_rewards[i])
    #                     current_rewards[i] = 0

    #                     # Uložení délky epizody
    #                     episode_lengths.append(current_lengths[i]) # <--- NOVÉ
    #                     current_lengths[i] = 0                     # <--- NOVÉ: Reset čítače

    #                     # Zjištění success rate
    #                     is_success = infos[i].get("is_success", False)
    #                     episode_successes.append(1 if is_success else 0)

    #         # Výpočet statistik pro tento model
    #         mean_reward = np.mean(episode_rewards)
    #         success_rate = np.mean(episode_successes)
    #         mean_length = np.mean(episode_lengths) # <--- NOVÉ: Průměrná délka

    #         log.info(f" -> Mean Reward: {mean_reward:.2f}, Success: {success_rate:.2%}, Avg Length: {mean_length:.1f}")

    #         eval_steps.append(step_count)
    #         avg_rewards.append(mean_reward)
    #         success_rates.append(success_rate)
    #         avg_lengths.append(mean_length) # <--- NOVÉ

    #     env.close()

    #     # 3. Vykreslení a uložení grafů

    #     # Graf 1: Average Reward
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(eval_steps, avg_rewards, marker='o', linestyle='-', color='b')
    #     plt.title(f'Training Progress - Average Reward ({config.get("task")})')
    #     plt.xlabel('Training Steps')
    #     plt.ylabel('Average Reward')
    #     plt.grid(True)
    #     plt.savefig(os.path.join(model_dir, "eval_reward_plot.png"))
    #     plt.close()

    #     # Graf 2: Success Rate
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(eval_steps, success_rates, marker='o', linestyle='-', color='g')
    #     plt.title(f'Training Progress - Success Rate ({config.get("task")})')
    #     plt.xlabel('Training Steps')
    #     plt.ylabel('Success Rate')
    #     plt.ylim(-0.05, 1.05)
    #     plt.grid(True)
    #     plt.savefig(os.path.join(model_dir, "eval_success_plot.png"))
    #     plt.close()

    #     # Graf 3: Average Episode Length (NOVÉ)
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(eval_steps, avg_lengths, marker='o', linestyle='-', color='r') # Červená barva
    #     plt.title(f'Training Progress - Average Episode Length ({config.get("task")})')
    #     plt.xlabel('Training Steps')
    #     plt.ylabel('Average Steps per Episode')
    #     plt.grid(True)
    #     plt.savefig(os.path.join(model_dir, "eval_length_plot.png"))
    #     plt.close()

    #     log.info(f"Evaluation complete. Plots saved to {model_dir}")
    #     quit()

    elif config.get("train_or_eval") == "eval_video":
        import matplotlib.pyplot as plt

        metasim_env = MetaSimVecEnv(
            scenario,
            task_name=config.get("task"),
            num_envs=config.get("num_envs", 1),
            sim=config.get("sim")
        )
        env = StableBaseline3VecEnv(metasim_env)

        # TODO fix numpy module issue when loading model only for cluster training
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.numeric'] = np.core.numeric

        # Load model
        log.info(f"Loading model from {config.get('load_model_path')}")
        model = PPO.load(
            config.get("load_model_path"),
            env=env,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # --- Nastavení videa ---
        video_save_path = config.get("video_save_path")
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

        observation = ObsSaver(video_path=video_save_path)
        slow = config.get("video_slowdown", 3)

        # --- Data pro reward graf ---
        reward_steps = []
        reward_per_step_mean = []
        reward_per_step_env0 = []
        cumulative_reward_mean = []

        cumulative_reward = 0.0

        # --- NOVÉ: Data pro stage změny ---
        stage_per_step_env0 = []
        stage_change_steps = []
        stage_change_from_to = []

        previous_stage_env0 = None

        # Inference
        obs = env.reset()
        max_steps = config.get("eval_max_steps", 1000)

        for step in range(max_steps):
            actions, _ = model.predict(obs, deterministic=True)

            # -----------------------------------------------------
            # NOVÉ: Stage čteme PŘED env.step().
            # Tento krok tedy odpovídá stage, ve které politika
            # právě vybírá akci.
            # -----------------------------------------------------
            try:
                actual_stage = (
                    metasim_env.env.handler.task.reward_functions[0]
                    .actual_stage
                )

                if actual_stage is None:
                    current_stage_env0 = 0
                else:
                    current_stage_env0 = int(
                        actual_stage
                        .detach()
                        .cpu()
                        .numpy()[0]
                    )

            except Exception as e:
                log.warning(f"Could not read actual_stage during eval_video: {e}")
                current_stage_env0 = 0

            stage_per_step_env0.append(current_stage_env0)

            if previous_stage_env0 is None:
                previous_stage_env0 = current_stage_env0
            elif current_stage_env0 != previous_stage_env0:
                stage_change_steps.append(step)
                stage_change_from_to.append(
                    (previous_stage_env0, current_stage_env0)
                )
                previous_stage_env0 = current_stage_env0

            obs, rewards, dones, infos = env.step(actions)

            rewards_np = np.asarray(rewards, dtype=np.float64)

            # -----------------------------------------------------
            # Uložení rewardu pro graf
            # -----------------------------------------------------
            mean_reward = float(np.mean(rewards_np))
            env0_reward = float(rewards_np[0])

            cumulative_reward += mean_reward

            reward_steps.append(step)
            reward_per_step_mean.append(mean_reward)
            reward_per_step_env0.append(env0_reward)
            cumulative_reward_mean.append(cumulative_reward)

            # -----------------------------------------------------
            # Uložení stavu do videa
            # -----------------------------------------------------
            states = metasim_env.env.handler.get_states()
            for _ in range(slow):
                observation.add(states)

            print(
                f"Step reward: {rewards} at step {step}, "
                f"stage env0: {current_stage_env0}"
            )

            if dones.any():
                log.info(f"Episode finished after {step + 1} steps")
                # Pokud chceš video pouze první epizody, odkomentuj break:
                # break

        observation.save()
        log.info(f"🎬 Video saved to {video_save_path}")

        # ---------------------------------------------------------
        # Graf 1: Reward v každém kroku evaluace + změny stage
        # ---------------------------------------------------------
        reward_plot_path = os.path.join(
            os.path.dirname(video_save_path),
            "eval_video_reward_per_step.png"
        )

        plt.figure(figsize=(12, 5))

        # Pokud máš více envs, mean reward je průměr přes paralelní prostředí.
        plt.plot(
            reward_steps,
            reward_per_step_mean,
            marker='o',
            linestyle='-',
            label="Mean reward per step"
        )

        # Pro num_envs > 1 může být užitečné vidět i env 0.
        if env.num_envs > 1:
            plt.plot(
                reward_steps,
                reward_per_step_env0,
                marker='x',
                linestyle='--',
                label="Env 0 reward per step"
            )

        # ---------------------------------------------------------
        # NOVÉ: Svislé čáry v místech změny stage
        # ---------------------------------------------------------
        ymin, ymax = plt.ylim()
        text_y = ymax - 0.08 * (ymax - ymin)

        used_stage_label = False

        for change_step, (old_stage, new_stage) in zip(
            stage_change_steps,
            stage_change_from_to
        ):
            if not used_stage_label:
                label = "Stage change"
                used_stage_label = True
            else:
                label = None

            plt.axvline(
                x=change_step,
                linestyle='--',
                linewidth=1.5,
                alpha=0.8,
                label=label
            )

            plt.text(
                change_step,
                text_y,
                f"{old_stage}→{new_stage}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=8
            )

        plt.title(f"Evaluation Video - Reward per Step ({config.get('task')})")
        plt.xlabel("Evaluation step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(reward_plot_path)
        plt.close()

        log.info(f"Reward per step plot saved to: {reward_plot_path}")

        # ---------------------------------------------------------
        # Graf 2: Kumulativní reward během evaluace + změny stage
        # ---------------------------------------------------------
        cumulative_reward_plot_path = os.path.join(
            os.path.dirname(video_save_path),
            "eval_video_cumulative_reward.png"
        )

        plt.figure(figsize=(12, 5))
        plt.plot(
            reward_steps,
            cumulative_reward_mean,
            marker='o',
            linestyle='-',
            label="Cumulative mean reward"
        )

        ymin, ymax = plt.ylim()
        text_y = ymax - 0.08 * (ymax - ymin)

        used_stage_label = False

        for change_step, (old_stage, new_stage) in zip(
            stage_change_steps,
            stage_change_from_to
        ):
            if not used_stage_label:
                label = "Stage change"
                used_stage_label = True
            else:
                label = None

            plt.axvline(
                x=change_step,
                linestyle='--',
                linewidth=1.5,
                alpha=0.8,
                label=label
            )

            plt.text(
                change_step,
                text_y,
                f"{old_stage}→{new_stage}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=8
            )

        plt.title(f"Evaluation Video - Cumulative Reward ({config.get('task')})")
        plt.xlabel("Evaluation step")
        plt.ylabel("Cumulative reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(cumulative_reward_plot_path)
        plt.close()

        log.info(f"Cumulative reward plot saved to: {cumulative_reward_plot_path}")

        # ---------------------------------------------------------
        # Graf 3: Samostatný graf stage v čase
        # ---------------------------------------------------------
        stage_plot_path = os.path.join(
            os.path.dirname(video_save_path),
            "eval_video_stage_per_step.png"
        )

        plt.figure(figsize=(12, 4))
        plt.step(
            reward_steps,
            stage_per_step_env0,
            where="post",
            label="Stage env 0"
        )

        plt.title(f"Evaluation Video - Stage per Step ({config.get('task')})")
        plt.xlabel("Evaluation step")
        plt.ylabel("Stage")
        plt.yticks(sorted(set(stage_per_step_env0)))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(stage_plot_path)
        plt.close()

        log.info(f"Stage per step plot saved to: {stage_plot_path}")

        env.close()
        quit()


    elif config.get("train_or_eval") == "load_and_train":
        metasim_env = MetaSimVecEnv(scenario, task_name=config.get("task"), num_envs=config.get("num_envs", 1), sim=config.get("sim"))
        env = StableBaseline3VecEnv(metasim_env)
        eval_env = StableBaseline3VecEnv(metasim_env)
        # load the model
        #TODO fix numpy module issue when loading model only for cluster training

        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.numeric'] = np.core.numeric



        log.info(f"Loading model from {config.get('load_model_path')}")
        model = PPO.load(config.get("load_model_path"), env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        model.set_env(env)
        model.learn(total_timesteps=config.get("total_timesteps", 1_000_000),
                    callback=[
                    SaveModelCallback(save_path=config.get("model_save_path"), save_freq=config.get("model_save_freq", 1_000_000),task_name=config.get("task")),
                    TensorboardMetricsCallback(
                        log_dir=config.get("tensorboard_log", "./ppo_tensorboard/"),
                        log_interval=config.get("tensorboard_log_interval", 100_000),
                        max_stage=config.get(
                            "max_stage",
                            4
                            if "simple" in config.get("task", "")
                            else (6 if "chairman" in config.get("task", "") else 3),
                        ),
                        verbose=1,
                    ),
                    # EvalCallback(
                    # eval_env=eval_env,
                    # eval_freq=config.get("eval_freq", 10_000_000),
                    # n_eval_episodes=config.get("n_eval_episodes", 5),
                    # log_dir=config.get("eval_log_dir", "./eval_logs"),
                    # save_best=True,
                    # best_model_dir=config.get("best_model_dir", "./best_models"),
                    # eval_max_steps=config.get("eval_max_steps", 1000)
                    #     )
                    ],
                    progress_bar=True,)

        #Save the model
        task_name = scenario.task.__class__.__name__[:-3]
        model.save(config.get("model_save_path", f"ppo_{task_name}_{config.get('sim', 'unknown_sim')}"))
        log.info("Model saved. Ending the training and closing the environment.")
        env.close()
        quit()
    elif config.get("train_or_eval") == "train_rsl":
        from rsl_rl.algorithms.ppo import PPO as RSLPPO
        from rsl_rl.runners import OnPolicyRunner
        from RSL_walk_new_env import RSLRLMetaSimEnv
        from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent
        from rsl_rl.storage import RolloutStorage

        # wrapper pro RSL-RL prostředí
        metasim_env = MetaSimVecEnv(scenario, task_name=config.get("task"), num_envs=config.get("num_envs", 1), sim=config.get("sim"))
        env = RSLRLMetaSimEnv(metasim_env)

        storage = RolloutStorage(
            training_type="rl",
            num_envs=config.get("num_envs", 1),
            num_transitions_per_env=config.get("n_steps", 128),
            obs=env.obs_dict,
            actions_shape=env.num_actions,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        policy = ActorCritic(
            obs = env.obs_space,
            obs_groups = {"policy": ["joint_pos", "gyro_obs", "command_obs"],
                          "critic": ["joint_pos", "gyro_obs", "command_obs", "right_ankle_roll_link", "left_ankle_roll_link", "torso_link"]},
            num_actions=env.num_actions,
        )
        algo = RSLPPO(
            policy=policy,
            storage=storage,
        )

        # runner config – kolik kroků a počet envs
        runner_cfg = {
            "num_envs": env.num_envs,
            "num_steps_per_iter": config.get("n_steps", 128),
            "total_iters": config.get("total_iterations", 5000),
        }

        runner = OnPolicyRunner(env, algo, runner_cfg)
        runner.learn()

    elif config.get("train_or_eval") == "train_dagger":
        VIZUALIZATION = config.get("visualization", False)

        from dagger_vp.student_net import VisionStudent
        from dagger_vp.dagger_trainer import DAggerBuffer, train_dagger_step
        from torch.utils.tensorboard import SummaryWriter
        import cv2

        scenario.dagger = 1  # for evaluation of student model in env wrapper

        metasim_env = MetaSimVecEnv(
            scenario,
            task_name=config.get("task"),
            num_envs=config.get("num_envs", 1),
            sim=config.get("sim")
        )

        env = StableBaseline3VecEnv(metasim_env)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tb_log_dir = config.get("tensorboard_log", "./dagger_tensorboard/")
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)

        save_dir = config.get("model_save_path", "./output/dagger_models/")
        os.makedirs(save_dir, exist_ok=True)
        save_freq = config.get("model_save_freq", 5000)

        log.info(f"Loading Expert model from {config.get('load_model_path')}")

        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.numeric"] = np.core.numeric

        expert_model = PPO.load(
            config.get("load_model_path"),
            env=env,
            device=device,
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "_last_obs": None,
                "_last_episode_starts": None,
            },
        )

        num_actions = env.action_space.shape[0]
        num_joints = env.action_space.shape[0] - 3

        student_model = VisionStudent(
            num_actions=num_actions,
            num_joints=num_joints
        ).to(device)

        optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=config.get("learning_rate", 3e-4)
        )

        dagger_buffer_device = config.get("dagger_buffer_device", "cpu")
        dagger_buffer_pin_memory = config.get(
            "dagger_buffer_pin_memory",
            dagger_buffer_device == "cpu" and device.startswith("cuda")
        )

        buffer = DAggerBuffer(
            max_samples=config.get("dagger_buffer_steps", 4000)
            * config.get("dagger_store_per_step", 32),
            img_shape=(3, 128, 128),
            num_joints=num_joints,
            num_actions=num_actions,
            device=device,
            storage_device=dagger_buffer_device,
            pin_memory=dagger_buffer_pin_memory,
        )
        log.info(
            f"DAgger buffer storage: {buffer.storage_device} | "
            f"train device: {device} | "
            f"pin_memory: {buffer.pin_memory}"
        )

        total_iterations = config.get("total_timesteps", 100_000)
        beta = config.get("beta_start", 1.0)
        beta_decay = config.get("beta_decay", 0.9995)

        store_per_step = config.get("dagger_store_per_step", 32)
        train_every = config.get("dagger_train_every", 20)
        updates_per_train = config.get("dagger_updates_per_train", 10)
        train_batch_size = config.get("dagger_batch_size", 512)

        expert_obs = env.reset()

        if VIZUALIZATION:
            cv2.namedWindow("Student camera input", cv2.WINDOW_NORMAL)

        # ------------------------------------------------------------------
        # Počítadla úspěšnosti
        # ------------------------------------------------------------------

        total_completed_envs = 0
        total_successful_envs = 0

        recent_completed_envs = 0
        recent_successful_envs = 0

        log.info("Starting DAgger Training...")

        for step in range(total_iterations):
            states = metasim_env.env.handler.get_states()

            # Kamera: [N, H, W, C] -> [N, C, H, W]
            rgb_tensor = states.cameras["camera0"].rgb.to(device)
            rgb_permuted = rgb_tensor.permute(0, 3, 1, 2).contiguous().float()

            # Očekáváme kameru 128x128, bez resize
            student_obs = rgb_permuted
            student_obs_uint8 = student_obs.to(torch.uint8)
            student_obs_net_input = student_obs / 255.0

            # Jointy z observation; ve wrapperu jsou na začátku observation
            joint_obs_tensor = torch.as_tensor(
                expert_obs[:, 3:num_joints+3],
                device=device,
                dtype=torch.float32
            )

            with torch.no_grad():
                expert_actions, _ = expert_model.predict(
                    expert_obs,
                    deterministic=True
                )

                expert_actions_tensor = torch.as_tensor(
                    expert_actions,
                    device=device,
                    dtype=torch.float32
                )

                student_model.eval()
                student_actions_tensor = student_model(
                    student_obs_net_input,
                    joint_obs_tensor
                )

                student_actions = student_actions_tensor.cpu().numpy()

            # Směs expert/student
            if random.random() < beta:
                env_actions = expert_actions
            else:
                env_actions = student_actions

            # Do bufferu uložit jen část env, ne všech 100
            buffer.add_batch(
                student_obs_uint8,
                joint_obs_tensor,
                expert_actions_tensor,
                store_count=store_per_step
            )

            expert_obs, rewards, dones, infos = env.step(env_actions)

            # ------------------------------------------------------------------
            # Počítání dokončených a úspěšných envů
            # ------------------------------------------------------------------
            # completed_this_step:
            #   kolik envů v tomto kroku skončilo, tedy done == True
            #
            # successful_this_step:
            #   kolik z právě skončených envů mělo info["is_success"] == True
            # ------------------------------------------------------------------

            completed_this_step = int(np.sum(dones))

            successful_this_step = sum(
                int(info.get("is_success", False))
                for done, info in zip(dones, infos)
                if done
            )

            total_completed_envs += completed_this_step
            total_successful_envs += successful_this_step

            recent_completed_envs += completed_this_step
            recent_successful_envs += successful_this_step

            # Trénovat méně často, ale více gradient kroky
            if step > 0 and step % train_every == 0:
                losses = []

                for _ in range(updates_per_train):
                    loss = train_dagger_step(
                        student_model,
                        optimizer,
                        buffer,
                        batch_size=train_batch_size
                    )
                    losses.append(loss)

                mean_loss = float(np.mean(losses)) if losses else 0.0

                total_success_rate = (
                    total_successful_envs / total_completed_envs
                    if total_completed_envs > 0
                    else 0.0
                )

                recent_success_rate = (
                    recent_successful_envs / recent_completed_envs
                    if recent_completed_envs > 0
                    else 0.0
                )

                writer.add_scalar("DAgger/MSE_Loss", mean_loss, step)
                writer.add_scalar("DAgger/Beta_Mix_Ratio", beta, step)
                writer.add_scalar("DAgger/Env_Mean_Reward", float(np.mean(rewards)), step)
                writer.add_scalar("DAgger/Buffer_Size", buffer.size, step)

                writer.add_scalar(
                    "DAgger/Total_Completed_Envs",
                    total_completed_envs,
                    step
                )

                writer.add_scalar(
                    "DAgger/Total_Successful_Envs",
                    total_successful_envs,
                    step
                )

                writer.add_scalar(
                    "DAgger/Total_Success_Rate",
                    total_success_rate,
                    step
                )

                writer.add_scalar(
                    "DAgger/Recent_Completed_Envs",
                    recent_completed_envs,
                    step
                )

                writer.add_scalar(
                    "DAgger/Recent_Successful_Envs",
                    recent_successful_envs,
                    step
                )

                writer.add_scalar(
                    "DAgger/Recent_Success_Rate",
                    recent_success_rate,
                    step
                )

                log.info(
                    f"Step {step}/{total_iterations} | "
                    f"Beta: {beta:.4f} | "
                    f"Loss: {mean_loss:.6f} | "
                    f"Buffer: {buffer.size} | "
                    f"Success: {total_successful_envs}/{total_completed_envs} "
                    f"({total_success_rate:.3f}) | "
                    f"Recent: {recent_successful_envs}/{recent_completed_envs} "
                    f"({recent_success_rate:.3f})"
                )

                # Vynulování recent statistik po každém logování,
                # aby ukazovaly pouze období od posledního logu.
                recent_completed_envs = 0
                recent_successful_envs = 0

            if VIZUALIZATION and step % 5 == 0:
                img_vis = student_obs_uint8[0].permute(1, 2, 0).detach().cpu().numpy()
                img_vis_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

                cv2.imshow("Student camera input", img_vis_bgr)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    log.info("Visualization interrupted by user (ESC).")
                    break

            if step > 0 and step % save_freq == 0:
                current_save_path = os.path.join(
                    save_dir,
                    f"student_model_step_{step}.pth"
                )

                torch.save(student_model.state_dict(), current_save_path)
                log.info(f"Checkpoint saved to {current_save_path}")

            beta = max(0.0, beta * beta_decay)

        final_save_path = os.path.join(save_dir, "student_model_final.pth")
        torch.save(student_model.state_dict(), final_save_path)

        final_success_rate = (
            total_successful_envs / total_completed_envs
            if total_completed_envs > 0
            else 0.0
        )

        log.info(
            f"DAgger Training Finished! "
            f"Final model saved to {final_save_path} | "
            f"Final Success: {total_successful_envs}/{total_completed_envs} "
            f"({final_success_rate:.3f})"
        )

        writer.close()
        env.close()

        if VIZUALIZATION:
            cv2.destroyAllWindows()

    elif config.get("train_or_eval") == "eval_dagger_video":
        from dagger_vp.student_net import VisionStudent
        import cv2

        scenario.dagger = 2

        metasim_env = MetaSimVecEnv(
            scenario,
            task_name=config.get("task"),
            num_envs=config.get("num_envs", 1),
            sim=config.get("sim"),
        )
        env = StableBaseline3VecEnv(metasim_env)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        num_actions = env.action_space.shape[0]
        num_joints = env.action_space.shape[0] - 3

        student_model = VisionStudent(
            num_actions=num_actions,
            num_joints=num_joints
        ).to(device)

        model_path = config.get("load_model_path")
        log.info(f"Loading Student model from {model_path}")
        student_model.load_state_dict(torch.load(model_path, map_location=device))
        student_model.eval()

        video_path = config.get("video_save_path", "./output/dagger_fpv_video.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        video_width, video_height = 128, 128
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (video_width, video_height))

        obs = env.reset()
        max_steps = config.get("eval_max_steps", 1000)

        log.info("Starting DAgger Evaluation...")

        try:
            for step in range(max_steps):
                states = metasim_env.env.handler.get_states()

                # --- video frame ---
                rgb_tensor_raw = states.cameras["camera0"].rgb
                frame_np = rgb_tensor_raw[0].detach().cpu().numpy().astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

                # --- přesně stejný preprocessing jako v train_dagger ---
                rgb_tensor = rgb_tensor_raw.to(device)
                rgb_permuted = rgb_tensor.permute(0, 3, 1, 2).contiguous().float()
                student_obs_net_input = rgb_permuted / 255.0

                joint_obs_tensor = torch.as_tensor(
                    obs[:, 3:num_joints + 3],
                    device=device,
                    dtype=torch.float32
                )

                with torch.no_grad():
                    student_actions_tensor = student_model(student_obs_net_input, joint_obs_tensor)
                    actions = student_actions_tensor.cpu().numpy()

                # volitelné, ale bezpečnější: oříznout akce do rozsahu action space
                actions = np.clip(actions, env.action_space.low, env.action_space.high)

                obs, rewards, dones, infos = env.step(actions)

                # Workaround na bug ve wrapperu:
                # step_wait() resetne env interně, ale vrátí staré obs.
                # Proto po done znovu resetneme, aby se joint_obs synchronizovalo s kamerou.
                if np.any(dones):
                    log.info(
                        f"Episode finished at step {step + 1}, "
                        f"success={infos[0].get('is_success', False)}"
                    )
                    obs = env.reset()

                if step % 100 == 0:
                    log.info(
                        f"Eval step: {step}/{max_steps} | "
                        f"reward={float(np.mean(rewards)):.4f}"
                    )

        finally:
            video_writer.release()
            env.close()

        log.info(f"🎬 FPV Video saved successfully to: {video_path}")



    elif config.get("train_or_eval") == "eval_dagger":
        import re
        import csv
        import matplotlib.pyplot as plt
        from dagger_vp.student_net import VisionStudent
        import cv2

        scenario.dagger = 2

        metasim_env = MetaSimVecEnv(
            scenario,
            task_name=config.get("task"),
            num_envs=config.get("num_envs", 1),
            sim=config.get("sim"),
        )
        env = StableBaseline3VecEnv(metasim_env)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_dir = config.get("load_model_path")
        if not os.path.isdir(model_dir):
            log.error(f"Provided load_model_path is not a directory: {model_dir}")
            env.close()
            exit(1)

        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.numeric"] = np.core.numeric

        # ---------------------------------------------------------
        # Pomocná funkce: pozice base_linku židle v XY rovině
        # ---------------------------------------------------------
        def get_chair_base_xy():
            states = metasim_env.env.handler.get_states()
            chair = states.objects["chair"]
            chair_base_idx = chair.body_names.index("base_link")
            chair_pos_xy = chair.body_state[:, chair_base_idx, :2]
            return chair_pos_xy.detach().cpu().numpy().copy()

        # ---------------------------------------------------------
        # 1) najít a seřadit všechny DAgger checkpointy
        # ---------------------------------------------------------
        model_files = []

        step_pattern = re.compile(r"student_model_step_(\d+)\.pth$")
        final_pattern = re.compile(r"student_model_final\.pth$")

        for filename in os.listdir(model_dir):
            step_match = step_pattern.match(filename)
            if step_match:
                step_count = int(step_match.group(1))
                model_files.append((step_count, filename))
                continue

            if final_pattern.match(filename):
                model_files.append((10**18, filename))

        model_files.sort(key=lambda x: x[0])

        if not model_files:
            log.warning(
                "No DAgger checkpoints found: expected student_model_step_XXX.pth or student_model_final.pth"
            )
            env.close()
            exit(0)

        log.info(f"Found {len(model_files)} DAgger checkpoints. Starting evaluation...")

        # ---------------------------------------------------------
        # 2) rozměry modelu
        # ---------------------------------------------------------
        obs = env.reset()

        num_actions = env.action_space.shape[0]
        num_joints = env.action_space.shape[0] - 3

        requested_eval_episodes = config.get("eval_episodes", env.num_envs)

        if requested_eval_episodes > env.num_envs:
            log.warning(
                f"eval_episodes={requested_eval_episodes} is larger than env.num_envs={env.num_envs}. "
                f"Using only {env.num_envs} episodes, because this evaluation mode allows each env to count only once."
            )

        n_eval_episodes = min(requested_eval_episodes, env.num_envs)
        max_total_steps = config.get("eval_total_step_cap", 5000)
        count_timeouts_as_failures = config.get("eval_count_timeouts_as_failures", True)

        num_eval_stages = config.get("num_eval_stages", 4)

        # ---------------------------------------------------------
        # Data pro grafy
        # ---------------------------------------------------------
        eval_steps = []

        avg_rewards = []
        std_rewards = []

        success_rates = []

        avg_lengths = []
        std_lengths = []

        avg_chair_displacements = []
        std_chair_displacements = []

        clean_loads = []

        avg_stage_steps_per_model = []

        # NOVÉ: kolik envů skončilo v jednotlivých stage
        end_stage_counts_per_model = []

        # NOVÉ: kolik envů splnilo celou úlohu
        success_counts_per_model = []

        csv_rows = []

        # ---------------------------------------------------------
        # 3) evaluace všech checkpointů
        # ---------------------------------------------------------
        for step_count, filename in model_files:
            full_path = os.path.join(model_dir, filename)
            log.info(f"Evaluating DAgger checkpoint: {filename}")

            student_model = VisionStudent(
                num_actions=num_actions,
                num_joints=num_joints
            ).to(device)

            try:
                ckpt = torch.load(full_path, map_location=device)
                missing, unexpected = student_model.load_state_dict(ckpt, strict=False)
            except Exception as e:
                log.error(f"Failed to load model {filename}: {e}")
                continue

            if len(missing) > 0 or len(unexpected) > 0:
                log.warning(
                    f"Checkpoint {filename} not loaded cleanly | "
                    f"missing={missing} | unexpected={unexpected}"
                )
                clean_load = 0
            else:
                clean_load = 1

            student_model.eval()

            obs = env.reset()

            episode_rewards = []
            episode_successes = []
            episode_lengths = []
            episode_stage_counts = []
            episode_chair_displacements = []

            # NOVÉ: stage, ve které jednotlivé envy skončily
            episode_end_stages = []

            current_rewards = np.zeros(env.num_envs, dtype=np.float64)
            current_lengths = np.zeros(env.num_envs, dtype=np.int64)

            current_stage_counts = np.zeros(
                (env.num_envs, num_eval_stages),
                dtype=np.float64
            )

            # NOVÉ: poslední známá stage pro každý env
            last_stage_before_step = np.zeros(env.num_envs, dtype=np.int64)

            # Počáteční pozice židle pro každý env
            initial_chair_xy = get_chair_base_xy()

            # Poslední známá pozice židle před env.step().
            # Používá se při done, protože wrapper může po done env interně resetovat.
            last_chair_xy_before_step = initial_chair_xy.copy()

            active_eval_envs = np.zeros(env.num_envs, dtype=bool)
            active_eval_envs[:n_eval_episodes] = True

            finished_envs = np.zeros(env.num_envs, dtype=bool)

            total_steps = 0

            while (
                np.sum(finished_envs & active_eval_envs) < n_eval_episodes
                and total_steps < max_total_steps
            ):
                running_envs = active_eval_envs & (~finished_envs)

                # -----------------------------------------------------
                # Uložíme pozici židle před krokem.
                # -----------------------------------------------------
                current_chair_xy = get_chair_base_xy()
                last_chair_xy_before_step[running_envs] = current_chair_xy[running_envs]

                states = metasim_env.env.handler.get_states()

                # stejný preprocessing jako v train_dagger / eval_dagger_video
                rgb_tensor_raw = states.cameras["camera0"].rgb.to(device)
                rgb_permuted = rgb_tensor_raw.permute(0, 3, 1, 2).contiguous().float()
                student_obs_net_input = rgb_permuted / 255.0

                joint_obs_tensor = torch.as_tensor(
                    obs[:, 3:num_joints + 3],
                    device=device,
                    dtype=torch.float32
                )

                with torch.no_grad():
                    actions_t = student_model(student_obs_net_input, joint_obs_tensor)
                    actions = actions_t.cpu().numpy()

                actions = np.clip(actions, env.action_space.low, env.action_space.high)

                # Pro envy, které už byly vyhodnoceny, posíláme nulovou akci.
                # Tyto envy se ale už nepočítají do statistik.
                if np.any(~running_envs):
                    actions[~running_envs] = 0.0

                # ---------------------------------------------------------
                # Stage čteme PŘED krokem prostředí.
                # ---------------------------------------------------------
                try:
                    actual_stage = (
                        metasim_env.env.handler.task.reward_functions[0]
                        .actual_stage
                    )

                    if actual_stage is None:
                        stages_before_step = np.zeros(env.num_envs, dtype=np.int64)
                    else:
                        stages_before_step = (
                            actual_stage
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int64)
                        )

                except Exception as e:
                    log.warning(f"Could not read actual_stage during DAgger eval: {e}")
                    stages_before_step = np.zeros(env.num_envs, dtype=np.int64)

                stages_before_step = np.clip(
                    stages_before_step,
                    0,
                    num_eval_stages - 1
                )

                for i in range(env.num_envs):
                    if running_envs[i]:
                        current_stage_counts[i, stages_before_step[i]] += 1
                        last_stage_before_step[i] = stages_before_step[i]

                obs, rewards, dones, infos = env.step(actions)

                rewards = np.asarray(rewards, dtype=np.float64)
                dones = np.asarray(dones, dtype=bool)

                current_rewards[running_envs] += rewards[running_envs]
                current_lengths[running_envs] += 1

                total_steps += 1

                for i in range(env.num_envs):
                    if running_envs[i] and dones[i]:
                        is_success = infos[i].get("is_success", False)

                        chair_displacement = float(
                            np.linalg.norm(
                                last_chair_xy_before_step[i] - initial_chair_xy[i]
                            )
                        )

                        end_stage = int(last_stage_before_step[i])
                        end_stage = int(np.clip(end_stage, 0, num_eval_stages - 1))

                        episode_rewards.append(float(current_rewards[i]))
                        episode_lengths.append(int(current_lengths[i]))
                        episode_successes.append(1 if is_success else 0)
                        episode_stage_counts.append(current_stage_counts[i].copy())
                        episode_chair_displacements.append(chair_displacement)

                        # NOVÉ: uložíme, ve které stage env skončil
                        episode_end_stages.append(end_stage)

                        finished_envs[i] = True

                        log.info(
                            f"Eval env {i} finished once | "
                            f"episode={len(episode_rewards)}/{n_eval_episodes} | "
                            f"length={current_lengths[i]} | "
                            f"reward={current_rewards[i]:.4f} | "
                            f"chair_displacement_xy={chair_displacement:.4f} m | "
                            f"end_stage={end_stage} | "
                            f"success={is_success}"
                        )

                        current_rewards[i] = 0.0
                        current_lengths[i] = 0
                        current_stage_counts[i, :] = 0.0

                # DŮLEŽITÉ:
                # Nevoláme obs = env.reset() při np.any(dones).
                # Hotové envy ignorujeme pomocí finished_envs.

            episodes_finished = len(episode_rewards)

            if episodes_finished < n_eval_episodes:
                log.warning(
                    f"Checkpoint {filename}: only {episodes_finished}/{n_eval_episodes} "
                    f"eval envs finished before max_total_steps={max_total_steps}."
                )

                if count_timeouts_as_failures:
                    running_envs = active_eval_envs & (~finished_envs)
                    timeout_chair_xy = get_chair_base_xy()

                    for i in range(env.num_envs):
                        if running_envs[i]:
                            chair_displacement = float(
                                np.linalg.norm(
                                    timeout_chair_xy[i] - initial_chair_xy[i]
                                )
                            )

                            end_stage = int(last_stage_before_step[i])
                            end_stage = int(np.clip(end_stage, 0, num_eval_stages - 1))

                            episode_rewards.append(float(current_rewards[i]))
                            episode_lengths.append(int(current_lengths[i]))
                            episode_successes.append(0)
                            episode_stage_counts.append(current_stage_counts[i].copy())
                            episode_chair_displacements.append(chair_displacement)

                            # NOVÉ: timeout započítáme jako konec v poslední známé stage
                            episode_end_stages.append(end_stage)

                            finished_envs[i] = True

                            log.info(
                                f"Eval env {i} timed out | "
                                f"episode={len(episode_rewards)}/{n_eval_episodes} | "
                                f"length={current_lengths[i]} | "
                                f"reward={current_rewards[i]:.4f} | "
                                f"chair_displacement_xy={chair_displacement:.4f} m | "
                                f"end_stage={end_stage} | "
                                f"success=False"
                            )

                            if len(episode_rewards) >= n_eval_episodes:
                                break

            # ---------------------------------------------------------
            # Statistiky checkpointu
            # ---------------------------------------------------------
            if len(episode_rewards) == 0:
                mean_reward = 0.0
                std_reward = 0.0

                success_rate = 0.0
                success_count = 0

                mean_length = 0.0
                std_length = 0.0

                mean_stage_counts = np.zeros(num_eval_stages, dtype=np.float64)
                end_stage_counts = np.zeros(num_eval_stages, dtype=np.int64)

                mean_chair_displacement = 0.0
                std_chair_displacement = 0.0

                log.warning(f"No episodes finished for checkpoint {filename}")
            else:
                mean_reward = float(np.mean(episode_rewards))
                std_reward = float(np.std(episode_rewards))

                success_rate = float(np.mean(episode_successes))
                success_count = int(np.sum(episode_successes))

                mean_length = float(np.mean(episode_lengths))
                std_length = float(np.std(episode_lengths))

                if len(episode_stage_counts) > 0:
                    mean_stage_counts = np.mean(
                        np.array(episode_stage_counts, dtype=np.float64),
                        axis=0
                    )
                else:
                    mean_stage_counts = np.zeros(num_eval_stages, dtype=np.float64)

                mean_chair_displacement = float(np.mean(episode_chair_displacements))
                std_chair_displacement = float(np.std(episode_chair_displacements))

                # NOVÉ: počet envů, které skončily v jednotlivých stage
                end_stage_counts = np.zeros(num_eval_stages, dtype=np.int64)
                for end_stage in episode_end_stages:
                    end_stage = int(np.clip(end_stage, 0, num_eval_stages - 1))
                    end_stage_counts[end_stage] += 1

            if filename == "student_model_final.pth":
                x_value = (max(eval_steps) + 1) if len(eval_steps) > 0 else 0
            else:
                x_value = step_count

            eval_steps.append(x_value)

            avg_rewards.append(mean_reward)
            std_rewards.append(std_reward)

            success_rates.append(success_rate)

            avg_lengths.append(mean_length)
            std_lengths.append(std_length)

            avg_chair_displacements.append(mean_chair_displacement)
            std_chair_displacements.append(std_chair_displacement)

            clean_loads.append(clean_load)
            avg_stage_steps_per_model.append(mean_stage_counts)

            # NOVÉ: data pro graf koncových stage
            end_stage_counts_per_model.append(end_stage_counts.copy())
            success_counts_per_model.append(success_count)

            csv_row = {
                "checkpoint": filename,
                "x_step": x_value,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "success_rate": success_rate,
                "success_count": success_count,
                "mean_length": mean_length,
                "std_length": std_length,
                "mean_chair_displacement_xy": mean_chair_displacement,
                "std_chair_displacement_xy": std_chair_displacement,
                "episodes_finished": len(episode_rewards),
                "requested_eval_episodes": n_eval_episodes,
                "clean_load": clean_load,
                "missing_keys_count": len(missing),
                "unexpected_keys_count": len(unexpected),
            }

            for stage_id in range(num_eval_stages):
                csv_row[f"mean_stage_{stage_id}_steps"] = float(mean_stage_counts[stage_id])

            # NOVÉ: do CSV uložíme i počty envů končících ve stage
            for stage_id in range(num_eval_stages):
                csv_row[f"end_stage_{stage_id}_count"] = int(end_stage_counts[stage_id])

            csv_rows.append(csv_row)

            log.info(
                f" -> Mean Reward: {mean_reward:.4f}, "
                f"Std Reward: {std_reward:.4f}, "
                f"Success: {success_rate:.2%}, "
                f"Success Count: {success_count}/{n_eval_episodes}, "
                f"Avg Length: {mean_length:.1f}, "
                f"Std Length: {std_length:.1f}, "
                f"Avg Stage Steps: {mean_stage_counts}, "
                f"End Stage Counts: {end_stage_counts}, "
                f"Avg Chair Displacement XY: {mean_chair_displacement:.4f} m, "
                f"Std Chair Displacement XY: {std_chair_displacement:.4f} m, "
                f"Episodes: {len(episode_rewards)}/{n_eval_episodes}, "
                f"Clean load: {bool(clean_load)}"
            )

        env.close()

        # ---------------------------------------------------------
        # 4) uložit CSV
        # ---------------------------------------------------------
        csv_path = os.path.join(model_dir, "eval_dagger_results.csv")

        csv_fieldnames = [
            "checkpoint",
            "x_step",
            "mean_reward",
            "std_reward",
            "success_rate",
            "success_count",
            "mean_length",
            "std_length",
            "mean_chair_displacement_xy",
            "std_chair_displacement_xy",
            "episodes_finished",
            "requested_eval_episodes",
            "clean_load",
            "missing_keys_count",
            "unexpected_keys_count",
        ]

        for stage_id in range(num_eval_stages):
            csv_fieldnames.append(f"mean_stage_{stage_id}_steps")

        for stage_id in range(num_eval_stages):
            csv_fieldnames.append(f"end_stage_{stage_id}_count")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=csv_fieldnames,
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        eval_steps_np = np.array(eval_steps, dtype=np.float64)

        avg_rewards_np = np.array(avg_rewards, dtype=np.float64)
        std_rewards_np = np.array(std_rewards, dtype=np.float64)

        avg_lengths_np = np.array(avg_lengths, dtype=np.float64)
        std_lengths_np = np.array(std_lengths, dtype=np.float64)

        avg_chair_disp_np = np.array(avg_chair_displacements, dtype=np.float64)
        std_chair_disp_np = np.array(std_chair_displacements, dtype=np.float64)

        # ---------------------------------------------------------
        # 5) graf reward + rozptyl
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(
            eval_steps,
            avg_rewards,
            marker='o',
            linestyle='-',
            label="Mean reward"
        )

        plt.fill_between(
            eval_steps_np,
            avg_rewards_np - std_rewards_np,
            avg_rewards_np + std_rewards_np,
            alpha=0.2,
            label="±1 std"
        )

        plt.title(f'DAgger Evaluation - Average Reward ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_dagger_reward_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 6) graf success rate
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(eval_steps, success_rates, marker='o', linestyle='-')
        plt.title(f'DAgger Evaluation - Success Rate ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Success Rate')
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_dagger_success_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 7) graf délky epizody + rozptyl
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(
            eval_steps,
            avg_lengths,
            marker='o',
            linestyle='-',
            label="Mean episode length"
        )

        plt.fill_between(
            eval_steps_np,
            avg_lengths_np - std_lengths_np,
            avg_lengths_np + std_lengths_np,
            alpha=0.2,
            label="±1 std"
        )

        plt.title(f'DAgger Evaluation - Average Episode Length ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Steps per Episode')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_dagger_length_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 8) graf posunutí židle + rozptyl
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(
            eval_steps,
            avg_chair_displacements,
            marker='o',
            linestyle='-',
            label="Mean chair displacement"
        )

        plt.fill_between(
            eval_steps_np,
            avg_chair_disp_np - std_chair_disp_np,
            avg_chair_disp_np + std_chair_disp_np,
            alpha=0.2,
            label="±1 std"
        )

        plt.title(f'DAgger Evaluation - Chair Displacement ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Chair displacement in XY plane [m]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_dagger_chair_displacement_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 9) stacked bar graf + označení success rate
        # ---------------------------------------------------------
        if len(avg_stage_steps_per_model) > 0:
            stage_matrix = np.array(avg_stage_steps_per_model, dtype=np.float64)

            plt.figure(figsize=(max(12, len(eval_steps) * 0.7), 6))

            x = np.arange(len(eval_steps))
            bottom = np.zeros(len(eval_steps), dtype=np.float64)

            for stage_id in range(num_eval_stages):
                plt.bar(
                    x,
                    stage_matrix[:, stage_id],
                    bottom=bottom,
                    label=f"Stage {stage_id}"
                )
                bottom += stage_matrix[:, stage_id]

            max_bar_height = float(np.max(bottom)) if len(bottom) > 0 else 0.0
            text_offset = max(5.0, 0.03 * max_bar_height)

            for i, success_rate in enumerate(success_rates):
                success_percent = success_rate * 100.0

                if success_rate > 0.0:
                    success_label = f"✓ {success_percent:.0f}%"
                else:
                    success_label = "✗ 0%"

                plt.text(
                    x[i],
                    bottom[i] + text_offset,
                    success_label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0
                )

            plt.title(
                f'DAgger Evaluation - Episode Length Split by Stage ({config.get("task")})'
            )
            plt.xlabel('Training Steps / Checkpoint')
            plt.ylabel('Average Steps per Episode')

            plt.xticks(
                x,
                [str(step) for step in eval_steps],
                rotation=45,
                ha="right"
            )

            plt.ylim(0, max_bar_height + 4 * text_offset)

            plt.legend(title="Stage")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "eval_dagger_stage_stacked_bar.png"))
            plt.close()

        # ---------------------------------------------------------
        # 10) NOVÝ graf: kolik envů skončilo v jednotlivých stage
        #     + kolik envů splnilo celou úlohu
        # ---------------------------------------------------------
        if len(end_stage_counts_per_model) > 0:
            end_stage_matrix = np.array(end_stage_counts_per_model, dtype=np.int64)
            success_counts_np = np.array(success_counts_per_model, dtype=np.int64)

            plt.figure(figsize=(max(12, len(eval_steps) * 0.7), 6))

            for stage_id in range(num_eval_stages):
                plt.plot(
                    eval_steps,
                    end_stage_matrix[:, stage_id],
                    marker='o',
                    linestyle='-',
                    label=f"Ended in stage {stage_id}"
                )

            plt.plot(
                eval_steps,
                success_counts_np,
                marker='o',
                linestyle='--',
                linewidth=2.5,
                label="Task success"
            )

            plt.title(
                f"DAgger Evaluation - Number of Environments Ending in Each Stage ({config.get('task')})"
            )
            plt.xlabel("Training Steps / Checkpoint")
            plt.ylabel("Number of environments")

            plt.ylim(0, n_eval_episodes + 1)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "eval_dagger_end_stage_counts_plot.png"))
            plt.close()

        log.info(
            f"DAgger evaluation complete. Plots and CSV saved to {model_dir}. "
            f"Stage stacked bar saved as eval_dagger_stage_stacked_bar.png. "
            f"Chair displacement plot saved as eval_dagger_chair_displacement_plot.png. "
            f"End-stage count plot saved as eval_dagger_end_stage_counts_plot.png"
        )
        quit()
    elif config.get("train_or_eval") == "train_grpo":
        from grpo.student_net_stochastic import VisionStudent
        from grpo.grpo_trainer import collect_parallel_episodes, build_grpo_batch, grpo_update
        from torch.utils.tensorboard import SummaryWriter
        import gc
        import copy
        scenario.dagger = 2
        metasim_env = MetaSimVecEnv(
            scenario,
            task_name=config.get("task"),
            num_envs=config.get("num_envs", 1),
            sim=config.get("sim")
        )
        env = StableBaseline3VecEnv(metasim_env)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        num_actions = env.action_space.shape[0]
        num_joints = env.action_space.shape[0]

        student_model = VisionStudent(
            num_actions=num_actions,
            num_joints=num_joints
        ).to(device)

        model_path = config.get("load_model_path")
        log.info(f"Loading DAgger Student model from {model_path}")
        ckpt = torch.load(model_path, map_location=device)

        missing, unexpected = student_model.load_state_dict(ckpt, strict=False)
        log.info(f"Missing keys when loading DAgger checkpoint: {missing}")
        log.info(f"Unexpected keys when loading DAgger checkpoint: {unexpected}")

        # Vytvoření referenčního modelu (původní DAgger politika)
        ref_model = copy.deepcopy(student_model)

        # Zmrazení vah referenčního modelu (nebudou se trénovat)
        for param in ref_model.parameters():
            param.requires_grad = False

        # Přepnutí do eval módu pro jistotu (např. kvůli BatchNorm/Dropout, i když je síť možná nemá)
        ref_model.eval()



        optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=config.get("grpo_learning_rate", 1e-5)
        )

        tb_log_dir = config.get("tensorboard_log", "./grpo_tensorboard/")
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)

        save_dir = config.get("model_save_path", "./output/grpo_models/")
        os.makedirs(save_dir, exist_ok=True)
        save_freq = config.get("model_save_freq", 50)

        total_updates = config.get("grpo_total_updates", 2000)
        group_size = config.get("grpo_group_size", 4)
        rollouts_per_batch = config.get("grpo_rollouts_per_batch", env.num_envs)
        success_bonus = config.get("grpo_success_bonus", 20.0)

        if rollouts_per_batch % group_size != 0:
            raise ValueError("grpo_rollouts_per_batch musí být násobek grpo_group_size")

        log.info("Starting GRPO fine-tuning...")
        for update in range(total_updates):
            episodes = collect_parallel_episodes(
                env=env,
                metasim_env=metasim_env,
                policy=student_model,
                device=device,
                num_episodes=rollouts_per_batch,
                max_steps=1000,
                success_bonus=success_bonus,
            )

            batch, rollout_stats = build_grpo_batch(
                episodes=episodes,
                group_size=group_size,
            )
            del episodes
            gc.collect()

            update_stats = grpo_update(
                policy=student_model,
                ref_model=ref_model,
                optimizer=optimizer,
                batch=batch,
                device=device,
                kl_coef=config.get("grpo_kl_coef", 0.05),
                clip_eps=config.get("grpo_clip_eps", 0.2),
                ent_coef=config.get("grpo_ent_coef", 1e-3),
                epochs=config.get("grpo_update_epochs", 4),
                minibatch_size=config.get("grpo_minibatch_size", 2048),
                max_grad_norm=config.get("grpo_max_grad_norm", 1.0),
            )

            writer.add_scalar("GRPO/MeanReturn", rollout_stats["mean_return"], update)
            writer.add_scalar("GRPO/StdReturn", rollout_stats["std_return"], update)
            writer.add_scalar("GRPO/SuccessRate", rollout_stats["success_rate"], update)
            writer.add_scalar("GRPO/MeanEpisodeLength", rollout_stats["mean_length"], update)
            writer.add_scalar("GRPO/KL_Divergence", update_stats["kl_div"], update)
            writer.add_scalar("GRPO/Loss", update_stats["loss"], update)
            writer.add_scalar("GRPO/PolicyLoss", update_stats["policy_loss"], update)
            writer.add_scalar("GRPO/Entropy", update_stats["entropy"], update)
            writer.add_scalar("GRPO/RatioMean", update_stats["ratio_mean"], update)

            log.info(
                f"[GRPO] update={update}/{total_updates} | "
                f"return={rollout_stats['mean_return']:.4f} | "
                f"success={rollout_stats['success_rate']:.3f} | "
                f"len={rollout_stats['mean_length']:.1f} | "
                f"loss={update_stats['loss']:.6f} | "
                f"entropy={update_stats['entropy']:.6f} | "
                f"kl_div={update_stats['kl_div']:.6f}"
            )
            del batch
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if update > 0 and update % save_freq == 0:
                save_path = os.path.join(save_dir, f"student_grpo_step_{update}.pth")
                torch.save(student_model.state_dict(), save_path)
                log.info(f"Saved checkpoint to {save_path}")

        final_path = os.path.join(save_dir, "student_grpo_final.pth")
        torch.save(student_model.state_dict(), final_path)
        log.info(f"GRPO finished. Final checkpoint saved to {final_path}")

        writer.close()
        env.close()
    elif config.get("train_or_eval") == "eval_grpo_video":
        from grpo.student_net_stochastic import VisionStudent
        from grpo.grpo_trainer import get_student_inputs_from_states
        import cv2

        scenario.dagger = 2

        metasim_env = MetaSimVecEnv(
            scenario,
            task_name=config.get("task"),
            num_envs=config.get("num_envs", 1),
            sim=config.get("sim"),
        )
        env = StableBaseline3VecEnv(metasim_env)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        num_actions = env.action_space.shape[0]
        num_joints = env.action_space.shape[0]

        student_model = VisionStudent(
            num_actions=num_actions,
            num_joints=num_joints
        ).to(device)

        model_path = config.get("load_model_path")
        log.info(f"Loading GRPO Student model from {model_path}")
        ckpt = torch.load(model_path, map_location=device)

        missing, unexpected = student_model.load_state_dict(ckpt, strict=False)
        log.info(f"Missing keys: {missing}")
        log.info(f"Unexpected keys: {unexpected}")

        if len(missing) > 0 or len(unexpected) > 0:
            log.warning("Checkpoint was NOT loaded cleanly. This can easily explain bad evaluation.")

        student_model.eval()

        video_path = config.get("video_save_path", "./output/grpo_fpv_video.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        show_fpv = config.get("show_fpv_window", False)
        deterministic = config.get("eval_deterministic", True)
        max_steps = config.get("eval_max_steps", 1000)
        debug_every = config.get("eval_debug_every", 25)

        low_t = torch.as_tensor(env.action_space.low, device=device, dtype=torch.float32)
        high_t = torch.as_tensor(env.action_space.high, device=device, dtype=torch.float32)

        _ = env.reset()

        # Zjistíme rozměr kamery přímo z prvního snímku
        imgs_u8, imgs_f32, joints_f32 = get_student_inputs_from_states(metasim_env, device)
        _, _, H, W = imgs_u8.shape  # imgs_u8 je CHW

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (W, H))

        log.info("Starting GRPO evaluation with FPV visualization...")

        episode_idx = 0

        try:
            for step in range(max_steps):
                # Použij přesně stejný input pipeline jako při train collectu
                imgs_u8, imgs_f32, joints_f32 = get_student_inputs_from_states(metasim_env, device)

                # Frame pro zobrazení / video (env 0)
                                # Frame pro zobrazení / video (env 0)
                frame_np = imgs_u8[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                with torch.no_grad():
                    actions_t, logp_t, entropy_t, mean_t = student_model.act(
                        imgs_f32, joints_f32, deterministic=deterministic
                    )
                    actions_t = torch.max(torch.min(actions_t, high_t), low_t)

                actions_np = actions_t.detach().cpu().numpy()
                obs, rewards, dones, infos = env.step(actions_np)

                # diagnostika pro env 0
                action_abs_mean = float(actions_t[0].abs().mean().item())
                mean_abs_mean = float(mean_t[0].abs().mean().item())
                joint_abs_mean = float(joints_f32[0].abs().mean().item())
                img_mean = float(imgs_f32[0].mean().item())
                img_min = float(imgs_f32[0].min().item())
                img_max = float(imgs_f32[0].max().item())
                policy_std_mean = float(torch.exp(student_model.log_std).mean().item())

                # Zapiš čistý frame bez overlay textu
                video_writer.write(frame_bgr)

                # Volitelné zobrazení čistého FPV okna bez overlaye
                if show_fpv:
                    cv2.imshow("GRPO - what robot sees", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        log.info("Evaluation interrupted by user.")
                        break
                if np.any(dones):
                    for i in range(len(dones)):
                        if dones[i]:
                            episode_idx += 1
                            log.info(
                                f"[Eval] episode={episode_idx} finished at global_step={step + 1} | "
                                f"env={i} | reward={float(rewards[i]):.4f} | "
                                f"success={infos[i].get('is_success', False)} | "
                                f"truncated={infos[i].get('TimeLimit.truncated', False)}"
                            )

                if step % debug_every == 0:
                    log.info(
                        f"[EvalDebug] step={step}/{max_steps} | "
                        f"reward={float(rewards[0]):.4f} | "
                        f"done={bool(dones[0])} | "
                        f"success={bool(infos[0].get('is_success', False))} | "
                        f"action_abs_mean={action_abs_mean:.6f} | "
                        f"mean_abs_mean={mean_abs_mean:.6f} | "
                        f"policy_std_mean={policy_std_mean:.6f} | "
                        f"joint_abs_mean={joint_abs_mean:.6f} | "
                        f"img_mean={img_mean:.6f}"
                    )

        finally:
            video_writer.release()
            if show_fpv:
                cv2.destroyAllWindows()
            env.close()

        log.info(f"GRPO FPV Video saved successfully to: {video_path}")


    elif config.get("train_or_eval") == "eval_grpo":
        import re
        import csv
        import matplotlib.pyplot as plt
        from grpo.student_net_stochastic import VisionStudent
        from grpo.grpo_trainer import get_student_inputs_from_states

        scenario.dagger = 2

        # Prostředí vytvořit JEN JEDNOU
        metasim_env = MetaSimVecEnv(
            scenario,
            task_name=config.get("task"),
            num_envs=config.get("num_envs", 1),
            sim=config.get("sim"),
        )
        env = StableBaseline3VecEnv(metasim_env)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_dir = config.get("load_model_path")
        if not os.path.isdir(model_dir):
            log.error(f"Provided load_model_path is not a directory: {model_dir}")
            env.close()
            exit(1)

        # Kompatibilita jako u PPO eval
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.numeric"] = np.core.numeric

        # ---------------------------------------------------------
        # Pomocná funkce: pozice base_linku židle v XY rovině
        # ---------------------------------------------------------
        def get_chair_base_xy():
            states = metasim_env.env.handler.get_states()
            chair = states.objects["chair"]
            chair_base_idx = chair.body_names.index("base_link")
            chair_pos_xy = chair.body_state[:, chair_base_idx, :2]
            return chair_pos_xy.detach().cpu().numpy().copy()

        # ---------------------------------------------------------
        # 1) najít a seřadit všechny GRPO checkpointy
        # ---------------------------------------------------------
        model_files = []

        step_pattern = re.compile(r"student_grpo_step_(\d+)\.pth$")
        final_pattern = re.compile(r"student_grpo_final\.pth$")

        for filename in os.listdir(model_dir):
            step_match = step_pattern.match(filename)
            if step_match:
                step_count = int(step_match.group(1))
                model_files.append((step_count, filename))
                continue

            if final_pattern.match(filename):
                model_files.append((10**18, filename))

        model_files.sort(key=lambda x: x[0])

        if not model_files:
            log.warning(
                "No GRPO checkpoints found: expected student_grpo_step_XXX.pth or student_grpo_final.pth"
            )
            env.close()
            exit(0)

        log.info(f"Found {len(model_files)} GRPO checkpoints. Starting evaluation...")

        # ---------------------------------------------------------
        # 2) zjistit rozměry modelu ze skutečného stavu
        # ---------------------------------------------------------
        _ = env.reset()
        states = metasim_env.env.handler.get_states()
        robot_name = metasim_env.scenario.robots[0].name

        num_actions = env.action_space.shape[0]
        num_joints = int(states.robots[robot_name].joint_pos.shape[1])

        low_t = torch.as_tensor(env.action_space.low, device=device, dtype=torch.float32)
        high_t = torch.as_tensor(env.action_space.high, device=device, dtype=torch.float32)

        deterministic = config.get("eval_deterministic", True)

        requested_eval_episodes = config.get("eval_episodes", env.num_envs)

        if requested_eval_episodes > env.num_envs:
            log.warning(
                f"eval_episodes={requested_eval_episodes} is larger than env.num_envs={env.num_envs}. "
                f"Using only {env.num_envs} episodes, because this evaluation mode allows each env to count only once."
            )

        # Každý env může odpovídat maximálně jedné epizodě.
        n_eval_episodes = min(requested_eval_episodes, env.num_envs)

        max_total_steps = config.get("eval_total_step_cap", 1000)

        # Když env neskončí do max_total_steps, započítáme ho jako neúspěch.
        count_timeouts_as_failures = config.get("eval_count_timeouts_as_failures", True)

        # Data pro grafy
        eval_steps = []

        avg_rewards = []
        std_rewards = []

        success_rates = []

        avg_lengths = []
        std_lengths = []

        avg_chair_displacements = []
        std_chair_displacements = []

        loaded_cleanly = []

        csv_rows = []

        # ---------------------------------------------------------
        # 3) hlavní smyčka přes všechny checkpointy
        # ---------------------------------------------------------
        for step_count, filename in model_files:
            full_path = os.path.join(model_dir, filename)
            log.info(f"Evaluating GRPO checkpoint: {filename}")

            student_model = VisionStudent(
                num_actions=num_actions,
                num_joints=num_joints
            ).to(device)

            try:
                ckpt = torch.load(full_path, map_location=device)
                missing, unexpected = student_model.load_state_dict(ckpt, strict=False)
            except Exception as e:
                log.error(f"Failed to load model {filename}: {e}")
                continue

            if len(missing) > 0 or len(unexpected) > 0:
                log.warning(
                    f"Checkpoint {filename} not loaded cleanly | "
                    f"missing={missing} | unexpected={unexpected}"
                )
                clean_load = 0
            else:
                clean_load = 1

            student_model.eval()

            # Reset prostředí před evaluací checkpointu
            _ = env.reset()

            episode_rewards = []
            episode_successes = []
            episode_lengths = []

            # NOVÉ: posunutí židle pro každou epizodu/env
            episode_chair_displacements = []

            current_rewards = np.zeros(env.num_envs, dtype=np.float64)
            current_lengths = np.zeros(env.num_envs, dtype=np.int64)

            # Počáteční pozice židle pro každý env
            initial_chair_xy = get_chair_base_xy()

            # Poslední známá pozice před stepem.
            # Používáme ji při done, protože env.step() může po done interně resetovat env.
            last_chair_xy_before_step = initial_chair_xy.copy()

            # ---------------------------------------------------------
            # DŮLEŽITÉ:
            # active_eval_envs říká, které envy počítáme do evaluace.
            # finished_envs říká, které z nich už dokončily svoji jedinou epizodu.
            #
            # Tím zajistíme:
            # 100 envů = 100 epizod,
            # každý env se započítá maximálně jednou.
            # ---------------------------------------------------------
            active_eval_envs = np.zeros(env.num_envs, dtype=bool)
            active_eval_envs[:n_eval_episodes] = True

            finished_envs = np.zeros(env.num_envs, dtype=bool)

            total_steps = 0

            while (
                np.sum(finished_envs & active_eval_envs) < n_eval_episodes
                and total_steps < max_total_steps
            ):
                running_envs = active_eval_envs & (~finished_envs)

                # Uložíme pozici židle před krokem.
                # Pokud wrapper po done resetuje env, máme pořád poslední validní pozici před ukončením.
                current_chair_xy = get_chair_base_xy()
                last_chair_xy_before_step[running_envs] = current_chair_xy[running_envs]

                imgs_u8, imgs_f32, joints_f32 = get_student_inputs_from_states(
                    metasim_env,
                    device
                )

                with torch.no_grad():
                    actions_t, _, _, _ = student_model.act(
                        imgs_f32,
                        joints_f32,
                        deterministic=deterministic
                    )

                    actions_t = torch.max(torch.min(actions_t, high_t), low_t)

                actions_np = actions_t.detach().cpu().numpy()

                # Pro envy, které už byly vyhodnoceny, posíláme nulovou akci.
                # Do statistik se ale už nikdy nezapočítají.
                if np.any(~running_envs):
                    actions_np[~running_envs] = 0.0

                _, rewards, dones, infos = env.step(actions_np)

                rewards = np.asarray(rewards, dtype=np.float64)
                dones = np.asarray(dones, dtype=bool)

                # Reward a délku počítáme pouze pro aktivní envy,
                # které ještě nebyly započítány.
                current_rewards[running_envs] += rewards[running_envs]
                current_lengths[running_envs] += 1

                total_steps += 1

                for i in range(env.num_envs):
                    # Započítáváme pouze první dokončení daného envu.
                    if running_envs[i] and dones[i]:
                        is_success = infos[i].get("is_success", False)

                        # Posunutí židle v XY rovině od začátku epizody do posledního kroku před done.
                        chair_displacement = float(
                            np.linalg.norm(
                                last_chair_xy_before_step[i] - initial_chair_xy[i]
                            )
                        )

                        episode_rewards.append(float(current_rewards[i]))
                        episode_lengths.append(int(current_lengths[i]))
                        episode_successes.append(1 if is_success else 0)
                        episode_chair_displacements.append(chair_displacement)

                        finished_envs[i] = True

                        log.info(
                            f"Eval env {i} finished once | "
                            f"episode={len(episode_rewards)}/{n_eval_episodes} | "
                            f"length={current_lengths[i]} | "
                            f"reward={current_rewards[i]:.4f} | "
                            f"chair_displacement_xy={chair_displacement:.4f} m | "
                            f"success={is_success}"
                        )

                        # Reset lokálních statistik pro pořádek.
                        # Env se může interně resetovat, ale díky finished_envs[i] = True
                        # už nebude znovu započítán.
                        current_rewards[i] = 0.0
                        current_lengths[i] = 0

                # DŮLEŽITÉ:
                # Nevoláme env.reset() po done.
                # Pokud by se resetovalo celé vektorové prostředí,
                # zničily by se běžící epizody ostatních envů.

            episodes_finished = len(episode_rewards)

            if episodes_finished < n_eval_episodes:
                log.warning(
                    f"Checkpoint {filename}: only {episodes_finished}/{n_eval_episodes} "
                    f"eval envs finished before max_total_steps={max_total_steps}."
                )

                if count_timeouts_as_failures:
                    running_envs = active_eval_envs & (~finished_envs)

                    # Aktuální pozice židle při timeoutu
                    timeout_chair_xy = get_chair_base_xy()

                    for i in range(env.num_envs):
                        if running_envs[i]:
                            chair_displacement = float(
                                np.linalg.norm(
                                    timeout_chair_xy[i] - initial_chair_xy[i]
                                )
                            )

                            episode_rewards.append(float(current_rewards[i]))
                            episode_lengths.append(int(current_lengths[i]))
                            episode_successes.append(0)
                            episode_chair_displacements.append(chair_displacement)

                            finished_envs[i] = True

                            log.info(
                                f"Eval env {i} timed out | "
                                f"episode={len(episode_rewards)}/{n_eval_episodes} | "
                                f"length={current_lengths[i]} | "
                                f"reward={current_rewards[i]:.4f} | "
                                f"chair_displacement_xy={chair_displacement:.4f} m | "
                                f"success=False"
                            )

                            if len(episode_rewards) >= n_eval_episodes:
                                break

            if len(episode_rewards) == 0:
                mean_reward = 0.0
                std_reward = 0.0

                success_rate = 0.0

                mean_length = 0.0
                std_length = 0.0

                mean_chair_displacement = 0.0
                std_chair_displacement = 0.0

                log.warning(f"No episodes finished for checkpoint {filename}")
            else:
                mean_reward = float(np.mean(episode_rewards))
                std_reward = float(np.std(episode_rewards))

                success_rate = float(np.mean(episode_successes))

                mean_length = float(np.mean(episode_lengths))
                std_length = float(np.std(episode_lengths))

                mean_chair_displacement = float(np.mean(episode_chair_displacements))
                std_chair_displacement = float(np.std(episode_chair_displacements))

            if filename == "student_grpo_final.pth":
                x_value = (max(eval_steps) + 1) if len(eval_steps) > 0 else 0
            else:
                x_value = step_count

            eval_steps.append(x_value)

            avg_rewards.append(mean_reward)
            std_rewards.append(std_reward)

            success_rates.append(success_rate)

            avg_lengths.append(mean_length)
            std_lengths.append(std_length)

            avg_chair_displacements.append(mean_chair_displacement)
            std_chair_displacements.append(std_chair_displacement)

            loaded_cleanly.append(clean_load)

            csv_rows.append({
                "checkpoint": filename,
                "x_step": x_value,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "success_rate": success_rate,
                "mean_length": mean_length,
                "std_length": std_length,
                "mean_chair_displacement_xy": mean_chair_displacement,
                "std_chair_displacement_xy": std_chair_displacement,
                "episodes_finished": len(episode_rewards),
                "requested_eval_episodes": n_eval_episodes,
                "clean_load": clean_load,
                "missing_keys_count": len(missing),
                "unexpected_keys_count": len(unexpected),
            })

            log.info(
                f" -> Mean Reward: {mean_reward:.4f}, "
                f"Std Reward: {std_reward:.4f}, "
                f"Success: {success_rate:.2%}, "
                f"Avg Length: {mean_length:.1f}, "
                f"Std Length: {std_length:.1f}, "
                f"Avg Chair Displacement XY: {mean_chair_displacement:.4f} m, "
                f"Std Chair Displacement XY: {std_chair_displacement:.4f} m, "
                f"Episodes: {len(episode_rewards)}/{n_eval_episodes}, "
                f"Clean load: {bool(clean_load)}"
            )

        env.close()

        # ---------------------------------------------------------
        # 4) uložit CSV
        # ---------------------------------------------------------
        csv_path = os.path.join(model_dir, "eval_grpo_results.csv")

        csv_fieldnames = [
            "checkpoint",
            "x_step",
            "mean_reward",
            "std_reward",
            "success_rate",
            "mean_length",
            "std_length",
            "mean_chair_displacement_xy",
            "std_chair_displacement_xy",
            "episodes_finished",
            "requested_eval_episodes",
            "clean_load",
            "missing_keys_count",
            "unexpected_keys_count",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=csv_fieldnames,
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        # Převod na numpy kvůli fill_between
        eval_steps_np = np.array(eval_steps, dtype=np.float64)

        avg_rewards_np = np.array(avg_rewards, dtype=np.float64)
        std_rewards_np = np.array(std_rewards, dtype=np.float64)

        avg_lengths_np = np.array(avg_lengths, dtype=np.float64)
        std_lengths_np = np.array(std_lengths, dtype=np.float64)

        avg_chair_disp_np = np.array(avg_chair_displacements, dtype=np.float64)
        std_chair_disp_np = np.array(std_chair_displacements, dtype=np.float64)

        # ---------------------------------------------------------
        # 5) graf reward + rozptyl
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(
            eval_steps,
            avg_rewards,
            marker='o',
            linestyle='-',
            label="Mean reward"
        )

        plt.fill_between(
            eval_steps_np,
            avg_rewards_np - std_rewards_np,
            avg_rewards_np + std_rewards_np,
            alpha=0.2,
            label="±1 std"
        )

        plt.title(f'GRPO Evaluation - Average Reward ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_grpo_reward_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 6) graf success rate
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(eval_steps, success_rates, marker='o', linestyle='-')
        plt.title(f'GRPO Evaluation - Success Rate ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Success Rate')
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_grpo_success_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 7) graf délky epizody + rozptyl
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(
            eval_steps,
            avg_lengths,
            marker='o',
            linestyle='-',
            label="Mean episode length"
        )

        plt.fill_between(
            eval_steps_np,
            avg_lengths_np - std_lengths_np,
            avg_lengths_np + std_lengths_np,
            alpha=0.2,
            label="±1 std"
        )

        plt.title(f'GRPO Evaluation - Average Episode Length ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Steps per Episode')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_grpo_length_plot.png"))
        plt.close()

        # ---------------------------------------------------------
        # 8) graf posunutí židle + rozptyl
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(
            eval_steps,
            avg_chair_displacements,
            marker='o',
            linestyle='-',
            label="Mean chair displacement"
        )

        plt.fill_between(
            eval_steps_np,
            avg_chair_disp_np - std_chair_disp_np,
            avg_chair_disp_np + std_chair_disp_np,
            alpha=0.2,
            label="±1 std"
        )

        plt.title(f'GRPO Evaluation - Chair Displacement ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Chair displacement in XY plane [m]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "eval_grpo_chair_displacement_plot.png"))
        plt.close()

        log.info(
            f"GRPO evaluation complete. Plots and CSV saved to {model_dir}. "
            f"Chair displacement plot saved as eval_grpo_chair_displacement_plot.png"
        )
        quit()

if __name__ == "__main__":
    main()

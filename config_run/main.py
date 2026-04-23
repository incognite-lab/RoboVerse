import sys
import os
import yaml
import torch
import random

from loguru import logger as log

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg, GyroSensorCfg, CommandCfg
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
        config_name = "g1_ChairMan_simple"
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
    elif config.get("task") == "chairman":
        from SB3_chairman_env import StableBaseline3VecEnv
        if scenario.robots[0].fix_base_link == False:
            scenario.robots[0].urdf_path = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"
            scenario.robots[0].fix_base_link = False
    elif config.get("task") == "chairmansimple" or config.get("task") == "chairmansimplegrpo":
        from SB3_chairman_env import StableBaseline3VecEnv
        if scenario.robots[0].fix_base_link == False:
            scenario.robots[0].urdf_path = "roboverse_data/robots/g1/urdf/g1_mygym_with_world.urdf"
            scenario.robots[0].fix_base_link = False



    #-----------------------------------------------
    if config.get("net_arch_pivf", False):
        policy_kwargs = dict({"net_arch":{"pi": config.get("net_arch_pi", [128, 128, 128]),
                                         "vf": config.get("net_arch_vf", [128, 128, 128])}})
    else:
        policy_kwargs = dict({"net_arch": config.get("net_arch", [128, 128, 128])})
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
                        log_interval=1000000,
                        max_stage=3,
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
        import matplotlib.pyplot as plt

        metasim_env = MetaSimVecEnv(scenario, task_name=config.get("task"), num_envs=config.get("num_envs", 1), sim=config.get("sim"))
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

        # Seřadíme podle kroku (step_count)
        model_files.sort(key=lambda x: x[0])

        if not model_files:
            log.warning("No 'model_{step}.zip' files found in directory.")
            exit(0)

        log.info(f"Found {len(model_files)} models. Starting evaluation...")

        # Data pro grafy
        eval_steps = []
        avg_rewards = []
        success_rates = []
        avg_lengths = []  # <--- NOVÉ: Seznam pro průměrné délky epizod

        # Počet epizod pro evaluaci jednoho modelu
        n_eval_episodes = config.get("eval_episodes", 20)

        # 2. Hlavní smyčka přes všechny modely
        for step_count, filename in model_files:
            full_path = os.path.join(model_dir, filename)
            log.info(f"Evaluating model: {filename} (Step: {step_count})")

            # Načtení modelu
            try:
                model = PPO.load(full_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
            except Exception as e:
                log.error(f"Failed to load model {filename}: {e}")
                continue

            # Evaluace jednoho modelu
            episode_rewards = []
            episode_successes = []
            episode_lengths = []  # <--- NOVÉ: Ukládání délek pro aktuální model

            # Reset prostředí
            obs = env.reset()

            # Pomocné proměnné pro akumulaci v běžících epizodách
            current_rewards = np.zeros(env.num_envs)
            current_lengths = np.zeros(env.num_envs)  # <--- NOVÉ: Čítač kroků pro každé prostředí

            # Běžíme dokud nemáme dostatek dokončených epizod
            while len(episode_rewards) < n_eval_episodes:
                actions, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(actions)

                current_rewards += rewards
                current_lengths += 1  # <--- NOVÉ: Zvýšení počtu kroků

                # Zpracování dokončených epizod
                for i in range(env.num_envs):
                    if dones[i]:
                        # Uložení celkové odměny
                        episode_rewards.append(current_rewards[i])
                        current_rewards[i] = 0

                        # Uložení délky epizody
                        episode_lengths.append(current_lengths[i]) # <--- NOVÉ
                        current_lengths[i] = 0                     # <--- NOVÉ: Reset čítače

                        # Zjištění success rate
                        is_success = infos[i].get("is_success", False)
                        episode_successes.append(1 if is_success else 0)

            # Výpočet statistik pro tento model
            mean_reward = np.mean(episode_rewards)
            success_rate = np.mean(episode_successes)
            mean_length = np.mean(episode_lengths) # <--- NOVÉ: Průměrná délka

            log.info(f" -> Mean Reward: {mean_reward:.2f}, Success: {success_rate:.2%}, Avg Length: {mean_length:.1f}")

            eval_steps.append(step_count)
            avg_rewards.append(mean_reward)
            success_rates.append(success_rate)
            avg_lengths.append(mean_length) # <--- NOVÉ

        env.close()

        # 3. Vykreslení a uložení grafů

        # Graf 1: Average Reward
        plt.figure(figsize=(10, 5))
        plt.plot(eval_steps, avg_rewards, marker='o', linestyle='-', color='b')
        plt.title(f'Training Progress - Average Reward ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Reward')
        plt.grid(True)
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
        plt.savefig(os.path.join(model_dir, "eval_success_plot.png"))
        plt.close()

        # Graf 3: Average Episode Length (NOVÉ)
        plt.figure(figsize=(10, 5))
        plt.plot(eval_steps, avg_lengths, marker='o', linestyle='-', color='r') # Červená barva
        plt.title(f'Training Progress - Average Episode Length ({config.get("task")})')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Steps per Episode')
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, "eval_length_plot.png"))
        plt.close()

        log.info(f"Evaluation complete. Plots saved to {model_dir}")
        quit()

    elif config.get("train_or_eval") == "eval_video":
        metasim_env = MetaSimVecEnv(scenario, task_name=config.get("task"), num_envs=config.get("num_envs", 1), sim=config.get("sim"))
        env = StableBaseline3VecEnv(metasim_env)
        #TODO fix numpy module issue when loading model only for cluster training
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.numeric'] = np.core.numeric

        # load the model
        log.info(f"Loading model from {config.get('load_model_path')}")
        model = PPO.load(config.get("load_model_path"), env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        # --- Nastavení videa ---
        os.makedirs(os.path.dirname(config.get("video_save_path")), exist_ok=True)
        observation = ObsSaver(video_path=config.get("video_save_path"))
        slow = config.get("video_slowdown", 3)
        # inference
        obs = env.reset()
        for step in range(config.get("eval_max_steps", 1000)):
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            states = metasim_env.env.handler.get_states()
            for _ in range(slow):
                observation.add(states)

            print(f"Step reward: {rewards} at step {step}")

            if dones.any():
                log.info(f"Episode finished after {step + 1} steps")
        observation.save()
        log.info(f"🎬 Video saved to {config.get('video_path')}")
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
                        log_interval=100000,
                        max_stage=3,
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
        from dagger.student_net import VisionStudent
        from dagger.dagger_trainer import DAggerBuffer, train_dagger_step
        from torch.utils.tensorboard import SummaryWriter
        import cv2
        scenario.dagger = 1 #for evaluation of student model in env wrapper
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
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.numeric'] = np.core.numeric
        expert_model = PPO.load(config.get("load_model_path"), env=env, device=device)

        num_actions = env.action_space.shape[0]
        num_joints = env.action_space.shape[0]

        student_model = VisionStudent(
            num_actions=num_actions,
            num_joints=num_joints
        ).to(device)

        optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=config.get("learning_rate", 3e-4)
        )

        buffer = DAggerBuffer(
            max_samples=config.get("dagger_buffer_steps", 4000) * config.get("dagger_store_per_step", 32),
            img_shape=(3, 128, 128),
            num_joints=num_joints,
            num_actions=num_actions,
            device=device
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

        log.info("Starting DAgger Training...")
        for step in range(total_iterations):
            states = metasim_env.env.handler.get_states()

            # kamera: [N, H, W, C] -> [N, C, H, W]
            rgb_tensor = states.cameras["camera0"].rgb.to(device)
            rgb_permuted = rgb_tensor.permute(0, 3, 1, 2).contiguous().float()

            # očekáváme kameru 128x128, bez resize
            student_obs = rgb_permuted
            student_obs_uint8 = student_obs.to(torch.uint8)
            student_obs_net_input = student_obs / 255.0

            # jointy z observation; v tvém wrapperu jsou na začátku observation
            joint_obs_tensor = torch.as_tensor(
                expert_obs[:, :num_joints],
                device=device,
                dtype=torch.float32
            )

            with torch.no_grad():
                expert_actions, _ = expert_model.predict(expert_obs, deterministic=True)
                expert_actions_tensor = torch.as_tensor(expert_actions, device=device, dtype=torch.float32)

                student_model.eval()
                student_actions_tensor = student_model(student_obs_net_input, joint_obs_tensor)
                student_actions = student_actions_tensor.cpu().numpy()

            # směs expert/student
            if random.random() < beta:
                env_actions = expert_actions
            else:
                env_actions = student_actions

            # do bufferu uložit jen část env, ne všech 100
            buffer.add_batch(
                student_obs_uint8,
                joint_obs_tensor,
                expert_actions_tensor,
                store_count=store_per_step
            )

            expert_obs, rewards, dones, infos = env.step(env_actions)

            # trénovat méně často, ale více gradient kroky
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

                writer.add_scalar("DAgger/MSE_Loss", mean_loss, step)
                writer.add_scalar("DAgger/Beta_Mix_Ratio", beta, step)
                writer.add_scalar("DAgger/Env_Mean_Reward", float(np.mean(rewards)), step)
                writer.add_scalar("DAgger/Buffer_Size", buffer.size, step)

                log.info(
                    f"Step {step}/{total_iterations} | "
                    f"Beta: {beta:.4f} | "
                    f"Loss: {mean_loss:.6f} | "
                    f"Buffer: {buffer.size}"
                )

            if VIZUALIZATION and step % 5 == 0:
                img_vis = student_obs_uint8[0].permute(1, 2, 0).detach().cpu().numpy()
                img_vis_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                cv2.imshow("Student camera input", img_vis_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    log.info("Visualization interrupted by user (ESC).")
                    break

            if step > 0 and step % save_freq == 0:
                current_save_path = os.path.join(save_dir, f"student_model_step_{step}.pth")
                torch.save(student_model.state_dict(), current_save_path)
                log.info(f"Checkpoint saved to {current_save_path}")

            beta = max(0.0, beta * beta_decay)

        final_save_path = os.path.join(save_dir, "student_model_final.pth")
        torch.save(student_model.state_dict(), final_save_path)
        log.info(f"DAgger Training Finished! Final model saved to {final_save_path}")

        writer.close()
        env.close()

        if VIZUALIZATION:
            cv2.destroyAllWindows()

    elif config.get("train_or_eval") == "eval_dagger_video":
        from dagger.student_net import VisionStudent
        import cv2
        import numpy as np

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
                    obs[:, :num_joints],
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

    elif config.get("train_or_eval") == "train_grpo":
        from grpo.student_net_stochastic import VisionStudent
        from grpo.grpo_trainer import collect_parallel_episodes, build_grpo_batch, grpo_update
        from torch.utils.tensorboard import SummaryWriter
        import gc
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
                optimizer=optimizer,
                batch=batch,
                device=device,
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
                f"entropy={update_stats['entropy']:.6f}"
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
        import cv2
        import numpy as np

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
        student_model.eval()

        video_path = config.get("video_save_path", "./output/grpo_fpv_video.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        video_width, video_height = 128, 128
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (video_width, video_height))

        low_t = torch.as_tensor(env.action_space.low, device=device, dtype=torch.float32)
        high_t = torch.as_tensor(env.action_space.high, device=device, dtype=torch.float32)

        _ = env.reset()
        max_steps = config.get("eval_max_steps", 1000)

        log.info("Starting GRPO evaluation...")

        try:
            for step in range(max_steps):
                states = metasim_env.env.handler.get_states()
                robot_name = metasim_env.scenario.robots[0].name

                rgb_tensor_raw = states.cameras["camera0"].rgb
                frame_np = rgb_tensor_raw[0].detach().cpu().numpy().astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

                imgs = rgb_tensor_raw.to(device).permute(0, 3, 1, 2).contiguous().float() / 255.0
                joints = states.robots[robot_name].joint_pos.to(device).float()

                with torch.no_grad():
                    actions_t, _, _, _ = student_model.act(imgs, joints, deterministic=True)
                    actions_t = torch.max(torch.min(actions_t, high_t), low_t)

                _, rewards, dones, infos = env.step(actions_t.detach().cpu().numpy())

                if np.any(dones):
                    log.info(
                        f"Episode finished at step {step + 1}, "
                        f"success={infos[0].get('is_success', False)}"
                    )

                if step % 100 == 0:
                    log.info(
                        f"Eval step: {step}/{max_steps} | "
                        f"reward={float(np.mean(rewards)):.4f}"
                    )

        finally:
            video_writer.release()
            env.close()

        log.info(f"🎬 GRPO FPV Video saved successfully to: {video_path}")


if __name__ == "__main__":
    main()

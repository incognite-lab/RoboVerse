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
        config_name = "g1_ChairMan"
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
        eval_env = StableBaseline3VecEnv(metasim_env)

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
                    TensorboardMetricsCallback(log_dir=config.get("tensorboard_log", "./ppo_tensorboard/")),
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
                    TensorboardMetricsCallback(log_dir=config.get("tensorboard_log", "./ppo_tensorboard/")),
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
        from dagger.student_net import VisionStudent
        from dagger.dagger_trainer import DAggerBuffer, train_dagger_step
        import torch.nn.functional as F
        from torch.utils.tensorboard import SummaryWriter # <--- PŘIDÁNO PRO TENSORBOARD

        metasim_env = MetaSimVecEnv(scenario, task_name=config.get("task"), num_envs=config.get("num_envs", 1), sim=config.get("sim"))
        env = StableBaseline3VecEnv(metasim_env)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- TENSORBOARD SETUP ---
        tb_log_dir = config.get("tensorboard_log", "./dagger_tensorboard/")
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)

        # --- MODEL SAVING SETUP ---
        save_dir = config.get("model_save_path", "./output/dagger_models/")
        os.makedirs(save_dir, exist_ok=True)
        save_freq = config.get("model_save_freq", 5000)

        # 1. Načtení EXPERTA (PPO) - Zmrazený, už se neučí
        log.info(f"Loading Expert model from {config.get('load_model_path')}")
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.numeric'] = np.core.numeric
        expert_model = PPO.load(config.get("load_model_path"), env=env, device=device)

        # 2. Inicializace STUDENTA a Bufferu
        num_actions = env.action_space.shape[0]
        student_model = VisionStudent(num_actions=num_actions).to(device)
        optimizer = torch.optim.Adam(student_model.parameters(), lr=3e-4)

        buffer = DAggerBuffer(max_size=15000, device=device)

        total_iterations = config.get("total_timesteps", 100_000)
        beta = 1.0
        beta_decay = 0.999

        expert_obs = env.reset()

        log.info("Starting DAgger Training...")
        for step in range(total_iterations):
            states = metasim_env.env.handler.get_states()

            # Extrakce a zmenšení obrazu pro trénink
            rgb_tensor = states.cameras["camera0"].rgb.to(device)
            rgb_permuted = rgb_tensor.permute(0, 3, 1, 2).float()
            student_obs_small = F.interpolate(rgb_permuted, size=(72, 128), mode='bilinear', align_corners=False)

            student_obs_uint8 = student_obs_small.to(dtype=torch.uint8)
            student_obs_net_input = student_obs_small / 255.0

            with torch.no_grad():
                expert_actions, _ = expert_model.predict(expert_obs, deterministic=True)
                expert_actions_tensor = torch.tensor(expert_actions, device=device)

                student_model.eval()
                student_actions_tensor = student_model(student_obs_net_input)
                student_actions = student_actions_tensor.cpu().numpy()

            if random.random() < beta:
                env_actions = expert_actions
            else:
                env_actions = student_actions

            buffer.add_batch(student_obs_uint8, expert_actions_tensor)
            expert_obs, rewards, dones, infos = env.step(env_actions)

            # Zápis do Tensorboardu a učení
            if step > 0 and step % 5 == 0:
                loss = train_dagger_step(student_model, optimizer, buffer, batch_size=128)

                # --- LOGOVÁNÍ TENSORBOARD ---
                writer.add_scalar("DAgger/MSE_Loss", loss, step)
                writer.add_scalar("DAgger/Beta_Mix_Ratio", beta, step)
                writer.add_scalar("DAgger/Env_Mean_Reward", rewards.mean().item(), step)

                log.info(f"Step {step}/{total_iterations} | Beta: {beta:.2f} | Loss: {loss:.5f}")

            # --- PRŮBĚŽNÉ UKLÁDÁNÍ MODELU ---
            if step > 0 and step % save_freq == 0:
                current_save_path = os.path.join(save_dir, f"student_model_step_{step}.pth")
                torch.save(student_model.state_dict(), current_save_path)
                log.info(f"Checkpoint saved to {current_save_path}")

            beta = max(0.0, beta * beta_decay)

        # --- FINÁLNÍ ULOŽENÍ A UKONČENÍ ---
        final_save_path = os.path.join(save_dir, "student_model_final.pth")
        torch.save(student_model.state_dict(), final_save_path)
        log.info(f"DAgger Training Finished! Final model saved to {final_save_path}")

        writer.close()
        env.close()
    elif config.get("train_or_eval") == "eval_dagger_video":
        from dagger.student_net import VisionStudent
        import torch.nn.functional as F
        import cv2  # <--- PRO TVORBU VIDEA Z KAMERY

        metasim_env = MetaSimVecEnv(scenario, task_name=config.get("task"), num_envs=config.get("num_envs", 1), sim=config.get("sim"))
        env = StableBaseline3VecEnv(metasim_env)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Načtení STUDENTA a jeho natrénovaných vah
        num_actions = env.action_space.shape[0]
        student_model = VisionStudent(num_actions=num_actions).to(device)

        model_path = config.get("load_model_path")
        log.info(f"Loading Student model from {model_path}")
        student_model.load_state_dict(torch.load(model_path, map_location=device))
        student_model.eval() # Přepnutí do eval módu (vypne dropout atd.)

        # 2. Nastavení tvorby videa
        video_path = config.get("video_save_path", "./output/dagger_fpv_video.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        # Očekáváme plné rozlišení z vaší kamery
        video_width, video_height = 640, 360
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Kodek pro .mp4
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (video_width, video_height))

        obs = env.reset()
        log.info("Starting DAgger Evaluation...")

        # 3. Evaluační smyčka
        for step in range(config.get("eval_max_steps", 1000)):
            states = metasim_env.env.handler.get_states()

            # Získání raw obrazu [B, H, W, C]
            rgb_tensor_raw = states.cameras["camera0"].rgb

            # --- Zápis do Videa (Použijeme prostředí s indexem 0) ---
            # Převod na numpy a uint8 (formát pro obrázky)
            frame_np = rgb_tensor_raw[0].cpu().numpy().astype(np.uint8)
            # OpenCV používá barevný prostor BGR, Genesis RGB. Musíme to přehodit.
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            # --- Zpracování obrazu pro neuronovou síť ---
            rgb_tensor_gpu = rgb_tensor_raw.to(device)
            rgb_permuted = rgb_tensor_gpu.permute(0, 3, 1, 2).float()
            # Student potřebuje zmenšený formát 128x72
            student_obs_small = F.interpolate(rgb_permuted, size=(72, 128), mode='bilinear', align_corners=False)
            student_obs_net_input = student_obs_small / 255.0

            # 4. Predikce a krok v prostředí
            with torch.no_grad():
                student_actions_tensor = student_model(student_obs_net_input)
                actions = student_actions_tensor.cpu().numpy()

            obs, rewards, dones, infos = env.step(actions)

            if step % 100 == 0:
                log.info(f"Eval step: {step}/{config.get('eval_max_steps', 1000)}")

        # 5. Úklid a uložení videa
        video_writer.release()
        log.info(f"🎬 FPV Video saved successfully to: {video_path}")
        env.close()
        quit()
if __name__ == "__main__":
    main()

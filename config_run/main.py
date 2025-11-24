import sys
import os
import yaml
import torch

from loguru import logger as log

from metasim.cfg.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg, GyroSensorCfg, CommandCfg
from metasim.constants import PhysicStateType, SimType
from metasim.wrapper.gym_vec_env import MetaSimVecEnv
from stable_baselines3 import PPO
from callbacks import TensorboardMetricsCallback, SaveModelCallback,RewardPlotCallback
import numpy as np


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
            camera.pos = tuple(map(float, camera.pos.strip("()").split(",")))
            camera.look_at = tuple(map(float, camera.look_at.strip("()").split(",")))
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
        config_name = "g1_walk_new_train"
        #config_name = "g1_reach_pos_ori_train"
        #config_name = "g1_stand_train"
        #config_name = "g1_walk_new_eval"
        #config_name = "g1_door_IK"
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
        force = config.get("force", False),
        force_x_min = config.get("force_x_min", 0.0),
        force_x_max = config.get("force_x_max", 0.0),
        force_y_min = config.get("force_y_min", 0.0),
        force_y_max = config.get("force_y_max", 0.0),

        )
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


    metasim_env = MetaSimVecEnv(scenario, task_name=config.get("task"), num_envs=config.get("num_envs", 1), sim=config.get("sim"))


    env = StableBaseline3VecEnv(metasim_env)
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
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=lr_schedule(float(config.get("learning_rate", 3e-4)), 1e-5),
           # learning_rate=config.get("learning_rate", 3e-4),
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
            device="cpu"#cuda" if torch.cuda.is_available() else "cpu",
        )
        model.learn(total_timesteps=config.get("total_timesteps", 1_000_000),
                    callback=[
                    SaveModelCallback(save_path=config.get("model_save_path"), save_freq=config.get("model_save_freq", 1_000_000),task_name=config.get("task")),
                    TensorboardMetricsCallback(log_dir=config.get("tensorboard_log", "./ppo_tensorboard/"))
                    ],
                    progress_bar=True,)

        #Save the model
        task_name = scenario.task.__class__.__name__[:-3]
        model.save(config.get("model_save_path", f"ppo_{task_name}_{config.get('sim', 'unknown_sim')}"))
        log.info("Model saved. Ending the training and closing the environment.")
        env.close()
        quit()
    elif config.get("train_or_eval") == "eval":

        # sys.modules['numpy._core'] = np.core
        # sys.modules['numpy._core.numeric'] = np.core.numeric

        # load the model
        log.info(f"Loading model from {config.get('load_model_path')}")
        model = PPO.load(config.get("load_model_path"), env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        # --- Nastavení videa ---
        os.makedirs(os.path.dirname(config.get("video_save_path")), exist_ok=True)
        observation = ObsSaver(video_path=config.get("video_save_path"))
        slow = config.get("video_slowdown", 3)
        # inference
        obs = env.reset()
        for step in range(config.get("eval_episodes", 1000)):
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
        # load the model
        log.info(f"Loading model from {config.get('load_model_path')}")
        model = PPO.load(config.get("load_model_path"), env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        model.set_env(env)
        model.learn(total_timesteps=config.get("total_timesteps", 1_000_000),
                    callback=[
                    SaveModelCallback(save_path=config.get("model_save_path"), save_freq=config.get("model_save_freq", 1_000_000),task_name=config.get("task")),
                    TensorboardMetricsCallback(log_dir=config.get("tensorboard_log", "./ppo_tensorboard/"))
                    ],
                    progress_bar=True,)

        #Save the model
        task_name = scenario.task.__class__.__name__[:-3]
        model.save(config.get("model_save_path", f"ppo_{task_name}_{config.get('sim', 'unknown_sim')}"))
        log.info("Model saved. Ending the training and closing the environment.")
        env.close()
        quit()


if __name__ == "__main__":
    main()

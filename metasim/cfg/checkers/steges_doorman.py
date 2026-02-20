import pickle
import os
from time import time
from metasim.utils.humanoid_robot_util import (
    neck_height_tensor,
    neck_height,
    right_palm_position,
    right_palm_orientation,
    door_angle_tensor,
)
from metasim.cfg.objects import BaseObjCfg
from metasim.utils.configclass import configclass
from metasim.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_from_matrix
from metasim.utils.tensor_util import tensor_to_str

try:
    from metasim.sim import BaseSimHandler
except:
    pass
from metasim.types import EnvState
import torch
from pathlib import Path



from metasim.utils.humanoid_robot_util import robot_position_tensor
from metasim.utils.humanoid_robot_util import right_palm_orientation
from metasim.utils.humanoid_robot_util import right_palm_position

import random



HEIGHT_THRESHOLD = 0.4 #ALL STAGES

DISTANCE_TO_DOOR_X_THRESHOLD = -0.5 #STAGE 0
DISTANCE_TO_DOOR_Y_THRESHOLD = 0.2 #STAGE 0

DISTANCE_TO_DOOR_HANDLE_THRESHOLD = 0.002 #STAGE 1
ORIENTATION_DISTANCE_HANDLE_THRESHOLD = 0.1 #STAGE 1

FINGER_CLOSED_THRESHOLD = 0.1 #STAGE 2
DISTANCE_TO_HANDLE_STAGE2_THRESHOLD = 0.4 #STAGE 2

HANDLE_UNLOCK_ANGLE_THRESHOLD = 0.4 #STAGE 3

DOOR_OPEN_ANGLE_THRESHOLD = 1.57 #STAGE 4

PASS_THROUGH_DOOR_X_THRESHOLD = 1.0 #STAGE 5

#-----------------------------------------------
#Define params for snapshots
#-----------------------------------------------
SNAPSHOT_DIR = Path("config_run/snapshots_door/")
MAX_SNAPSHOTS = 5
NAME_OF_THE_OBJECT = "door"





def distance_from_fingers_to_catch_handle(states:list[EnvState], handler:BaseSimHandler, env: int):
    """
    check if fingers are close to handle
    """
    # 1. Definice kloubů, které chceme kontrolovat (ukazováček a prostředníček jsou pro úchop nejdůležitější)
    # Pro pravou ruku je zavřená pozice (flexe) na HORNÍM limitu (pozitivní hodnoty).
    finger_joints_limits = {
        "right_hand_index_0_joint": 1.57079632,   # Max hodnota = zavřeno
        "right_hand_index_1_joint": 1.74532925,   # Max hodnota = zavřeno
        "right_hand_middle_0_joint": 1.57079632,  # Max hodnota = zavřeno
        "right_hand_middle_1_joint": 1.74532925,  # Max hodnota = zavřeno
        # Palec můžeme přidat volitelně, často stačí prsty pro "obemknutí"
        "right_hand_thumb_2_joint": 0.0 # U palce je to složitější, 0 bývá často 'přitisknuto' k dlani v určitých modelech,
                                        # ale podle vašich limitů (-1.74, 0) je 0 spíše narovnáno.
                                        # Pro jednoduchost a robustnost se často sledují jen prsty.
    }
    robot = states.robots[handler.robot.name]
    joint_names = robot.joint_names.tolist()
    joint_pos = robot.joint_pos[env]

    distances = []
    for joint_name, closed_limit in finger_joints_limits.items():
        idx = joint_names.index(joint_name)
        current_pos = joint_pos[idx]
        distances.append(torch.abs(current_pos - closed_limit))


    avg_distance = torch.stack(distances).mean()

    return avg_distance



def common_doorman_checker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    COMMON CHECKER FOR DOORMAN TASK
    Check if the robot has fallen.
    """
    is_fallen = neck_height_tensor(states, handler.robot.name)[env] < HEIGHT_THRESHOLD
    return is_fallen

def stege0_chacker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    WALK TO DOOR
    Check if the robot has fallen (stage 0).

    Check if the robot is close enough to the door to consider the stage successful.

    """
    terminated = common_doorman_checker(states, handler, env)
    robot_pos = robot_position_tensor(states, handler.robot.name)[env]
    #base_link_idx = states.objects[NAME_OF_THE_OBJECT].body_names.index("base_link")
    door_pos = states.objects[NAME_OF_THE_OBJECT].body_state[env,0,:3]
    distance_x = torch.abs(robot_pos[0] - door_pos[0])
    distance_y = torch.abs(robot_pos[1] - door_pos[1])

    if not terminated and distance_x >= DISTANCE_TO_DOOR_X_THRESHOLD and distance_y < DISTANCE_TO_DOOR_Y_THRESHOLD:
        terminated = True
        success = True
    else:
        success = False

    #implemenotva success condition
    return terminated,success



def stege1_chacker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    PREGRASP(REACH HANDLE)
    Check if the robot has fallen (stage 1).
    """
    terminated = common_doorman_checker(states, handler, env)
    palm_pos = right_palm_position(states, handler.robot.name, ee_name="endeffector")[env]
    palm_ori = right_palm_orientation(states, handler.robot.name, ee_name="endeffector")[env]
    handle_idx = states.objects["door"].body_names.index("door_handle")
    handle_pos = states.objects["door"].body_state[env, handle_idx, :3]
    handle_ori = states.objects["door"].body_state[env, handle_idx, 3:7]
    distance = torch.norm(palm_pos - handle_pos)

    # Compute quaternion distance
    dot_product = torch.abs(torch.sum(handle_ori * palm_ori))
    ori_distance = 1.0 - dot_product
    if not terminated and distance < DISTANCE_TO_DOOR_HANDLE_THRESHOLD and ori_distance < ORIENTATION_DISTANCE_HANDLE_THRESHOLD:
        terminated = True
        success = True
    else:
        success = False


    return terminated, success
def stege2_chacker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    GRASP HANDLE
    Check if the robot has fallen (stage 2).
    """
    terminated = common_doorman_checker(states, handler, env)
    palm_pos = right_palm_position(states, handler.robot.name, ee_name="endeffector")[env]
    handle_idx = states.objects["door"].body_names.index("door_handle")
    handle_pos = states.objects["door"].body_state[env, handle_idx, :3]
    distance = torch.norm(palm_pos - handle_pos)


    #check position of fingers
    finger_distance = distance_from_fingers_to_catch_handle(states,handler,env)
    if  not terminated \
        and finger_distance < FINGER_CLOSED_THRESHOLD \
        and distance < DISTANCE_TO_HANDLE_STAGE2_THRESHOLD:

        terminated = True
        success = True
    elif distance >= DISTANCE_TO_HANDLE_STAGE2_THRESHOLD:

        terminated = True
        success = False
    else:
        success = False

    return terminated, success
def stege3_chacker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    PULL HANDLE DOWN
    Check if the robot has fallen (stage 3).
    """
    terminated = common_doorman_checker(states, handler, env)
    #handle_idx = states.objects["door"].body_names.index("door_handle")
    handle_angle = states.objects["door"].joint_pos[env, 0]
    if not terminated and torch.abs(handle_angle) > HANDLE_UNLOCK_ANGLE_THRESHOLD:
        terminated = True
        success = True
    else:
        success = False

    return terminated, success
def stege4_chacker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    SWING (OPEN DOOR)
    Check if the robot has fallen (stage 4).
    """
    terminated = common_doorman_checker(states, handler, env)
    #door_idx = states.objects["door"].body_names.index("door_hinge")
    door_angle = states.objects["door"].joint_pos[env, 1]
    if not terminated and torch.abs(door_angle) >= DOOR_OPEN_ANGLE_THRESHOLD:
        terminated = True
        success = True
    else:
        success = False
    return terminated, success
def stege5_chacker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    PASS THROUGH DOOR
    Check if the robot has fallen (stage 5).
    """
    terminated = common_doorman_checker(states, handler, env)
    robot_pos = robot_position_tensor(states, handler.robot.name)[env]
    door_pos = states.objects["door"].body_state[env,0,:3]
    distance_x = (robot_pos[0] - door_pos[0])
    if not terminated and distance_x > PASS_THROUGH_DOOR_X_THRESHOLD:
        terminated = True
        success = True
    else:
        success = False
    return terminated, success



#-----------------------------------------------
#Definitions of reset for each stage
#-----------------------------------------------
def reset_doorman(handler: BaseSimHandler, env_ids: list[int] | None = None):
    """
    Reset s logikou "Curriculum Learning":
    1. Zkontroluje, které stage (1-5) již mají uložené snapshoty.
    2. Náhodně vybere stage mezi 0 a maximální dostupnou stage.
    3. Inicializuje robota (buď procedurálně pro stage 0, nebo načtením snapshotu).
    """
    states = None

    if env_ids is None:
        env_ids = list(range(handler.num_envs))

    # Inicializace pole pro trackování stage v reward function, pokud neexistuje
    current_stages_tensor = handler.task.reward_functions[0].actual_stage
    if current_stages_tensor is None:
        current_stages_tensor = torch.tensor([0] * handler.num_envs, device=handler.device)
        current_stages_completed = torch.tensor([0] * handler.num_envs, device=handler.device)
        for i in range(len(handler.task.reward_functions)):
            handler.task.reward_functions[i].actual_stage = current_stages_tensor
            handler.task.reward_functions[i].completed_stages = current_stages_completed

    # --- KROK 1: Zjištění maximální dostupné stage ---
    # Projdeme složky a zjistíme, kam až jsme se dostali.
    # Stage 0 je dostupná vždy (procedurální).
    max_available_stage = 0

    # Předpokládáme max stage 5 dle definice
    for i in range(1, 6):
        stage_dir = SNAPSHOT_DIR / f"stage_{i}"
        # Stage považujeme za dostupnou, pokud složka existuje a obsahuje alespoň jeden .pkl soubor
        if stage_dir.exists() and any(stage_dir.glob("*.pkl")):
            max_available_stage = i
        else:
            # Pokud chybí např. stage 2, nemá smysl hledat stage 3 (curriculum je postupné)
            break

    print(f"DEBUG: Max available stage found: {max_available_stage}")

    # --- KROK 2: Resetování jednotlivých prostředí ---
    for env in env_ids:
        # Náhodná volba stage: 0 až max_available_stage
        new_stage = random.randint(0, max_available_stage)

        # Aktualizace informace o stage v reward funkcích
        for i in range(len(handler.task.reward_functions)):
            handler.task.reward_functions[i].actual_stage[env] = new_stage
            handler.task.reward_functions[i].completed_stages[env] = 0

        print(f"Resetting env {env} to stage {new_stage} (Max avail: {max_available_stage})")

        state = None

        # Pokus o načtení stavu pro stage > 0
        if new_stage > 0:
            state = load_snapshot_doorman(stage=new_stage)

        # --- KROK 3: Fallback a Stage 0 ---
        # Pokud je stage 0, NEBO pokud načtení vyšší stage selhalo (state je None),
        # provedeme inicializaci na stage 0.
        if state is None:
            if new_stage > 0:
                print(f"Warning: Failed to load snapshot for stage {new_stage}, reverting env {env} to Stage 0.")
                # Musíme opravit i záznam v reward funkci zpět na 0
                for i in range(len(handler.task.reward_functions)):
                    handler.task.reward_functions[i].actual_stage[env] = 0
                    handler.task.reward_functions[i].completed_stages[env] = 0
            state = stage0_init(handler.robot.name)

        # Sestavení listu states pro handler (zachování původní logiky pole)
        if states is None:
            states = [state] * handler.num_envs
        else:
            states[env] = state

    handler.set_states(states=states, env_ids=env_ids)


def save_snapshot_doorman(handler: BaseSimHandler, env_id: int, stage: int) -> None:
    """
    Uloží aktuální stav prostředí (robota a objektů) do souboru pro danou stage.
    Kontroluje limit 100 snapshotů - pokud je překročen, smaže nejstarší.
    """
    # 1. Příprava adresáře pro danou stage
    stage_dir = SNAPSHOT_DIR / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    # 2. Získání aktuálního stavu z handleru
    full_states = handler.get_states()
    snapshot_data = {
        "robots": {},
        "objects": {}
    }
    # Extrahuje data robota (převedeme na CPU a Numpy pro uložení)
    robot_name = handler.robot.name
    robot_states = full_states.robots[robot_name]
    joint_names = robot_states.joint_names.tolist()
    joint_pos = robot_states.joint_pos[env_id].detach().cpu().numpy()
    joint_vel = robot_states.joint_vel[env_id].detach().cpu().numpy()

    robot_pos = robot_states.root_state[env_id,:3].detach()
    robot_rot = robot_states.root_state[env_id,3:7].detach()
    dof_pos = {}
    dof_vel = {}
    for i, name in enumerate(joint_names):
        dof_pos[name] = joint_pos[i]
        dof_vel[name] = joint_vel[i]



    snapshot_data["robots"][robot_name] = {
        "pos": robot_pos,
        "rot": robot_rot,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
    }


    # Extrahuje data objektů (např. dveře)
    for obj_name, obj_state in full_states.objects.items():
        joint_names = obj_state.joint_names.tolist()
        joint_pos = obj_state.joint_pos[env_id].detach().cpu().numpy()
        joint_vel = obj_state.joint_vel[env_id].detach().cpu().numpy()
        dof_pos = {}
        dof_vel = {}
        for i, name in enumerate(joint_names):
            dof_pos[name] = joint_pos[i]
            dof_vel[name] = joint_vel[i]


        snapshot_data["objects"][obj_name] = {
            "pos": obj_state.root_state[env_id,:3].detach(),
            "rot": obj_state.root_state[env_id,3:7].detach(),
            "dof_vel": dof_vel,
            "dof_pos": dof_pos,
        }
    # 3. Kontrola limitu snapshotů (Mazání nejstaršího)
    # Získáme seznam všech .pkl souborů v adresáři
    list_of_files = sorted(stage_dir.glob("*.pkl"), key=os.path.getctime)

    while len(list_of_files) >= MAX_SNAPSHOTS:
        oldest_file = list_of_files.pop(0) # První je nejstarší
        try:
            os.remove(oldest_file)
        except OSError as e:
            print(f"Error deleting old snapshot: {e}")

    # 4. Uložení nového snapshotu
    # Název souboru obsahuje timestamp pro unikátnost
    timestamp = int(time() * 1000)
    filename = stage_dir / f"snapshot_{timestamp}_{env_id}.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(snapshot_data, f)



def load_snapshot_doorman(stage: int) -> dict | None:
    """
    Načte náhodný snapshot pro danou stage.
    Vrací slovník se strukturou { "robots": {...}, "objects": {...} },
    který je kompatibilní s handler.set_states().
    """
    stage_dir = SNAPSHOT_DIR / f"stage_{stage}"

    # 1. Kontrola existence adresáře
    if not stage_dir.exists():
        return None

    # 2. Získání seznamu všech snapshotů (.pkl soubory)
    # glob vrací iterátor, převedeme na list
    list_of_files = list(stage_dir.glob("*.pkl"))

    if not list_of_files:
        return None # Adresář existuje, ale je prázdný

    # 3. Náhodný výběr jednoho souboru (Staged Reset logika)
    random_file = random.choice(list_of_files)

    # 4. Načtení dat
    try:
        with open(random_file, 'rb') as f:
            snapshot_data = pickle.load(f)

        # Data jsou již uložena jako {"robots": ..., "objects": ...} a hodnoty jsou numpy array/dict,
        # což je přesně to, co handler.set_states obvykle zpracovává.
        return snapshot_data

    except Exception as e:
        print(f"Chyba při načítání snapshotu {random_file}: {e}")
        return None


def stage0_init(robot_name: str):
    state = {
        "robots": {
            robot_name: {
                "pos" : torch.tensor([-1.0,0.0,0.8]),
                "rot" : torch.tensor([1.0,0.0,0.0,0.0]),
                "dof_pos": {
                    "left_hip_pitch_joint": 0.0,
                    "left_hip_roll_joint": 0.0,
                    "left_hip_yaw_joint": 0.0,
                    "left_knee_joint": 0.0,
                    "left_ankle_pitch_joint": 0.0,
                    "left_ankle_roll_joint": 0.0,
                    "right_hip_pitch_joint": 0.0,
                    "right_hip_roll_joint": 0.0,
                    "right_hip_yaw_joint": 0.0,
                    "right_knee_joint": 0.0,
                    "right_ankle_pitch_joint": 0.0,
                    "right_ankle_roll_joint": 0.0,
                    "waist_yaw_joint": 0.0,
                    "waist_roll_joint": 0.0,
                    "waist_pitch_joint": 0.0,
                    "left_shoulder_pitch_joint": 0.0,
                    "left_shoulder_roll_joint": 0.0,
                    "left_shoulder_yaw_joint": 0.0,
                    "left_elbow_joint": 1.0,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    "left_wrist_yaw_joint": 0.0,
                    "right_shoulder_pitch_joint": 0.0,
                    "right_shoulder_roll_joint": 0.0,
                    "right_shoulder_yaw_joint": 0.0,
                    "right_elbow_joint": 1.0,
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
            "door": {
                "pos": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "dof_pos":{'door_hinge': 0.0,
                           'door_handle_joint' :0.0
                           }
            },
        },
    }

    return state

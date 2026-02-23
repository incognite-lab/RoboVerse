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

DISTANCE_TO_CHAIR_X_THRESHOLD = 0.7 #STAGE 0
DISTANCE_TO_CHAIR_Y_THRESHOLD = 0.2 #STAGE 0

DISTANCE_TO_CHAIR_HANDLE_THRESHOLD = 0.03 #STAGE 1
ORIENTATION_DISTANCE_HANDLE_THRESHOLD = 0.03 #STAGE 1

GRASP_DRIFT_THRESHOLD = 0.05 #STAGE 2
GRASP_FORCE_THRESHOLD = 2.0 #STAGE 2
FINGER_TIPS = ["thumb_2", "index_1", "middle_1"] #STAGE 2

HANDLE_UNLOCK_ANGLE_THRESHOLD = 0.4 #STAGE 3

DOOR_OPEN_ANGLE_THRESHOLD = 1.57 #STAGE 4

PASS_THROUGH_DOOR_X_THRESHOLD = 1.0 #STAGE 5

POS_THRESHOLD = 0.3 #stage 0-2
ORI_DOT_PRODUCT_THRESHOLD = 0.9 #stage 0-2

SNAPSHOT_DIR = Path("config_run/snapshots_chair/")
MAX_SNAPSHOTS = 1000



def check_finger_contacts_success(handler: BaseSimHandler, env_id: int, target_obj_name="chair") -> bool:
    """
    Pomocná funkce: Zkontroluje, zda VŠECHNY špičky prstů (levé i pravé ruky)
    mají kontakt se židlí silnější než GRASP_FORCE_THRESHOLD.
    """
    try:
        contacts = handler.get_states().robots[handler.robot.name].contact

    except Exception as e:
        # print(f"Contact check failed: {e}")
        return False

    # Množina detekovaných špiček, které mají dostatečný kontakt
    detected_tips_left = set()
    detected_tips_right = set()

    # Očekávaný počet špiček na jednu ruku (z FINGER_TIPS)
    required_tips_count = len(FINGER_TIPS)

    for c in contacts:
        # 1. Filtr na Env ID (pokud get_contact vrací všechna envs, musíme filtrovat)
        if c.get('env_id') is not None and c['env_id'] != env_id:
            continue

        # 2. Kontrola kolize se židlí
        if c['body_a'] != target_obj_name and c['body_b'] != target_obj_name:
            continue

        # 3. Kontrola síly
        force = c.get('force', 0.0)
        # Pokud je síla vektor, uděláme normu, pokud skalár, použijeme přímo
        if isinstance(force, (list, tuple, torch.Tensor)):
            force = torch.norm(torch.tensor(force))

        if force < GRASP_FORCE_THRESHOLD:
            continue

        # 4. Identifikace ruky a prstu
        # Zjistíme, který link patří robotovi
        robot_link = c['link_a'] if c['body_b'] == target_obj_name else c['link_b']

        # Kontrola, zda jde o špičku
        is_tip = any(tip in robot_link for tip in FINGER_TIPS)
        if not is_tip:
            continue

        # Rozlišení levé a pravé ruky
        if "left" in robot_link:
            # Uložíme název špičky (např. "thumb_2")
            for tip in FINGER_TIPS:
                if tip in robot_link:
                    detected_tips_left.add(tip)
        elif "right" in robot_link:
            for tip in FINGER_TIPS:
                if tip in robot_link:
                    detected_tips_right.add(tip)

    # 5. Vyhodnocení
    # Success je pouze tehdy, když obě ruce mají kontakt na všech definovaných špičkách
    success_left = len(detected_tips_left) >= required_tips_count
    success_right = len(detected_tips_right) >= required_tips_count

    return success_left and success_right
def check_movement_chair(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:

    idx_base_chair = states.objects["chair"].body_names.index("base_link")
    chair_pos = states.objects["chair"].body_state[env, idx_base_chair, :3]
    chair_ori = states.objects["chair"].body_state[env, idx_base_chair, 3:7]
    initial_chair_ori = torch.tensor([7.0739e-01, 8.4260e-08, 0.0000e+00, 7.0683e-01], device=chair_ori.device)
    initial_chair_pos = torch.tensor([0.75, 0.0, 0.1], device=chair_pos.device) # Assuming chair starts at origin
    pos_diff = torch.norm(chair_pos - initial_chair_pos)
    dot_product = torch.abs(torch.sum(chair_ori * initial_chair_ori))
    if pos_diff > POS_THRESHOLD or dot_product < ORI_DOT_PRODUCT_THRESHOLD:
        return True
    return False
def common_chairman_checker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    COMMON CHECKER FOR CHAIRMAN TASK
    Check if the robot has fallen.
    """
    is_fallen = neck_height_tensor(states, handler.robot.name)[env] < HEIGHT_THRESHOLD
    return is_fallen

def stege0_chacker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    WALK TO CHAIR
    Check if the robot has fallen (stage 0).

    Check if the robot is close enough to the chair to consider the stage successful.

    """
    terminated = common_chairman_checker(states, handler, env) or check_movement_chair(states, handler, env)
    robot_pos = robot_position_tensor(states, handler.robot.name)[env]
    chair_base_link_idx = states.objects["chair"].body_names.index("base_link")
    chair_pos = states.objects["chair"].body_state[env,chair_base_link_idx,:3]
    distance_x = torch.abs(robot_pos[0] - chair_pos[0])
    distance_y = torch.abs(robot_pos[1] - chair_pos[1])

    if not terminated and distance_x <= DISTANCE_TO_CHAIR_X_THRESHOLD and distance_y < DISTANCE_TO_CHAIR_Y_THRESHOLD:
        terminated = True
        success = True
    else:
        success = False

    #implemenotva success condition
    return terminated,success



def stege1_chacker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    PREGRASP
    Check if the robot has fallen (stage 1).
    """
    terminated = common_chairman_checker(states, handler, env) or check_movement_chair(states, handler, env)
    right_ee_pos = right_palm_position(states, handler.robot.name, ee_name="endeffector")[env]
    right_ee_ori = right_palm_orientation(states, handler.robot.name, ee_name="endeffector")[env]
    left_ee_pos = right_palm_position(states, handler.robot.name, ee_name="left_endeffector")[env]
    left_ee_ori = right_palm_orientation(states, handler.robot.name, ee_name="left_endeffector")[env]
    right_handle_idx = states.objects["chair"].body_names.index("target_hand_right")
    right_handle_pos = states.objects["chair"].body_state[env, right_handle_idx, :3]
    right_handle_ori = states.objects["chair"].body_state[env, right_handle_idx, 3:7]
    left_handle_idx = states.objects["chair"].body_names.index("target_hand_left")
    left_handle_pos = states.objects["chair"].body_state[env, left_handle_idx, :3]
    left_handle_ori = states.objects["chair"].body_state[env, left_handle_idx, 3:7]
    left_distance = torch.norm(left_ee_pos - left_handle_pos)
    right_distance = torch.norm(right_ee_pos - right_handle_pos)
    # Compute quaternion distance
    left_dot_product = torch.abs(torch.sum(left_handle_ori * left_ee_ori))
    right_dot_product = torch.abs(torch.sum(right_handle_ori * right_ee_ori))
    right_ori_distance = 1.0 - right_dot_product
    left_ori_distance = 1.0 - left_dot_product
    if not terminated and left_distance < DISTANCE_TO_CHAIR_HANDLE_THRESHOLD and right_distance < DISTANCE_TO_CHAIR_HANDLE_THRESHOLD and left_ori_distance < ORIENTATION_DISTANCE_HANDLE_THRESHOLD and right_ori_distance < ORIENTATION_DISTANCE_HANDLE_THRESHOLD:
        terminated = True
        success = True
    else:
        success = False


    return terminated, success
def stege2_chacker(states: list[EnvState], handler: BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    STAGE 2: GRASP CHECKER
    1. Terminate if fallen.
    2. Terminate if hands drift too far from target handles.
    3. Success if all finger tips touch the chair with Force > 1N.
    """
    # --- 1. Safety Check (Falling) ---
    terminated = common_chairman_checker(states, handler, env) or check_movement_chair(states, handler, env)
    if terminated:
        return True, False

    # --- 2. Drift Check (Are hands still close to handles?) ---
    right_ee_pos = right_palm_position(states, handler.robot.name, ee_name="endeffector")[env]
    left_ee_pos = right_palm_position(states, handler.robot.name, ee_name="left_endeffector")[env]

    right_handle_idx = states.objects["chair"].body_names.index("target_hand_right")
    right_handle_pos = states.objects["chair"].body_state[env, right_handle_idx, :3]

    left_handle_idx = states.objects["chair"].body_names.index("target_hand_left")
    left_handle_pos = states.objects["chair"].body_state[env, left_handle_idx, :3]

    dist_right = torch.norm(right_ee_pos - right_handle_pos)
    dist_left = torch.norm(left_ee_pos - left_handle_pos)

    # Pokud se kterákoliv ruka vzdálí příliš -> FAIL
    if dist_right > GRASP_DRIFT_THRESHOLD or dist_left > GRASP_DRIFT_THRESHOLD:
        return True, False # Terminated=True, Success=False

    # --- 3. Contact Force Check (Success Condition) ---
    # Voláme pomocnou funkci pro kontrolu fyzikálních kontaktů
    has_firm_grasp = check_finger_contacts_success(handler, env, target_obj_name="chair")

    if has_firm_grasp:
        return True, True # Terminated=True (stage done), Success=True

    # Pokud nespadl, nevzdálil se, ale ještě nedrží pevně -> pokračujeme
    return False, False
def stege3_chacker(states:list[EnvState], handler:BaseSimHandler, env: int) -> torch.BoolTensor:
    """
    PULL HANDLE DOWN
    Check if the robot has fallen (stage 3).
    """
    terminated = common_chairman_checker(states, handler, env)
    #handle_idx = states.objects["chair"].body_names.index("handle")
    handle_angle = states.objects["chair"].joint_pos[env, 0]
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
    terminated = common_chairman_checker(states, handler, env)
    #handle_idx = states.objects["chair"].body_names.index("handle")
    chair_angle = states.objects["chair"].joint_pos[env, 1]
    if not terminated and torch.abs(chair_angle) >= CHAIR_OPEN_ANGLE_THRESHOLD:
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
    terminated = common_chairman_checker(states, handler, env)
    robot_pos = robot_position_tensor(states, handler.robot.name)[env]
    chair_pos = states.objects["chair"].body_state[env,0,:3]
    distance_x = (robot_pos[0] - chair_pos[0])
    if not terminated and distance_x > PASS_THROUGH_CHAIR_X_THRESHOLD:
        terminated = True
        success = True
    else:
        success = False
    return terminated, success



#-----------------------------------------------
#Definitions of reset for each stage
#-----------------------------------------------
def reset_chairman(handler: BaseSimHandler, env_ids: list[int] | None = None):
    """
    Reset s logikou "Curriculum Learning":
    1. Zkontroluje, které stage (1-5) již mají uložené snapshoty.
    2. Náhodně vybere stage mezi 0 a maximální dostupnou stage.
    3. Inicializuje robota (buď procedurálně pro stage 0, nebo načtením snapshotu).
    """
    states = [stage0_init(handler.robot.name)] * handler.num_envs

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
    for i in range(1, 3):#TODO 6
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
        #new_stage = 2 #TODO DEBUG!!! --- IGNORE ---
        # Aktualizace informace o stage v reward funkcích
        for i in range(len(handler.task.reward_functions)):
            handler.task.reward_functions[i].actual_stage[env] = new_stage
            handler.task.reward_functions[i].completed_stages[env] = 0

        print(f"Resetting env {env} to stage {new_stage} (Max avail: {max_available_stage})")

        state = None

        # Pokus o načtení stavu pro stage > 0
        if new_stage > 0:
            state = load_snapshot_chairman(stage=new_stage)

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


def save_snapshot_chairman(handler: BaseSimHandler, env_id: int, stage: int) -> None:
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



def load_snapshot_chairman(stage: int) -> dict | None:
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
    if robot_name == "g1_slider":
        state = {
            "robots": {
                "g1_slider": {
                    "dof_pos": {
                        "baseslide_joint": 0.0,
                        "baseslide_joint2": -1.5,
                        "baserot_joint": 0.0,
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.0,
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

                    }
            }
        }
    elif robot_name == "g1_with_hands":
        state = {
            "robots": {
                "g1_with_hands": {
                    "pos" : torch.tensor([-0.1,0.0,0.8]),
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
                "chair": {
                        "pos": torch.tensor([
                            0.0,
                            0.0,
                            0.5
                        ]),
                        "rot": torch.tensor([
                            1.0,
                            0.0,
                            0.0,
                            0.0
                        ]),
                        "dof_pos":{
                            "floor_slide_x": 0.5,
                            "floor_slide_y": 0.0,
                            #"seat_swivel" : 0.0,
                            "floor_rotate_z": 0.0


                        }

                    }
            },
        }

    return state

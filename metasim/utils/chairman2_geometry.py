"""Shared physical measurements for Chairman2 rewards and success checks.

Units: metres, seconds, radians. Palm normals are left -Y / right +Y in
G1's palm frames (also the signs used by its fixed end-effector offsets).
The palm mesh spans roughly +/-2 cm in Y; the old 7 cm EE offset is outside
that mesh. Use a point on the actual palm surface instead.
"""
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from metasim.utils.chair_navigation import chair_back_direction_xy, forward_direction_xy

NUM_STAGES = 5
TASK_VERSION = 'chairman2_five_stage_v2'
# URDF FK at nominal pelvis height 0.8 m admits bilateral surface targets
# with <1 mm error and palm/elbow angles below 10 degrees at this distance.
APPROACH_DISTANCE = 0.745
POSITION_TOLERANCE = 0.05
JOINT_TOLERANCE = 0.15
HEADING_TOLERANCE = math.radians(5)
PALM_ANGLE_TOLERANCE = math.radians(10)
ELBOW_ANGLE_TOLERANCE = math.radians(10)
STILL_SPEED = 0.08
STILL_YAW_SPEED = 0.10
ROBOT_DRIFT_TOLERANCE = 0.05
CHAIR_DRIFT_TOLERANCE = 0.03
PULL_DISTANCE = 1.0
PULL_TOLERANCE = 0.05
CONTACT_RADIUS = 0.06
CONTACT_FORCE_MIN = 0.5
CONTACT_FORCE_MAX = 60.0
LIFT_HEIGHT = 0.12
# Stage 1 uses task-space geometry rather than a prescribed arm pose.
HAND_FORWARD_REACH_MIN = 0.50
HAND_ABOVE_BACKREST_MIN = 0.0
# Stage 2 moves the end effectors from above the backrest toward the seat side.
HAND_BEHIND_BACKREST_MIN = 0.10
UPPER_BODY_COM_DEADZONE = 0.05
UPPER_BODY_COM_SCALE = 0.10
G1_WITHOUT_HANDS_URDF = (
    Path(__file__).resolve().parents[2]
    / "roboverse_data/robots/g1/urdf/g1_mygym_without_hand.urdf"
)
G1_WITH_HANDS_URDF = (
    Path(__file__).resolve().parents[2]
    / "roboverse_data/robots/g1/urdf/g1_mygym.urdf"
)
STAGE0_JOINT_TARGETS = {
    'left_shoulder_pitch_joint': 0.87, 'left_shoulder_roll_joint': 0.45,
    'left_shoulder_yaw_joint': 0.25, 'left_elbow_joint': -0.89, 'left_wrist_roll_joint': 1.12,
    'right_shoulder_pitch_joint': 0.87, 'right_shoulder_roll_joint': -0.45,
    'right_shoulder_yaw_joint': -0.25, 'right_elbow_joint': -0.89, 'right_wrist_roll_joint': -1.12,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
}


def rotate(q, v):
    q = torch.nn.functional.normalize(q, dim=-1)
    v = torch.broadcast_to(v, q.shape[:-1] + (3,))
    t = 2 * torch.linalg.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.linalg.cross(q[..., 1:], t, dim=-1)


@lru_cache(maxsize=None)
def upper_body_inertials(urdf_path: str = str(G1_WITHOUT_HANDS_URDF)):
    """Return mass and local COM offset for the complete torso subtree."""
    root = ET.parse(urdf_path).getroot()
    children = defaultdict(list)
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            children[parent.attrib["link"]].append(child.attrib["link"])

    selected = set()
    pending = ["torso_link"]
    while pending:
        link_name = pending.pop()
        if link_name in selected:
            continue
        selected.add(link_name)
        pending.extend(children.get(link_name, ()))

    inertials = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        if name not in selected:
            continue
        inertial = link.find("inertial")
        if inertial is None or inertial.find("mass") is None:
            continue
        mass = float(inertial.find("mass").attrib["value"])
        if mass <= 0:
            continue
        origin = inertial.find("origin")
        xyz = (0.0, 0.0, 0.0) if origin is None else tuple(
            float(value) for value in origin.attrib.get("xyz", "0 0 0").split()
        )
        if len(xyz) != 3:
            raise ValueError(f"Invalid inertial origin for URDF link {name}: {xyz}")
        inertials[name] = (mass, xyz)
    if "torso_link" not in inertials:
        raise ValueError(f"URDF {urdf_path} has no massive torso_link")
    return inertials


_UPPER_BODY_TENSOR_CACHE = {}


def upper_body_center_of_mass(robot, urdf_path: str = str(G1_WITHOUT_HANDS_URDF)):
    """Mass-weighted world COM from torso through head, arms, hands and fingers."""
    body_names = tuple(str(name) for name in robot.body_names)
    key = (urdf_path, body_names, str(robot.body_state.device), robot.body_state.dtype)
    cached = _UPPER_BODY_TENSOR_CACHE.get(key)
    if cached is None:
        inertials = upper_body_inertials(urdf_path)
        indices, masses, offsets = [], [], []
        for index, name in enumerate(body_names):
            if name in inertials:
                mass, offset = inertials[name]
                indices.append(index)
                masses.append(mass)
                offsets.append(offset)
        if not indices:
            raise ValueError("No upper-body inertial links are present in the robot state")
        cached = (
            torch.as_tensor(indices, dtype=torch.long, device=robot.body_state.device),
            robot.body_state.new_tensor(masses),
            robot.body_state.new_tensor(offsets),
        )
        _UPPER_BODY_TENSOR_CACHE[key] = cached
    indices, masses, offsets = cached
    links = robot.body_state.index_select(1, indices)
    link_com = links[..., :3] + rotate(links[..., 3:7], offsets[None])
    return (link_com * masses[None, :, None]).sum(1) / masses.sum()


def angle_between(a, b):
    return torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(a, b, dim=-1), dim=-1),
                       (a * b).sum(-1))


def planar_angle(a, b):
    return torch.atan2((a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]).abs(), (a*b).sum(-1))


def hand_task_space_metrics(torso, end_effectors, targets, chair_dir):
    """Measure the stage-1/2 hand goals in robot/chair-relative coordinates."""
    forward = forward_direction_xy(torso[:, 3:7])
    ee_from_torso_xy = end_effectors[..., :2] - torso[:, None, :2]
    hand_forward_reach = (ee_from_torso_xy * forward[:, None]).sum(-1)
    right_target_height = targets[:, 1, 2]
    hand_height_above_backrest = end_effectors[..., 2] - right_target_height[:, None]
    # chair_dir points from the seat toward the robot side of the backrest.
    # A positive value therefore means that the end effector crossed the
    # backrest plane toward the seat.
    hand_behind_backrest = (
        (targets[..., :2] - end_effectors[..., :2]) * chair_dir[:, None]
    ).sum(-1)
    hand_below_target = right_target_height[:, None] - end_effectors[..., 2]
    return {
        'hand_forward_reach': hand_forward_reach,
        'hand_height_above_backrest': hand_height_above_backrest,
        'hand_behind_backrest': hand_behind_backrest,
        'hand_below_target': hand_below_target,
    }


def hand_contacts(states, robot_name, points, targets):
    """Per-hand physical contact, backrest contact and force magnitude.

    Prefer solver contact positions. Without them require an actual contact
    plus proximity of the palm surface. Never infer contact from distance alone.
    """
    robot = states.robots[robot_name]
    n = robot.joint_pos.shape[0]
    any_contact = torch.zeros((n, 2), dtype=torch.bool, device=points.device)
    backrest = any_contact.clone()
    forces = points.new_zeros((n, 2))
    contact = getattr(robot, 'contact', None)
    if contact is None or contact['link_a'].shape[1] == 0:
        return any_contact, backrest, forces
    mapping = states.extras.get('global_link_map', {})
    chair_ids = [i for i, (entity, _) in mapping.items() if entity == 'chair']
    if not chair_ids:
        return any_contact, backrest, forces
    a, b = contact['link_a'], contact['link_b']
    valid = contact['valid_mask'].bool()
    chair_ids = torch.as_tensor(chair_ids, device=a.device)
    f = contact.get('force_b', contact.get('force'))
    if f is None:
        return any_contact, backrest, forces
    magnitude = torch.linalg.vector_norm(f, dim=-1)
    positions = contact.get('position', contact.get('pos'))
    for hand, side in enumerate(('left', 'right')):
        ids = [i for i, (entity, link) in mapping.items()
               if entity == robot_name and link.startswith(side + '_hand_')]
        if not ids:
            continue
        ids = torch.as_tensor(ids, device=a.device)
        pair = valid & ((torch.isin(a, ids) & torch.isin(b, chair_ids)) |
                        (torch.isin(b, ids) & torch.isin(a, chair_ids)))
        loaded = pair & (magnitude >= CONTACT_FORCE_MIN)
        any_contact[:, hand] = pair.any(-1)
        near = torch.linalg.vector_norm(points[:, hand]-targets[:, hand], dim=-1) <= CONTACT_RADIUS
        if positions is not None:
            region = torch.linalg.vector_norm(positions-targets[:, hand, None], dim=-1) <= CONTACT_RADIUS
            backrest[:, hand] = (loaded & region).any(-1) & near
        else:
            backrest[:, hand] = loaded.any(-1) & near
        forces[:, hand] = torch.where(pair, magnitude, 0.0).sum(-1)
    return any_contact, backrest, forces


def measure(states, robot_name, task):
    robot, chair = states.robots[robot_name], states.objects['chair']
    body = lambda name: robot.body_state[:, robot.body_names.index(name)]
    base, torso = body('pelvis'), body('torso_link')
    chair_body = chair.body_state[:, chair.body_names.index('base_link')]
    palms = torch.stack([body(s+'_hand_palm_link') for s in ('left', 'right')], 1)
    end_effectors = torch.stack([body('left_endeffector'), body('endeffector')], 1)
    targets = torch.stack([chair.body_state[:, chair.body_names.index('target_hand_'+s), :3]
                           for s in ('left', 'right')], 1)
    offsets = palms.new_tensor([[0.05, -0.02, 0], [0.05, 0.02, 0]])
    normals = palms.new_tensor([[0, -1, 0], [0, 1, 0]])
    rotated_offsets = rotate(palms[..., 3:7], offsets)
    points = palms[..., :3] + rotated_offsets
    normal = rotate(palms[..., 3:7], normals)
    palm_angle = angle_between(normal, normal.new_tensor([0, 0, -1]).expand_as(normal))
    elbows = []
    for side in ('left', 'right'):
        shoulder, elbow, wrist = (body(side+'_'+link)[:, :3]
                                  for link in ('shoulder_roll_link', 'elbow_link', 'wrist_roll_link'))
        elbows.append(angle_between(elbow-shoulder, wrist-elbow))
    elbow_angle = torch.stack(elbows, 1)
    chair_dir = chair_back_direction_xy(chair_body[:, 3:7])
    robot_anchor = getattr(task, 'chairman_robot_anchor', base[:, :2])
    chair_anchor = getattr(task, 'chairman_chair_anchor', chair_body[:, :3])
    direction = getattr(task, 'chairman_pull_direction', chair_dir)
    chair_heading = getattr(task, 'chairman_chair_heading', chair_dir)
    displacement = chair_body[:, :2] - chair_anchor[:, :2]
    pull = (displacement * direction).sum(-1)
    any_contact, contact, forces = hand_contacts(states, robot_name, points, targets)
    point_velocity = palms[..., 7:10] + torch.linalg.cross(palms[..., 10:13], rotated_offsets, dim=-1)
    target_velocity = chair_body[:, None, 7:10] + torch.linalg.cross(
        chair_body[:, None, 10:13].expand_as(targets), targets-chair_body[:, None, :3], dim=-1)
    arm_ids = [list(robot.joint_names).index(name) for name in STAGE0_JOINT_TARGETS]
    q = robot.joint_pos[:, arm_ids]
    upper_com = upper_body_center_of_mass(robot)
    upper_com_offset = upper_com[:, :2] - base[:, :2]
    forward = forward_direction_xy(torso[:, 3:7])
    hand_metrics = hand_task_space_metrics(torso, end_effectors, targets, chair_dir)
    heading = planar_angle(forward, chair_body[:, :2]-torso[:, :2])
    direction_valid = (torch.linalg.vector_norm(forward, dim=-1) > 1e-6) & (
        torch.linalg.vector_norm(chair_body[:, :2]-torso[:, :2], dim=-1) > 1e-6)
    heading = torch.where(direction_valid, heading, torch.full_like(heading, math.pi))
    return dict(
        approach_error=torch.linalg.vector_norm(base[:, :2]-(chair_body[:, :2]+APPROACH_DISTANCE*chair_dir), dim=-1),
        heading=heading, pose0=(q-q.new_tensor(list(STAGE0_JOINT_TARGETS.values()))).abs(),
        **hand_metrics,
        robot_drift=torch.linalg.vector_norm(base[:, :2]-robot_anchor, dim=-1),
        chair_drift=torch.linalg.vector_norm(displacement, dim=-1),
        chair_yaw=planar_angle(chair_dir, chair_heading),
        robot_speed=torch.linalg.vector_norm(base[:, 7:9], dim=-1), robot_yaw_speed=base[:, 12].abs(),
        chair_speed=torch.linalg.vector_norm(chair_body[:, 7:9], dim=-1), chair_yaw_speed=chair_body[:, 12].abs(),
        arm_speed=robot.joint_vel[:, arm_ids].abs().amax(-1),
        upper_body_com=upper_com,
        upper_body_com_offset=upper_com_offset,
        upper_body_com_horizontal_error=torch.linalg.vector_norm(upper_com_offset, dim=-1),
        palm_angle=palm_angle, elbow_angle=elbow_angle,
        palm_error=torch.linalg.vector_norm(points-targets, dim=-1),
        contact=contact, any_contact=any_contact, contact_force=forces,
        hand_slip=torch.linalg.vector_norm(point_velocity-target_velocity, dim=-1),
        pull=pull, pull_error=(pull-PULL_DISTANCE).abs(),
        lateral=torch.linalg.vector_norm(displacement-pull[:, None]*direction, dim=-1),
        backward_speed=(base[:, 7:9]*direction).sum(-1),
        sideways_speed=torch.linalg.vector_norm(base[:, 7:9]-(base[:, 7:9]*direction).sum(-1, keepdim=True)*direction, dim=-1),
        lift_error=torch.linalg.vector_norm(points-(targets+targets.new_tensor([0, 0, LIFT_HEIGHT])), dim=-1),
        lift_height=points[..., 2]-targets[..., 2],
        lift_xy=torch.linalg.vector_norm(points[..., :2]-targets[..., :2], dim=-1),
        upright=rotate(torso[:, 3:7], torso.new_tensor([0, 0, 1]))[:, 2],
        finite=torch.isfinite(robot.body_state).flatten(1).all(-1) & torch.isfinite(robot.joint_pos).all(-1)
               & torch.isfinite(chair.body_state).flatten(1).all(-1),
    )

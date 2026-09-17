"""Shared physical measurements for Chairman2 rewards and success checks.

Units: metres, seconds, radians. Palm normals are left -Y / right +Y in
G1's palm frames (also the signs used by its fixed end-effector offsets).
The palm mesh spans roughly +/-2 cm in Y; the old 7 cm EE offset is outside
that mesh. Use a point on the actual palm surface instead.
"""
import math
import torch
from metasim.utils.chair_navigation import chair_back_direction_xy, forward_direction_xy

NUM_STAGES = 5
TASK_VERSION = 'chairman2_five_stage_v1'
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
STAGE0_JOINT_TARGETS = {
    'left_shoulder_pitch_joint': 1.51, 'left_shoulder_roll_joint': 0.93,
    'left_shoulder_yaw_joint': 1.15, 'left_elbow_joint': -0.59, 'left_wrist_roll_joint': 0.0,
    'right_shoulder_pitch_joint': 1.51, 'right_shoulder_roll_joint': -0.93,
    'right_shoulder_yaw_joint': -1.15, 'right_elbow_joint': -0.59, 'right_wrist_roll_joint': 0.0,
}
STAGE1_JOINT_TARGETS = {
    'left_shoulder_pitch_joint': -1.66, 'left_shoulder_roll_joint': 0.23,
    'left_shoulder_yaw_joint': 0.0, 'left_elbow_joint': 1.45, 'left_wrist_roll_joint': 1.35,
    'right_shoulder_pitch_joint': -1.66, 'right_shoulder_roll_joint': -0.23,
    'right_shoulder_yaw_joint': 0.0, 'right_elbow_joint': 1.45, 'right_wrist_roll_joint': -1.35,
}


def rotate(q, v):
    q = torch.nn.functional.normalize(q, dim=-1)
    v = torch.broadcast_to(v, q.shape[:-1] + (3,))
    t = 2 * torch.linalg.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.linalg.cross(q[..., 1:], t, dim=-1)


def angle_between(a, b):
    return torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(a, b, dim=-1), dim=-1),
                       (a * b).sum(-1))


def planar_angle(a, b):
    return torch.atan2((a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]).abs(), (a*b).sum(-1))


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
    forward = forward_direction_xy(torso[:, 3:7])
    heading = planar_angle(forward, chair_body[:, :2]-torso[:, :2])
    direction_valid = (torch.linalg.vector_norm(forward, dim=-1) > 1e-6) & (
        torch.linalg.vector_norm(chair_body[:, :2]-torso[:, :2], dim=-1) > 1e-6)
    heading = torch.where(direction_valid, heading, torch.full_like(heading, math.pi))
    return dict(
        approach_error=torch.linalg.vector_norm(base[:, :2]-(chair_body[:, :2]+APPROACH_DISTANCE*chair_dir), dim=-1),
        heading=heading, pose0=(q-q.new_tensor(list(STAGE0_JOINT_TARGETS.values()))).abs(),
        pose1=(q-q.new_tensor(list(STAGE1_JOINT_TARGETS.values()))).abs(),
        robot_drift=torch.linalg.vector_norm(base[:, :2]-robot_anchor, dim=-1),
        chair_drift=torch.linalg.vector_norm(displacement, dim=-1),
        chair_yaw=planar_angle(chair_dir, chair_heading),
        robot_speed=torch.linalg.vector_norm(base[:, 7:9], dim=-1), robot_yaw_speed=base[:, 12].abs(),
        chair_speed=torch.linalg.vector_norm(chair_body[:, 7:9], dim=-1), chair_yaw_speed=chair_body[:, 12].abs(),
        arm_speed=robot.joint_vel[:, arm_ids].abs().amax(-1),
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

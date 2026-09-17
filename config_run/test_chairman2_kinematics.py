"""URDF checks for the specified poses and a feasible bilateral contact pose."""
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from metasim.utils import chairman2_geometry as g


class Chairman2KinematicsTest(unittest.TestCase):
    def test_contact_pose_is_reachable_without_moving_robot(self):
        joints = ET.parse('roboverse_data/robots/g1/urdf/g1_mygym_without_hand.urdf').getroot().findall('joint')
        for pose in (g.STAGE0_JOINT_TARGETS, g.STAGE1_JOINT_TARGETS):
            for name, angle in pose.items():
                joint = next(j for j in joints if j.get('name') == name)
                self.assertNotEqual(joint.get('type'), 'fixed')
                self.assertLessEqual(float(joint.find('limit').get('lower')), angle)
                self.assertGreaterEqual(float(joint.find('limit').get('upper')), angle)
        # Witness found by constrained FK fitting at nominal pelvis height.
        # It is a reachability regression, not a policy or commanded target.
        for side, sign, q in (
            ('left', -1, [-1.1556886, .13332045, .46640664, 1.28716329, 1.77451891]),
            ('right', 1, [-1.15579804, -.13311604, -.46754742, 1.28717572, -1.77693774]),
        ):
            values = dict(zip([n for n in g.STAGE1_JOINT_TARGETS if n.startswith(side)], q))
            base = np.eye(4)
            base[:3, :3] = Rotation.from_euler('z', -np.pi/2).as_matrix()
            base[:3, 3] = [0, g.APPROACH_DISTANCE, .8]
            transforms = {'pelvis': base}
            for _ in range(10):
                for joint in joints:
                    parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                    if child in transforms or parent not in transforms:
                        continue
                    origin = joint.find('origin')
                    transform = np.eye(4)
                    if origin is not None:
                        transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                        transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                    rotation = np.eye(4)
                    if joint.get('type') != 'fixed':
                        axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                        rotation[:3, :3] = Rotation.from_rotvec(axis*values.get(joint.get('name'), 0)).as_matrix()
                    transforms[child] = transforms[parent] @ transform @ rotation
            palm = transforms[side+'_hand_palm_link']
            point = palm[:3, 3] + palm[:3, :3] @ np.array([.05, sign*.02, 0])
            normal = palm[:3, :3] @ np.array([0, sign, 0])
            target = np.array([.15 if side == 'left' else -.15, .2902036238, .9632205982])
            self.assertLess(np.linalg.norm(point-target), .002)
            self.assertLess(np.arccos(-normal[2]), g.PALM_ANGLE_TOLERANCE)
            shoulder, elbow, wrist = [transforms[side+'_'+link][:3, 3] for link in
                                      ('shoulder_roll_link', 'elbow_link', 'wrist_roll_link')]
            upper, forearm = elbow-shoulder, wrist-elbow
            self.assertLess(np.arctan2(np.linalg.norm(np.cross(upper, forearm)), upper@forearm), g.ELBOW_ANGLE_TOLERANCE)


if __name__ == '__main__': unittest.main()

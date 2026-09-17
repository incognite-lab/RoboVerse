"""Five-stage success, contact geometry and per-row timing regressions."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from metasim.utils import chairman2_geometry as g
from metasim.cfg.checkers import stages_chairman2 as c
from metasim.cfg.checkers import _ChairMan2Checker


def scene(n=2):
    names = ['pelvis', 'torso_link'] + [s+'_'+link for s in ('left','right')
             for link in ('shoulder_roll_link','elbow_link','wrist_roll_link','hand_palm_link')]
    robot = SimpleNamespace(body_names=names, body_state=torch.zeros(n,len(names),13),
        joint_names=list(g.STAGE0_JOINT_TARGETS), joint_pos=torch.zeros(n,10), joint_vel=torch.zeros(n,10), contact=None)
    robot.body_state[:,:,3] = 1
    robot.body_state[:,:,2] = 1
    robot.body_state[:,0,1] = g.APPROACH_DISTANCE
    robot.body_state[:,1,1] = g.APPROACH_DISTANCE
    robot.body_state[:,:2,3:7] = torch.tensor([2**-.5,0,0,-2**-.5])
    chair = SimpleNamespace(body_names=['base_link','target_hand_left','target_hand_right'], body_state=torch.zeros(n,3,13))
    chair.body_state[:,:,3] = 1
    chair.body_state[:,1:,:3] = torch.tensor([[.15,.29,.96],[-.15,.29,.96]])
    for hand,side in enumerate(('left','right')):
        index=names.index(side+'_hand_palm_link')
        robot.body_state[:,index,3:7] = torch.tensor([2**-.5, (1 if hand==0 else -1)*2**-.5,0,0])
        q=robot.body_state[:,index,3:7]
        offset=torch.tensor([.05, -.02 if hand==0 else .02,0])
        robot.body_state[:,index,:3] = chair.body_state[:,hand+1,:3]-g.rotate(q,offset)
        shoulder=names.index(side+'_shoulder_roll_link'); elbow=names.index(side+'_elbow_link'); wrist=names.index(side+'_wrist_roll_link')
        robot.body_state[:,shoulder,:3]=torch.tensor([0.,0.,1.2])
        robot.body_state[:,elbow,:3]=torch.tensor([.2,0.,1.2])
        robot.body_state[:,wrist,:3]=torch.tensor([.4,0.,1.2])
    states=SimpleNamespace(robots={'g1_without_hands':robot},objects={'chair':chair},extras={'global_link_map':{
        1:('g1_without_hands','left_hand_palm_link'),2:('g1_without_hands','right_hand_palm_link'),3:('chair','base_link')}})
    handler=SimpleNamespace(robot=SimpleNamespace(name='g1_without_hands'),num_envs=n,device='cpu',
        scenario=SimpleNamespace(sim_params=SimpleNamespace(dt=.002),decimation=5), task=SimpleNamespace(), get_states=lambda:states)
    return states,handler


def contact(states):
    robot=states.robots['g1_without_hands']; n=robot.joint_pos.shape[0]
    robot.contact={'link_a':torch.tensor([[1,3]]).expand(n,-1), 'link_b':torch.tensor([[3,2]]).expand(n,-1),
        'valid_mask':torch.ones(n,2,dtype=torch.bool), 'force_b':torch.tensor([[[0.,0.,2.],[0.,0.,-2.]]]).expand(n,-1,-1),
        'position':states.objects['chair'].body_state[:,1:,:3].clone()}


class Chairman2CheckerTest(unittest.TestCase):
    def test_contact_requires_force_both_hands_and_backrest_region(self):
        states,h=scene(); m=g.measure(states,h.robot.name,h.task)
        self.assertFalse(m['contact'].any())
        contact(states); m=g.measure(states,h.robot.name,h.task)
        self.assertTrue(c.success_conditions(m)[:,2].all())
        states.robots[h.robot.name].contact['position'][0,0,2]-=.4
        states.robots[h.robot.name].contact['valid_mask'][1,1]=False
        m=g.measure(states,h.robot.name,h.task)
        self.assertFalse(c.success_conditions(m)[:,2].any())
        self.assertTrue(m['any_contact'][0].all())

    def test_wrong_palm_or_bent_elbow_cannot_be_compensated(self):
        states,h=scene(); contact(states); robot=states.robots[h.robot.name]
        robot.body_state[0,robot.body_names.index('left_hand_palm_link'),3:7]=torch.tensor([1.,0,0,0])
        robot.body_state[1,robot.body_names.index('right_wrist_roll_link'),:3]=torch.tensor([.2,.2,1.2])
        m=g.measure(states,h.robot.name,h.task)
        self.assertGreater(m['palm_angle'][0,0],g.PALM_ANGLE_TOLERANCE)
        self.assertGreater(m['elbow_angle'][1,1],g.ELBOW_ANGLE_TOLERANCE)
        self.assertFalse(c.success_conditions(m)[:,2].any())

    def test_walk_requires_pose_and_torso_heading_and_hold(self):
        states,h=scene(); robot=states.robots[h.robot.name]
        robot.joint_pos[:]=torch.tensor(list(g.STAGE0_JOINT_TARGETS.values()))
        robot.joint_pos[1,-1]+=.3
        stage=torch.zeros(2,dtype=torch.long)
        for _ in range(24):
            fail,success=c.evaluate_stages(states,h,stage)
            self.assertFalse(success.any())
        fail,success=c.evaluate_stages(states,h,stage)
        self.assertEqual(success.tolist(),[True,False])
        robot.body_state[0,robot.body_names.index('torso_link'),3:7]=torch.tensor([1.,0,0,0])
        _,success=c.evaluate_stages(states,h,stage)
        self.assertFalse(success.any())

    def test_pull_requires_contact_and_stopping_and_fixed_anchor(self):
        states,h=scene(); contact(states)
        stage=torch.full((2,),3,dtype=torch.long)
        c.evaluate_stages(states,h,stage)
        states.objects['chair'].body_state[:,:,:2]+=torch.tensor([0.,g.PULL_DISTANCE])
        states.robots[h.robot.name].body_state[:,:,:2]+=torch.tensor([0.,g.PULL_DISTANCE])
        states.robots[h.robot.name].contact['position'][:,:,:2]+=torch.tensor([0.,g.PULL_DISTANCE])
        for _ in range(40): fail,success=c.evaluate_stages(states,h,stage)
        self.assertTrue(success.all())
        states.objects['chair'].body_state[0,0,7]=.2
        states.robots[h.robot.name].contact['valid_mask'][1]=False
        fail,success=c.evaluate_stages(states,h,stage)
        self.assertFalse(success.any()); self.assertFalse(fail.any())
        for _ in range(10): fail,_=c.evaluate_stages(states,h,stage)
        self.assertTrue(fail[1]); self.assertFalse(fail[0])

    def test_release_requires_both_hands_clear_and_failure_wins(self):
        states,h=scene(); robot=states.robots[h.robot.name]
        for side in ('left','right'):
            robot.body_state[:,robot.body_names.index(side+'_hand_palm_link'),2]+=.12
        stage=torch.full((2,),4,dtype=torch.long)
        for _ in range(25): fail,success=c.evaluate_stages(states,h,stage)
        self.assertTrue(success.all())
        contact(states)
        self.assertFalse(c.evaluate_stages(states,h,stage)[1].any())
        robot.contact=None
        robot.body_state[0,robot.body_names.index('left_shoulder_roll_link'),2]=-.5
        fail,success=c.evaluate_stages(states,h,stage)
        self.assertTrue(fail[0]); self.assertFalse(success[0])

    def test_checker_attributes_transition_to_source_policy(self):
        states,h=scene(); r=SimpleNamespace(actual_stage=torch.zeros(2,dtype=torch.long),completed_stages=torch.zeros(2,dtype=torch.long),metrics=None)
        h.task.reward_functions=[r]; h.task.use_snapshot_curriculum=False
        states.robots[h.robot.name].joint_pos[:]=torch.tensor(list(g.STAGE0_JOINT_TARGETS.values()))
        checker=_ChairMan2Checker()
        for _ in range(25): checker.check(h)
        self.assertEqual(r.actual_stage.tolist(),[1,1])
        self.assertEqual(h.task.reward_stage.tolist(),[0,0])
        self.assertEqual(h.task.completed_stage_events.tolist(),[0,0])
        self.assertFalse(h.task.just_finished.any())
        checker.check(h)
        self.assertEqual(h.task.completed_stage_events.tolist(),[-1,-1])

    def test_hold_duration_scales_with_control_dt(self):
        for decimation in (5,10):
            states,h=scene(); contact(states); h.scenario.decimation=decimation
            count=math.ceil(.25/(.002*decimation))
            stage=torch.full((2,),2,dtype=torch.long)
            for _ in range(count-1): self.assertFalse(c.evaluate_stages(states,h,stage)[1].any())
            self.assertTrue(c.evaluate_stages(states,h,stage)[1].all())


if __name__=='__main__': unittest.main()

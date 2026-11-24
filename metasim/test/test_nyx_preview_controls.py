"""Camera controls preserve the free pose and reproduce the robot optical frame."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from config_run.nyx_preview_controls import PreviewCamera, PreviewWindow
from metasim.cfg.sensors import NyxGaussianSplatCameraCfg


class PreviewControlsTest(unittest.TestCase):
    def controller(self, **kwargs):
        free = NyxGaussianSplatCameraCfg(gaussian_splat_path='fixture.ply',pos=(0,0,1),look_at=(1,0,1))
        robot = NyxGaussianSplatCameraCfg(gaussian_splat_path='fixture.ply',mount_to='g1',mount_link='head',
            mount_pos=(.1,0,0),mount_quat=(2**-.5,2**-.5,0,0))
        link = SimpleNamespace(name='head',get_pos=Mock(return_value=np.array([[1,2,3]])),
                               get_quat=Mock(return_value=np.array([[2**-.5,0,0,2**-.5]])))
        sensor = SimpleNamespace(update_camera_pose=Mock())
        handler = SimpleNamespace(camera_inst_dict={'camera0':sensor},
                                  object_inst_dict={'g1':SimpleNamespace(links=[link])})
        return PreviewCamera(handler,free,robot,**kwargs),sensor,link

    def test_wasd_and_rf_translation_and_diagonal_speed(self):
        for key,direction in [('w',(1,0,0)),('s',(-1,0,0)),('a',(0,1,0)),
                              ('d',(0,-1,0)),('r',(0,0,1)),('f',(0,0,-1))]:
            with self.subTest(key=key):
                control,_,_ = self.controller()
                before = control.free_pos.copy()
                control.update({key},set(),.1)
                np.testing.assert_allclose(control.free_pos-before,np.array(direction)*.1,atol=1e-12)
        control,_,_ = self.controller()
        before = control.free_pos.copy()
        control.update({'w','d','r'},set(),.1)
        self.assertAlmostEqual(np.linalg.norm(control.free_pos-before),.1)

    def test_turning_pitch_shift_and_stall_limit(self):
        control,_,_ = self.controller(turn_speed=90)
        control.update({'q'},set(),.1)
        np.testing.assert_allclose(control.free_direction,[np.cos(np.pi/20),np.sin(np.pi/20),0],atol=1e-12)
        control.update({'e'},set(),.1)
        np.testing.assert_allclose(control.free_direction,[1,0,0],atol=1e-12)
        before = control.free_pos.copy()
        control.update({'w','shift'},set(),10)
        self.assertAlmostEqual(np.linalg.norm(control.free_pos-before),.3)
        for _ in range(30):
            control.update({'up'},set(),.1)
        self.assertLess(control.free_direction[2],1)
        control.update({'a'},set(),.1)
        self.assertTrue(np.isfinite(control.free_pos).all())

    def test_robot_camera_matches_optical_minus_z_plus_y_and_preserves_free_pose(self):
        control,sensor,link = self.controller()
        control.update({'w'},set(),.1)
        free = tuple(x.copy() for x in control.pose())
        control.update(set(),{'c'},0)
        self.assertTrue(control.robot_view)
        pos,lookat,up = control.pose()
        np.testing.assert_allclose(pos,[1,2.1,3],atol=1e-12)
        np.testing.assert_allclose(lookat,[0,2.1,3],atol=1e-12)
        np.testing.assert_allclose(up,[0,0,1],atol=1e-12)
        control.update({'w','q','r'},set(),.1)
        np.testing.assert_allclose(control.pose()[0],pos,atol=1e-12)
        link.get_pos.return_value = np.array([[4,5,6]])
        control.update(set(),set(),.1)
        np.testing.assert_allclose(control.pose()[0],[4,5.1,6],atol=1e-12)
        control.update(set(),{'tab'},0)
        for actual,expected in zip(control.pose(),free):
            np.testing.assert_allclose(actual,expected,atol=1e-12)
        self.assertFalse(control.robot_view)
        sensor.update_camera_pose.assert_called()

    def test_direct_camera_selection_and_home_reset(self):
        control,_,_ = self.controller()
        control.update({'w'},set(),.1)
        control.update(set(),{'2'},0)
        self.assertTrue(control.robot_view)
        control.update(set(),{'1'},0)
        self.assertFalse(control.robot_view)
        control.update(set(),{'h'},0)
        np.testing.assert_allclose(control.free_pos,[0,0,1])
        for invalid in (0,float('nan'),-1):
            with self.assertRaises(ValueError):
                self.controller(speed=invalid)

    def test_window_only_moves_on_held_keys_when_focused_and_consumes_presses(self):
        window = object.__new__(PreviewWindow)
        window.window = object()
        window.pressed = {'c'}
        window.keys = {'w':1,'c':2}
        window.glfw = SimpleNamespace(poll_events=Mock(),FOCUSED=10,PRESS=1,
            KEY_LEFT_SHIFT=3,KEY_RIGHT_SHIFT=4,get_window_attrib=Mock(return_value=True),
            KEY_LEFT_CONTROL=5,KEY_RIGHT_CONTROL=6,
            get_key=lambda win,key: int(key in (1,3)))
        held,pressed = window.poll()
        self.assertEqual(held,{'w','shift'})
        self.assertEqual(pressed,{'c'})
        self.assertEqual(window.poll()[1],set())
        window.glfw.get_window_attrib.return_value = False
        self.assertEqual(window.poll()[0],set())


if __name__ == '__main__':
    unittest.main()

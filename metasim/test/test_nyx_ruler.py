"""Metric geometry, keypad control, and scale recommendations for Nyx ruler."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from metasim.utils.nyx_ruler import RulerSettings,MeasuringRuler


def ruler(**kwargs):
    settings = RulerSettings(**kwargs)
    ticks = [1000]*int(np.ceil(settings.max_length_m))+[100]*9+[10]*9+[1]*9
    return MeasuringRuler(settings,[(t,Mock()) for t in ticks],[Mock(),Mock()])


def camera(**kwargs):
    return SimpleNamespace(gaussian_scale=(4.,4.,4.),gaussian_position=(0.,0.,2.),gaussian_z_up=True,**kwargs)


class RulerTest(unittest.TestCase):
    def test_default_dimensions_and_contiguous_box_geometry_to_one_millimetre(self):
        r = ruler()
        self.assertEqual(r.settings.size_m,(1.,.02,.02))
        for length in (.001,.01,.031,.999,1.,1.234,9.999,10.):
            r.length_mm = round(length*1000)
            r.rotation = Rotation.from_euler('xyz',[23,47,91],degrees=True)
            cursor = 0.
            for (ticks,_),pos in zip(r.parts,r.layout()):
                if pos[2] == -100:
                    continue
                local = r.rotation.inv().apply(pos-r.position)
                np.testing.assert_allclose(local,[cursor+ticks/2000,0,0],atol=1e-12)
                cursor += ticks/1000
            self.assertAlmostEqual(cursor,length,places=12)
            self.assertAlmostEqual(np.linalg.norm(r.endpoint_b-r.position),length,places=12)

    def test_endpoint_a_is_anchored_during_resize_and_reference_is_retained(self):
        r = ruler()
        start = r.position.copy()
        r.update(set(),{'kp_enter'},0,camera())
        r.update(set(),{'kp_add'},0,camera())
        self.assertEqual(r.length_m,1.01)
        self.assertEqual(r.reference_length_m,1.)
        r.update({'control'},{'kp_subtract'},0,camera())
        self.assertEqual(r.length_m,1.009)
        np.testing.assert_array_equal(r.position,start)
        r.length_mm = 1
        r.update(set(),{'kp_subtract'},0,camera())
        self.assertEqual(r.length_m,.001)
        r.length_mm = 10000
        r.update(set(),{'kp_add'},0,camera())
        self.assertEqual(r.length_m,10.)

    def test_keypad_movement_and_rotation_do_not_conflict_with_camera_digits(self):
        for key,vector in [('kp_6',[1,0,0]),('kp_4',[-1,0,0]),('kp_8',[0,1,0]),
                           ('kp_2',[0,-1,0]),('kp_9',[0,0,1]),('kp_3',[0,0,-1])]:
            r = ruler()
            before = r.position.copy()
            r.update({key},set(),.1,camera())
            np.testing.assert_allclose(r.position-before,np.array(vector)*.025,atol=1e-12)
        r = ruler()
        r.update({'kp_7'},set(),.1,camera())
        self.assertGreater(r.endpoint_b[1],r.position[1])
        r.update({'kp_divide'},set(),.1,camera())
        self.assertNotEqual(r.endpoint_b[2],r.position[2])
        old = r.position.copy()
        r.update({'w','2','r'},set(),.1,camera())
        np.testing.assert_array_equal(old,r.position)

    def test_scale_ratio_and_translation_preserve_endpoint_a(self):
        r = ruler()
        r.length_mm = 2000
        m = r.measurement(camera())
        self.assertEqual(m['scale_correction'],.5)
        self.assertEqual(m['recommended_gaussian_scale'],[2.,2.,2.])
        old_t = np.array(camera().gaussian_position)
        new_t = np.array(m['recommended_gaussian_position'])
        np.testing.assert_allclose((r.position-old_t)*.5+new_t,r.position)
        self.assertEqual(m['size_m'],[2.,.02,.02])
        cfg = camera()
        cfg.gaussian_z_up = False
        m = r.measurement(cfg)
        a = r.position[[0,2,1]]*np.array([1,1,-1])
        np.testing.assert_allclose((a-old_t)*.5+np.array(m['recommended_gaussian_position']),a)

    def test_visibility_reset_and_updates_only_on_change(self):
        r = ruler()
        r.apply()
        for _,entity in r.parts:
            entity.set_pos.reset_mock()
        r.apply()
        self.assertTrue(all(not entity.set_pos.called for _,entity in r.parts))
        r.update(set(),{'kp_decimal'},0,camera())
        self.assertFalse(r.visible)
        self.assertTrue(all(pos[2] == -100 for pos in r.layout()))
        r.update({'kp_6'},{'kp_add'},.1,camera())
        r.update(set(),{'kp_0'},0,camera())
        self.assertTrue(r.visible)
        self.assertEqual(r.length_m,1.)
        np.testing.assert_array_equal(r.position,r.settings.position)

    def test_json_report_and_finite_positive_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'nested/report.json'
            r = ruler(report_path=str(path))
            result = r.save(camera())
            self.assertTrue(path.is_file())
            self.assertEqual(result['length_resolution_m'],.001)
        for kwargs in [dict(size_m=(0,.02,.02)),dict(size_m=(.0001,.02,.02)),
                       dict(size_m=(1,.02,float('nan'))),dict(position=(0,0)),dict(move_speed=0),
                       dict(reference_length_m=-1),dict(max_length_m=100),dict(length_step_m=.0001),
                       dict(enabled='false')]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                RulerSettings(**kwargs)


if __name__ == '__main__':
    unittest.main()

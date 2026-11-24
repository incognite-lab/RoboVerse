"""The simulator floor can be collision-only without hiding other geometry."""
from pathlib import Path
from dataclasses import fields, MISSING
import unittest

import yaml
from metasim.cfg.scenario import ScenarioCfg


class NyxGroundTest(unittest.TestCase):
    def test_default_scenario_keeps_original_visible_ground(self):
        field = next(field for field in fields(ScenarioCfg) if field.name == 'render_ground')
        default = field.default if field.default is not MISSING else field.default_factory()
        self.assertIs(default,True)

    def test_separate_nyx_configs_hide_ground_but_keep_sim_geometry(self):
        folder = Path(__file__).resolve().parents[2]/'config_run/configs/chairman_separate'
        paths = list(folder.glob('train_dagger*.yaml'))+[folder/'eval_dagger.yaml',
                folder/'eval_dagger_video.yaml',folder/'calibrate_nyx.yaml']
        for path in paths:
            with self.subTest(path=path.name):
                config = yaml.safe_load(path.read_text())
                self.assertIs(config['render_ground'],False)
                self.assertTrue(all(camera['params']['render_sim_geometry']
                                    for camera in config['cameras'].values()))


if __name__ == '__main__':
    unittest.main()

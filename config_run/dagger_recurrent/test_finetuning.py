"""Visual initialization, frozen warmup, fine-tuning and offline resume."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torchvision.models import ResNet18_Weights, resnet18

from .buffer import train_sequence_step
from .checkpoint import IMU_LAYOUT, load_student, model_kwargs, restore_optimizer, save_checkpoint
from .finetuning import VisualFineTuning
from .inputs import IMU_BODY_NAME
from .student_net import RecurrentVisionStudent
from .test_pipeline import FakeEnvironment, TinyStudent


class FineTuningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(41)

    @staticmethod
    def calibrated_backbone(weights):
        # Network-free unit fixture. A separate verification uses real ImageNet weights.
        assert weights is ResNet18_Weights.DEFAULT
        model = resnet18(weights=None)
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.fill_(.2)
                module.running_var.fill_(.7)
                module.num_batches_tracked.fill_(17)
        return model

    def pretrained_model(self, **kwargs):
        with patch('config_run.dagger_recurrent.student_net.resnet18', side_effect=self.calibrated_backbone):
            return RecurrentVisionStudent(pretrained_visual=True, hidden_size=8, **kwargs)

    @staticmethod
    def batch(model):
        values = (torch.rand(2, 3, 3, 32, 32), torch.randn(2, 3, 43)*.2,
                  torch.zeros(2, 3, 43), torch.zeros(2, 3, 6), torch.zeros(2, 3, 34))
        target = torch.randn(2, 3, 34)*.1
        return SimpleNamespace(sample=lambda *args: (values, target, model.initial_hidden(2),
                                                      torch.zeros(2, 3, dtype=torch.bool)))

    @staticmethod
    def update(model, optimizer, schedule, batch):
        health = {}
        loss = train_sequence_step(model, optimizer, batch, 2, 2, 1, diagnostics=health)
        unlocked = schedule.after_update(optimizer)
        return loss, health, unlocked

    def test_pretrained_batchnorm_is_preserved_and_stays_frozen_through_finetuning(self):
        model = self.pretrained_model(visual_batch_size=1)
        norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        self.assertTrue(norms)
        self.assertFalse(any(isinstance(module, torch.nn.GroupNorm) for module in model.visual_encoder.modules()))
        schedule = VisualFineTuning(model, dict(dagger_new_visual_warmup_updates=2), pretrained=True)
        optimizer = schedule.make_optimizer()
        batch = self.batch(model)
        original = {name: value.clone() for name,value in model.visual_encoder.state_dict().items()}
        before_gru = model.gru.weight_ih_l0.detach().clone()
        for count in (1, 2):
            loss, health, unlocked = self.update(model, optimizer, schedule, batch)
            self.assertTrue(loss > 0)
            self.assertEqual(unlocked, count == 2)
            self.assertEqual(health['visual_gradient_norm'], 0.)
            self.assertGreater(health['gru_gradient_norm'], 0.)
            for name, value in model.visual_encoder.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(before_gru, model.gru.weight_ih_l0))
        self.assertTrue(all(p.requires_grad for p in model.visual_encoder.parameters()))
        self.assertEqual(optimizer.param_groups[1]['lr'], 1e-5)
        _, health, _ = self.update(model, optimizer, schedule, batch)
        self.assertGreater(health['visual_gradient_norm'], 0.)
        self.assertFalse(torch.equal(model.visual_encoder.conv1.weight, original['conv1.weight']))
        for module in norms:
            self.assertFalse(module.training)
            torch.testing.assert_close(module.running_mean, torch.full_like(module.running_mean, .2))
            torch.testing.assert_close(module.running_var, torch.full_like(module.running_var, .7))
            self.assertEqual(int(module.num_batches_tracked), 17)

    def test_scratch_uses_groupnorm_and_trains_immediately_even_with_long_warmup(self):
        model = RecurrentVisionStudent(hidden_size=8)
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.visual_encoder.modules()))
        schedule = VisualFineTuning(model, dict(dagger_new_visual_warmup_updates=99999), pretrained=False)
        optimizer = schedule.make_optimizer()
        self.assertTrue(schedule.active)
        self.assertEqual(schedule.metrics()['visual_warmup_updates_remaining'], 0)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [1e-4, 1e-5])
        grouped = [id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters()})
        before = model.visual_encoder.conv1.weight.detach().clone()
        _, health, _ = self.update(model, optimizer, schedule, self.batch(model))
        self.assertGreater(health['visual_gradient_norm'], 0)
        self.assertFalse(torch.equal(before, model.visual_encoder.conv1.weight))

    def test_zero_warmup_enables_immediate_finetuning(self):
        model = self.pretrained_model()
        schedule = VisualFineTuning(model, dict(dagger_new_visual_warmup_updates=0), pretrained=True)
        self.assertTrue(schedule.active)
        self.assertTrue(all(p.requires_grad for p in model.visual_encoder.parameters()))

    def test_pretrained_resume_is_offline_and_preserves_updates_lr_and_optimizer(self):
        config = dict(dagger_new_pretrained_visual=True, dagger_new_hidden_size=8,
                      dagger_new_visual_warmup_updates=3)
        model = self.pretrained_model()
        schedule = VisualFineTuning(model, config, pretrained=True)
        optimizer = schedule.make_optimizer()
        batch = self.batch(model)
        self.update(model, optimizer, schedule, batch)
        env = FakeEnvironment()
        metadata = dict(model_kwargs=model_kwargs(config, 43), joint_names=list(env.robot_joint_names),
                        action_names=list(env.action_names), imu_layout=IMU_LAYOUT, imu_body_name=IMU_BODY_NAME,
                        camera_name='camera0', control_dt=.02, image_shape=[3,32,32],
                        pretrained_visual=True, visual_finetuning=schedule.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'warmup.pth'
            save_checkpoint(path, model, optimizer, 20, metadata)
            for _ in range(3):
                self.update(model, optimizer, schedule, batch)
            # weights=None here is deliberate: all calibrated BN buffers and
            # encoder weights must come from the checkpoint, without a download.
            with patch('config_run.dagger_recurrent.student_net.resnet18', wraps=resnet18) as builder:
                restored, meta, _ = load_student(path, env, env.base, 'cpu')
                builder.assert_called_once_with(weights=None)
            resumed = VisualFineTuning(restored, {}, pretrained=True, saved=meta['visual_finetuning'])
            restored_optimizer = resumed.make_optimizer()
            self.assertEqual(restore_optimizer(path, restored_optimizer, 'cpu', student=restored), 20)
            resumed.apply(restored_optimizer)
            self.assertFalse(resumed.active)
            self.assertEqual(resumed.completed_updates, 1)
            self.assertEqual(restored_optimizer.param_groups[1]['lr'], 0)
            for _ in range(3):
                self.update(restored, restored_optimizer, resumed, batch)
            self.assertTrue(resumed.active)
            self.assertEqual(resumed.completed_updates, 4)
            for key, expected in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)
            meta['visual_finetuning'] = resumed.state_dict()
            save_checkpoint(path, restored, restored_optimizer, 40, meta)
            reloaded, meta, _ = load_student(path, env, env.base, 'cpu')
            active = VisualFineTuning(reloaded, {}, pretrained=True, saved=meta['visual_finetuning'])
            self.assertTrue(active.active)
            self.assertEqual(active.completed_updates, 4)

    def test_legacy_single_group_optimizer_migration_preserves_each_parameter_momentum(self):
        model = RecurrentVisionStudent(hidden_size=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        train_sequence_step(model, optimizer, self.batch(model), 2, 2, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'previous_v2.pth'
            save_checkpoint(path, model, optimizer, 42, {})
            schedule = VisualFineTuning(model, {}, pretrained=False)
            migrated = schedule.make_optimizer()
            restore_optimizer(path, migrated, 'cpu', student=model)
            for parameter, state in optimizer.state.items():
                for key, value in state.items():
                    torch.testing.assert_close(migrated.state[parameter][key], value, rtol=0, atol=0)
            self.assertEqual([g['name'] for g in migrated.param_groups], ['policy','visual'])

    def test_invalid_learning_rates_and_warmup_fail_before_training(self):
        model = RecurrentVisionStudent(hidden_size=8)
        for config in (dict(dagger_new_visual_learning_rate=-1), dict(dagger_new_visual_learning_rate=float('nan')),
                       dict(dagger_new_visual_learning_rate=1.), dict(dagger_new_visual_warmup_updates=-1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                VisualFineTuning(model, config)

    def test_runner_resumes_warmup_and_refuses_changing_visual_initialization(self):
        from . import runner
        import numpy as np
        env = FakeEnvironment()
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.full((2,34), .2, np.float32))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, '_device', return_value='cpu'), \
                patch.object(runner, 'load_policy_router', return_value=(teacher, {})), \
                patch.object(runner, 'RecurrentVisionStudent', TinyStudent), \
                patch('config_run.dagger_recurrent.checkpoint.RecurrentVisionStudent', TinyStudent), \
                patch('config_run.dagger_recurrent.student_net.resnet18',
                      side_effect=lambda weights=None: resnet18(weights=None)):
            config = dict(train_or_eval='train_dagger_new', expert_model_path='fake', model_save_path=directory,
                          tensorboard_log=directory+'/logs', total_timesteps=1, dagger_new_hidden_size=8,
                          dagger_new_pretrained_visual=True, dagger_new_visual_warmup_updates=2,
                          dagger_image_width=32, dagger_image_height=32, dagger_buffer_steps=2,
                          dagger_new_sequence_length=1, dagger_new_burn_in=0, dagger_new_batch_size=1,
                          dagger_train_every=1, dagger_updates_per_train=1, dagger_min_train_sample_ratio=0,
                          beta_schedule='linear', beta_start=1., beta_end=1., model_save_freq=0)
            runner.train_dagger(config, env.base, env)
            path = Path(directory)/'recurrent/student_model_final.pth'
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved['metadata']['visual_finetuning']['completed_updates'], 1)
            original = saved['student_state_dict']['visual_encoder.1.weight'].clone()
            config.update(train_or_eval='load_and_train_dagger_new', load_student_path=str(path), total_timesteps=4)
            runner.train_dagger(config, env.base, env)
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved['metadata']['visual_finetuning']['completed_updates'], 4)
            self.assertFalse(torch.equal(original, saved['student_state_dict']['visual_encoder.1.weight']))
            config.update(dagger_new_pretrained_visual=False, total_timesteps=5)
            with self.assertRaisesRegex(ValueError, 'Cannot change pretrained ResNet initialization'):
                runner.train_dagger(config, env.base, env)


if __name__ == '__main__':
    unittest.main()

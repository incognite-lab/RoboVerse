"""Reproductions of temporal archive gaps, teacher-only deadlock and torn resume."""
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import runner
from .buffer import SequenceBuffer
from .checkpoint import load_student, restore_optimizer, save_checkpoint
from .evaluation import EvaluationBeta
from .test_pipeline import FakeEnvironment, TinyStudent


class TrainingAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_archive_supervision_has_no_burn_in_gaps_even_after_ring_wrap(self):
        replay = SequenceBuffer(7, 1, (3,1,1), 43, (1,2), demonstrations_per_stage=64,
                                demonstration_length=4, demonstration_burn_in=2)
        for step in range(30):
            marker = torch.full((1,1), float(step))
            values = (torch.full((1,3,1,1), step, dtype=torch.uint8), marker.expand(1,43),
                      marker.expand(1,43), marker.expand(1,6), marker.expand(1,34))
            replay.add(values, marker.expand(1,34), marker.view(1,1,1).expand(1,1,2),
                       torch.tensor([step % 5 == 0]))
            # Previously every occurrence of this brief phase was in burn-in.
            replay.record_demonstrations(torch.tensor([4 if step % 4 == 0 else 0]), torch.tensor([True]))
        covered = set()
        for windows in replay.demonstrations.values():
            for image, proprio, target, hidden, reset in windows:
                torch.testing.assert_close(target[1:,0]-target[:-1,0], torch.ones(3))
                torch.testing.assert_close(image[:,0,0,0].float(), target[:,0])
                torch.testing.assert_close(proprio[:,0], target[:,0])
                self.assertTrue(torch.equal(reset, target[:,0].long() % 5 == 0))
                covered.update(target[2:,0].long().tolist())
        self.assertEqual(covered, set(range(2,30)))
        self.assertIn(4, replay.demonstrations)
        self.assertTrue(any((window[2][2:,0].long() % 4 == 0).any() for window in replay.demonstrations[4]))
        replay.clear()
        self.assertEqual(replay.steps_added, 0)

    def test_failed_student_still_gets_on_policy_data_after_teacher_warmup(self):
        config = dict(dagger_teacher_warmup_steps=100, dagger_beta_after_warmup=.9,
                      dagger_eval_required_passes=2)
        gate = EvaluationBeta(config)
        failed = dict(episodes_finished=32, requested_episodes=32, success_rate=0.)
        passed = dict(failed, success_rate=1.)
        self.assertEqual(gate.beta_at_step(99), 1.)
        self.assertEqual(gate.beta_at_step(100), .9)
        for step in range(100,200):
            self.assertLessEqual(gate.observe(failed, step), .9)
        self.assertAlmostEqual(gate.observe(passed,200), .9)
        self.assertAlmostEqual(gate.observe(passed,201), .8)
        gate.observe(failed,202)
        self.assertAlmostEqual(gate.observe(failed,203), .9)
        resumed = EvaluationBeta(config, gate.state_dict(), start_step=500)
        self.assertAlmostEqual(resumed.beta_at_step(501), .9)
        default_gate = EvaluationBeta(dict(dagger_teacher_warmup_steps=0))
        self.assertEqual(default_gate.beta_at_step(0), .9)
        fixed_teacher = EvaluationBeta(dict(beta_start=1., beta_end=1., dagger_teacher_warmup_steps=0))
        self.assertEqual(fixed_teacher.beta_at_step(0), 1.)

    def test_collection_transitions_from_teacher_to_student_with_anchors_preserved(self):
        env = FakeEnvironment()
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.full((2,34), .2, np.float32))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, '_device', return_value='cpu'), \
                patch.object(runner, 'load_policy_router', return_value=(teacher, {})), \
                patch.object(runner, 'RecurrentVisionStudent', TinyStudent):
            runner.train_dagger(dict(expert_model_path='fake', model_save_path=directory,
                                     tensorboard_log=directory+'/logs', total_timesteps=6,
                                     dagger_image_width=32, dagger_image_height=32,
                                     dagger_buffer_steps=24, dagger_new_hidden_size=8,
                                     dagger_train_every=100, beta_schedule='evaluation',
                                     dagger_teacher_warmup_steps=2, dagger_beta_after_warmup=0.,
                                     dagger_eval_every=100, dagger_teacher_anchor_fraction=.5), env.base, env)
        for actions in env.actions[:2]:
            np.testing.assert_array_equal(actions, np.full((2,34), .2, np.float32))
        for actions in env.actions[2:]:
            self.assertEqual(int(np.all(np.isclose(actions,.2),axis=1).sum()),1)

    def test_same_step_wrong_optimizer_is_rejected_without_mutating_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = torch.nn.Linear(2,1), torch.nn.Linear(2,1)
            optimizer = torch.optim.AdamW(first.parameters(),lr=1e-4)
            other_optimizer = torch.optim.AdamW(second.parameters(),lr=9e-4)
            save_checkpoint(root/'first.pth', first, optimizer, 10, {})
            save_checkpoint(root/'second.pth', second, other_optimizer, 10, {})
            shutil.copyfile(root/'second.pth.training.pth', root/'first.pth.training.pth')
            with self.assertRaisesRegex(ValueError,'different checkpoint generations'):
                restore_optimizer(root/'first.pth',optimizer,'cpu')
            self.assertEqual(optimizer.param_groups[0]['lr'],1e-4)

    def test_step_mismatch_is_rejected_before_optimizer_state_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pth'
            model = torch.nn.Linear(2,1)
            optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
            save_checkpoint(path,model,optimizer,10,{})
            state_path = Path(str(path)+'.training.pth')
            payload = torch.load(state_path,weights_only=True)
            payload['next_step']=9
            payload['optimizer_state_dict']['param_groups'][0]['lr']=9e-4
            torch.save(payload,state_path)
            with self.assertRaisesRegex(ValueError,'steps differ'):
                restore_optimizer(path,optimizer,'cpu')
            self.assertEqual(optimizer.param_groups[0]['lr'],1e-4)

    def test_interrupted_training_preserves_final_and_can_resume_with_its_own_archive(self):
        env = FakeEnvironment()
        teacher = SimpleNamespace(predict=lambda obs, deterministic: np.full((2,34),.2,np.float32))
        original_train = runner.train_sequence_step
        count = 0
        def interrupted_update(*args,**kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('injected interruption')
            return original_train(*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner,'_device',return_value='cpu'), \
                patch.object(runner,'load_policy_router',return_value=(teacher,{})), \
                patch.object(runner,'RecurrentVisionStudent',TinyStudent), \
                patch('config_run.dagger_recurrent.checkpoint.RecurrentVisionStudent',TinyStudent):
            root = Path(directory)/'recurrent'
            root.mkdir()
            existing = ['student_model_final.pth','student_model_final.pth.training.pth','teacher_demonstrations.pth']
            for name in existing:
                (root/name).write_bytes(b'previous successful training')
            config = dict(train_or_eval='train_dagger_new',expert_model_path='fake',model_save_path=directory,
                          tensorboard_log=directory+'/logs',total_timesteps=4,dagger_new_hidden_size=8,
                          dagger_image_width=32,dagger_image_height=32,dagger_buffer_steps=2,
                          dagger_new_sequence_length=1,dagger_new_burn_in=0,dagger_new_batch_size=1,
                          dagger_train_every=1,dagger_updates_per_train=1,dagger_min_train_sample_ratio=0,
                          dagger_demonstrations_per_stage=2,beta_schedule='linear',beta_start=1.,beta_end=1.,
                          model_save_freq=0)
            with patch.object(runner,'train_sequence_step',side_effect=interrupted_update):
                with self.assertRaisesRegex(RuntimeError,'injected interruption'):
                    runner.train_dagger(config,env.base,env)
            for name in existing:
                self.assertEqual((root/name).read_bytes(),b'previous successful training')
            recovery = root/'student_model_interrupted_step_1.pth'
            student,metadata,step = load_student(recovery,env,env.base,'cpu')
            self.assertEqual(step,1)
            self.assertTrue((root/metadata['demonstration_archive']).is_file())
            self.assertNotEqual(metadata['demonstration_archive'],'teacher_demonstrations.pth')
            config.update(train_or_eval='load_and_train_dagger_new',load_student_path=str(recovery),total_timesteps=3)
            runner.train_dagger(config,env.base,env)
            finished = torch.load(root/'student_model_final.pth',weights_only=True)
            self.assertEqual(finished['next_step'],3)


if __name__ == '__main__':
    unittest.main()

"""Temporal consistency, clean replay, frozen-backbone learning, and RNG resume."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .. import visual_augmentation as aug
from . import runner
from .buffer import SequenceBuffer,train_sequence_step
from .finetuning import VisualFineTuning
from .test_pipeline import FakeEnvironment,TinyStudent


class VisualAugmentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def settings(self, **values):
        defaults = dict(BRIGHTNESS_RANGE=(.8,.8),CONTRAST_RANGE=(1.,1.),GAMMA_RANGE=(1.,1.),
                        WHITE_BALANCE_RANGE=(1.,1.),NOISE_STD_RANGE=(0.,0.),
                        BLUR_PROBABILITY=0.,CLEAN_PROBABILITY=0.)
        defaults.update(values)
        return patch.multiple(aug,**defaults)

    def test_clean_images_are_exact_and_source_is_never_modified(self):
        images = torch.rand(3,4,3,12,12)
        before = images.clone()
        with self.settings(CLEAN_PROBABILITY=1.):
            transform = aug.VisualAugmentation(2)
            result = transform.sequence(images)
        self.assertNotEqual(result.data_ptr(),images.data_ptr())
        torch.testing.assert_close(result,before,rtol=0,atol=0)
        torch.testing.assert_close(images,before,rtol=0,atol=0)

    def test_sampling_ignores_global_default_device_set_by_simulator(self):
        # Meta exercises a non-CPU global default without requiring a GPU.
        with torch.device('meta'):
            transform = aug.VisualAugmentation(5)
            for force in (False,True):
                parameters = transform.sample(3,force=force)
                for name in parameters.describe():
                    self.assertEqual(getattr(parameters,name).device.type,'cpu')

    def test_one_style_spans_all_24_frames_including_chunk_boundaries(self):
        images = torch.rand(1,1,3,12,12).expand(2,24,3,12,12).clone()
        with self.settings(BRIGHTNESS_RANGE=(.7,1.1),WHITE_BALANCE_RANGE=(.9,1.1),
                           GAMMA_RANGE=(.8,1.2),BLUR_PROBABILITY=1.,IMAGE_CHUNK_SIZE=5):
            transform = aug.VisualAugmentation(7)
            result = transform.sequence(images)
        torch.testing.assert_close(result,result[:,:1].expand_as(result),rtol=0,atol=0)
        self.assertFalse(torch.equal(result[0],result[1]))
        self.assertFalse(torch.equal(result,images))

    def test_sensor_noise_is_new_per_frame_and_outputs_stay_in_range(self):
        with self.settings(BRIGHTNESS_RANGE=(1.,1.),NOISE_STD_RANGE=(.03,.03)):
            result = aug.VisualAugmentation(5).sequence(torch.full((2,3,3,12,12),.5))
        self.assertFalse(torch.equal(result[0,0],result[0,1]))
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()),0)
        self.assertLessEqual(float(result.max()),1)

    def test_rollout_styles_only_change_for_reset_environments(self):
        transform = aug.VisualAugmentation(9)
        images = torch.full((3,3,12,12),.5)
        transform.rollout(images,torch.ones(3,dtype=torch.bool))
        first = transform.episode_parameters.describe()
        transform.rollout(images,torch.zeros(3,dtype=torch.bool))
        self.assertEqual(first,transform.episode_parameters.describe())
        transform.rollout(images,torch.tensor([False,True,False]))
        after = transform.episode_parameters.describe()
        for key in first:
            for row in (0,2):
                self.assertEqual(first[key][row],after[key][row])
        self.assertNotEqual(first['brightness'][1],after['brightness'][1])
        torch.testing.assert_close(images,torch.full_like(images,.5),rtol=0,atol=0)

    def test_independent_rng_and_exact_resume_of_sampling_and_noise(self):
        images = torch.full((8,2,3,12,12),.5)
        global_state = torch.get_rng_state().clone()
        transform = aug.VisualAugmentation(6)
        transform.sequence(images)
        saved = transform.state_dict()
        expected = transform.sequence(images)
        resumed = aug.VisualAugmentation(999,saved=saved)
        actual = resumed.sequence(images)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),global_state,rtol=0,atol=0)

    def test_chunking_preserves_results_without_sensor_noise(self):
        images = torch.rand(3,5,3,12,12)
        with self.settings(BLUR_PROBABILITY=1.):
            parameters = aug.VisualAugmentation(4).sample(3,force=True)
            with patch.object(aug,'IMAGE_CHUNK_SIZE',1):
                first = aug.augment_visual_input(images,parameters)
            with patch.object(aug,'IMAGE_CHUNK_SIZE',64):
                second = aug.augment_visual_input(images,parameters)
        torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_burn_in_and_supervised_tail_share_augmentation_while_frozen_resnet_stays_fixed(self):
        class RecordingStudent(TinyStudent):
            def __init__(self,**kwargs):
                super().__init__(**kwargs)
                self.seen = []
            def forward(self,*values,**kwargs):
                self.seen.append(values[0].detach().clone())
                return super().forward(*values,**kwargs)
        student = RecordingStudent(hidden_size=8)
        schedule = VisualFineTuning(student,dict(dagger_new_finetune_visual=False))
        optimizer = schedule.make_optimizer()
        before_visual = {k:v.clone() for k,v in student.visual_encoder.state_dict().items()}
        before_gru = student.gru.weight_ih_l0.detach().clone()
        images = torch.full((2,3,3,32,32),.4)
        target = torch.full((2,3,34),.1)
        values = (images,torch.zeros(2,3,43),torch.zeros(2,3,43),torch.zeros(2,3,6),torch.zeros(2,3,34))
        replay = SimpleNamespace(sample=lambda *args:(tuple(v.clone() for v in values),target.clone(),
                     student.initial_hidden(2),torch.zeros(2,3,dtype=torch.bool)))
        health = {}
        with self.settings():
            loss = train_sequence_step(student,optimizer,replay,2,2,1,diagnostics=health,
                                       augmentation=aug.VisualAugmentation(3))
        self.assertTrue(np.isfinite(loss))
        self.assertEqual([x.shape[1] for x in student.seen],[1,2])
        for seen in student.seen:
            torch.testing.assert_close(seen,torch.full_like(seen,.32))
        torch.testing.assert_close(images,torch.full_like(images,.4),rtol=0,atol=0)
        for key,value in student.visual_encoder.state_dict().items():
            torch.testing.assert_close(value,before_visual[key],rtol=0,atol=0)
        self.assertFalse(torch.equal(before_gru,student.gru.weight_ih_l0))
        self.assertEqual(health['visual_gradient_norm'],0)
        self.assertGreater(health['gru_gradient_norm'],0)

    def test_runner_stores_clean_rgb_resumes_augmentation_and_evaluates_clean(self):
        captured,stored = [],[]
        class Student(TinyStudent):
            def forward(self,*values,**kwargs):
                if not self.training:
                    captured.append(values[0].detach().clone())
                return super().forward(*values,**kwargs)
        class Replay(SequenceBuffer):
            def add(self,values,*args):
                stored.append((values[0].clone(),args[0].clone()))
                return super().add(values,*args)
        env = FakeEnvironment()
        env.states.cameras['camera0'].rgb.fill_(100)
        teacher = SimpleNamespace(predict=lambda obs,deterministic:np.full((2,34),.2,np.float32))
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            stack.enter_context(self.settings())
            stack.enter_context(patch.object(runner,'_device',return_value='cpu'))
            stack.enter_context(patch.object(runner,'load_policy_router',return_value=(teacher,{})))
            stack.enter_context(patch.object(runner,'RecurrentVisionStudent',Student))
            stack.enter_context(patch('config_run.dagger_recurrent.checkpoint.RecurrentVisionStudent',Student))
            stack.enter_context(patch.object(runner,'SequenceBuffer',Replay))
            config=dict(train_or_eval='train_dagger_new',expert_model_path='fake',
                model_save_path=folder,tensorboard_log=folder+'/logs',total_timesteps=4,
                dagger_visual_augmentation=True,dagger_new_finetune_visual=False,
                dagger_new_hidden_size=8,dagger_image_width=32,dagger_image_height=32,
                dagger_buffer_steps=4,dagger_new_sequence_length=2,dagger_new_burn_in=1,
                dagger_new_batch_size=1,dagger_train_every=2,dagger_updates_per_train=1,
                dagger_min_train_sample_ratio=0,beta_schedule='linear',beta_start=1.,beta_end=1.,
                dagger_log_every=2,model_save_freq=0,eval_episodes=2,eval_total_step_cap=4)
            runner.train_dagger(config,env.base,env)
            for image in captured:
                torch.testing.assert_close(image,torch.full_like(image,80/255))
            for image,labels in stored:
                self.assertTrue((image==100).all())
                torch.testing.assert_close(labels,torch.full_like(labels,.2))
            path=Path(folder)/'recurrent/student_model_final.pth'
            metadata=torch.load(path,weights_only=True)['metadata']
            self.assertTrue(metadata['visual_augmentation']['enabled'])
            self.assertIn('rollout_rng',metadata['visual_augmentation'])
            config.update(train_or_eval='load_and_train_dagger_new',load_student_path=str(path),total_timesteps=8)
            runner.train_dagger(config,env.base,env)
            captured.clear()
            row=runner.evaluate_checkpoint(config,env.base,env,path,'cpu')
            self.assertEqual(row['episodes_finished'],2)
            self.assertFalse(row['eval_visual_augmentation'])
            self.assertTrue(captured)
            for image in captured:
                torch.testing.assert_close(image,torch.full_like(image,100/255))

            config.update(eval_visual_augmentation=True,eval_augmentation_seed=41)
            for video in (False, True):
                captured.clear()
                frames = []
                writer = SimpleNamespace(isOpened=lambda:True,write=frames.append,release=lambda:None)
                with patch.object(aug,'CLEAN_PROBABILITY',1.), patch('cv2.VideoWriter',return_value=writer):
                    row=runner.evaluate_checkpoint(config,env.base,env,path,'cpu',video=video)
                self.assertTrue(row['eval_visual_augmentation'])
                self.assertEqual(row['eval_augmentation_seed'],41)
                self.assertTrue(captured)
                for image in captured:
                    torch.testing.assert_close(image,torch.full_like(image,80/255))
                if video:
                    self.assertEqual(len(frames),len(captured))
                    for frame in frames:
                        np.testing.assert_array_equal(frame,np.full((32,32,3),80,np.uint8))

    def test_invalid_inputs_and_ranges_fail_clearly(self):
        parameters=aug.VisualAugmentation().sample(2)
        with self.assertRaises(ValueError):
            aug.augment_visual_input(torch.zeros(2,3,8,8,dtype=torch.uint8),parameters)
        with self.assertRaises(ValueError):
            aug.augment_visual_input(torch.zeros(3,3,8,8),parameters)
        with patch.object(aug,'GAMMA_RANGE',(-1.,1.)),self.assertRaises(ValueError):
            aug.VisualAugmentation()


if __name__ == '__main__':
    unittest.main()

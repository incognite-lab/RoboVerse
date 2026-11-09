"""Persistent stage evaluator in a separate process (Nyx has global singletons)."""
from __future__ import annotations

from copy import deepcopy
import io
import multiprocessing as mp
import traceback

import torch


def _serve(connection, config, metadata):
    env = None
    try:
        # Genesis must initialize before any CUDA model allocation. Spawn also
        # isolates Nyx's native global renderer from the training renderer.
        from .runner import build_stage_evaluation_env, validate_action_layout
        from .inputs import StudentInputs
        from .student_net import RecurrentVisionStudent
        from .evaluation import evaluate_policy, evaluation_resets
        try:
            from ..separate_ppo_trainer import load_policy_router
        except ImportError:
            from separate_ppo_trainer import load_policy_router
        torch.set_num_threads(int(config.get('dagger_torch_threads',4)))
        base, env = build_stage_evaluation_env(config)
        validate_action_layout(env)
        if list(env.action_names) != metadata['action_names']:
            raise ValueError('Stage evaluation action layout differs from training')
        if list(env.robot_joint_names) != metadata['joint_names']:
            raise ValueError('Stage evaluation joint layout differs from training')
        device = env.torch_device
        student = RecurrentVisionStudent(**metadata['model_kwargs']).to(device)
        adapter = StudentInputs(base,env,device,metadata['camera_name'],metadata['joint_names'],
                                image_size=metadata['image_shape'][-2:])
        teacher, _ = load_policy_router(env,config.get('expert_model_path') or config['load_model_path'],
                                       device=str(device))
        while True:
            request = connection.recv()
            if request is None:
                break
            try:
                if request['weights'] is not None:
                    student.load_state_dict(torch.load(io.BytesIO(request['weights']),map_location=device,
                                                       weights_only=True))
                with evaluation_resets(env,request['start_stage'],request['stop_stage']):
                    row = evaluate_policy(student,adapter,env,request['episodes'],request['max_steps'],
                                          teacher=teacher if request['expert'] else None)
                connection.send(dict(row=row))
            except RuntimeError as error:
                if 'no snapshot is available' not in str(error):
                    raise
                connection.send(dict(missing_snapshot=str(error)))
    except BaseException:
        try:
            connection.send(dict(error=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class StageProbeProcess:
    def __init__(self, config, metadata):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve,args=(child,deepcopy(config),deepcopy(metadata)),
                                       name='dagger-stage-evaluation',daemon=True)
        self.process.start()
        child.close()

    def evaluate(self, student, *, start_stage, stop_stage, episodes, max_steps, expert):
        weights = None
        if not expert:
            stream = io.BytesIO()
            torch.save({key:value.detach().cpu() for key,value in student.state_dict().items()},stream)
            weights = stream.getvalue()
        self.connection.send(dict(weights=weights,start_stage=start_stage,stop_stage=stop_stage,
                                  episodes=episodes,max_steps=max_steps,expert=expert))
        while not self.connection.poll(1):
            if not self.process.is_alive():
                raise RuntimeError(f'Stage evaluation process exited with code {self.process.exitcode}')
        try:
            result = self.connection.recv()
        except EOFError as error:
            raise RuntimeError('Stage evaluation process closed its connection') from error
        if 'error' in result:
            raise RuntimeError('Stage evaluation process failed:\n'+result['error'])
        if 'missing_snapshot' in result:
            raise RuntimeError(result['missing_snapshot'])
        return result['row']

    def close(self):
        try:
            if self.process.is_alive():
                self.connection.send(None)
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            self.connection.close()

"""Original native benchmark with durable before-reset physical measurements."""
from dataclasses import dataclass, replace
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

from supermimic.envs import make_env_cfg
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import run_benchmark, summarize
from supermimic.eval.physical_capture import ControlCapture


class PhysicalRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.writer = cfg.params['writer']
        self.io = StateIO(env)
        self.previous = {}
        self.seen = set()
        self.episodes = [0] * env.num_envs
        self.steps = [0] * env.num_envs

    @property
    def command(self):
        return self._env.command_manager.get_term('motion')

    def record_post_reset(self, env_ids):
        states = self.io.read().cpu().numpy()
        frames = self.command.frame.cpu().numpy()
        clips = self.command.library.frame_clip[self.command.frame].cpu().numpy()
        for environment in env_ids.tolist():
            self.episodes[environment] += 1
            self.previous.pop(environment, None)
            clip = int(clips[environment])
            if clip in self.seen:
                continue
            self.seen.add(clip)
            self.steps[environment] = 0
            self.previous[environment] = (states[environment].copy(), int(frames[environment]), clip)
            self.writer.start(environment=environment, episode=self.episodes[environment],
                clip=clip, frame=int(frames[environment]), state=states[environment])

    def _record(self, env_ids, terminal):
        env = self._env
        states = self.io.read().cpu().numpy()
        frames = self.command.frame.cpu().numpy()
        action = env.action_manager.action.cpu().numpy()
        executed = env.action_manager.get_term('joint_pos').executed_raw.cpu().numpy()
        reward = env.reward_buf.cpu().numpy()
        terminated = env.reset_terminated.cpu().numpy()
        timeout = env.reset_time_outs.cpu().numpy()
        height_term = env.termination_manager.get_term('root_height').cpu().numpy()
        tilt_term = env.termination_manager.get_term('root_tilt').cpu().numpy()
        for environment in env_ids.tolist():
            if environment not in self.previous:
                continue
            before, frame, clip = self.previous[environment]
            step = self.steps[environment]
            self.writer.append(environment=environment, episode=self.episodes[environment],
                clip=clip, control=step, source_frame=frame, post_command_frame=int(frames[environment]),
                pre_state=before, post_state=states[environment], action=action[environment],
                executed_action=executed[environment], reward=reward[environment],
                physical_terminated=bool(terminated[environment]), timeout=bool(timeout[environment]),
                root_height_term=bool(height_term[environment]), root_tilt_term=bool(tilt_term[environment]),
                captured_before_reset=terminal, physical_time_s=(step+1)*env.step_dt)
            self.steps[environment] += 1
            self.previous[environment] = (states[environment].copy(), int(frames[environment]), clip)

    def record_pre_reset(self, env_ids):
        self._record(env_ids, True)
        for environment in env_ids.tolist():
            self.previous.pop(environment, None)

    def record_post_step(self):
        self._record((~self._env.reset_buf).nonzero().squeeze(-1), False)


@dataclass(kw_only=True)
class PhysicalRecorderCfg(RecorderTermCfg):
    func: type = PhysicalRecorder


def run_recorded_benchmark(policy, files, *, output, source_revision, checkpoint_artifact_id,
                           device='cuda:0', num_envs=1024, future_frames=10, anchor_obs=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    writer = ControlCapture(output/'physical')
    metadata = dict(source_revision=source_revision, checkpoint_artifact_id=checkpoint_artifact_id,
        reference_files=list(map(str, files)), control_dt_s=.02, physics_dt_s=.005,
        state_representation='MuJoCo pelvis qpos36 and qvel35; root angular velocity in body frame',
        capture='authoritative qpos/qvel before auto-reset; derived SDK body poses not used',
        capacity_evidence_available=False, contact_force_available=False,
        geometry='original donor pelvis model', model_artifact_id='supermimic-ba610534-original-pelvis-g1',
        native_parent_revision='ba6105349069bf040f1fdad60873d13138d505a2', compact_library=True, library_fp16=False,
        native_return_denominator='0.1 times actual executed steps',
        full_request_return_denominator='0.1 times declared native request controls')
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    inventory = []
    def environment_factory(args, device):
        args = replace(args, compact_library=True, extra_command_kwargs={'library_fp16': False})
        config = make_env_cfg(args)
        config.recorders['physical'] = PhysicalRecorderCfg(params={'writer': writer})
        env = ManagerBasedRlEnv(config, device=device)
        import mujoco
        model = env.sim.mj_model
        mujoco.mj_saveModel(model, str(output/'native-model.mjb'), None)
        metadata['body_names'] = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
        metadata['joint_names'] = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
        metadata['mujoco_version'] = mujoco.__version__
        (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
        library = env.command_manager.get_term('motion').library
        inventory.extend(dict(index=i, name=c.name, frames=c.length, start=c.start)
                         for i, c in enumerate(library.clips))
        (output/'reference-inventory.json').write_text(json.dumps(inventory, indent=2))
        return env
    try:
        results = run_benchmark(policy, files, device=device, num_envs=num_envs,
            future_frames=future_frames, anchor_obs=anchor_obs, environment_factory=environment_factory)
        payload = dict(native_summary=summarize(results), native_clips=[vars(row) for row in results])
        payload['full_request_readings'] = [dict(name=row.name, full_request_controls=row.frames,
            executed_controls=row.steps, missing_suffix_controls=max(0, row.frames-row.steps),
            full_request_return=row.norm_return*row.steps/row.frames,
            native_clip_completed=not row.failed and row.steps == row.frames,
            denominator_definition='native clip frame count times 0.1; distinct from executed-step native return')
            for row in results]
        (output/'native-results.json').write_text(json.dumps(payload, indent=2))
        return results
    finally:
        writer.close()

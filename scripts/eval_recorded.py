"""Record the original native benchmark; use --zero-policy only for timing probes."""
from pathlib import Path

import torch
import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.eval.recorded_benchmark import run_recorded_benchmark
from supermimic.eval.physical_capture import selection_files


def main(out: Path, source_revision: str, checkpoint_artifact_id: str,
         npz_dir: Path | None = None, packed_library: Path | None = None,
         selection: Path | None = None, reference_artifact_id: str | None = None,
         ckpt: Path | None = None, clips: list[str] | None = None,
         num_envs: int = 1024, device: str = 'cuda:0', zero_policy: bool = False):
    if sum(value is not None for value in (npz_dir, packed_library, selection)) != 1:
        raise ValueError('select one original NPZ directory, packed library or frozen selection')
    selected = None
    if selection is not None:
        if clips is not None:
            raise ValueError('a frozen selection cannot be filtered after declaration')
        files, selected = selection_files(selection)
        if reference_artifact_id is not None and reference_artifact_id != selected['reference_artifact_id']:
            raise ValueError('selected reference artifact differs')
        reference_artifact_id = selected['reference_artifact_id']
    else:
        if packed_library is not None and clips is not None:
            raise ValueError('packed library evaluation consumes its complete declared selection')
        files = [str(packed_library)] if packed_library is not None else [str(path) for path in discover_npz(npz_dir, names=clips)]
        if clips is not None and {Path(path).stem for path in files} != set(clips):
            raise ValueError('some explicitly selected original NPZ clips are missing')
    if zero_policy:
        if ckpt is not None:
            raise ValueError('zero-policy timing probes require no checkpoint')
        def policy(observation, goal):
            return torch.zeros((len(observation), 29), device=observation.device)
        config = dict(env=dict(future_frames=10, anchor_obs=False))
    else:
        if ckpt is None:
            raise ValueError('a fixed checkpoint is required')
        policy, config = load_policy_from_checkpoint(ckpt, device)
    run_recorded_benchmark(policy, files, output=out, source_revision=source_revision,
        checkpoint_artifact_id=checkpoint_artifact_id, device=device, num_envs=num_envs,
        future_frames=config['env']['future_frames'], anchor_obs=config['env'].get('anchor_obs', False),
        checkpoint_path=str(ckpt.resolve()) if ckpt is not None else None,
        reference_artifact_id=reference_artifact_id, selection=selected)


if __name__ == '__main__':
    tyro.cli(main)

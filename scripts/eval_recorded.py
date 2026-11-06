"""Record the original native benchmark; use --zero-policy only for timing probes."""
from pathlib import Path

import torch
import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.eval.recorded_benchmark import run_recorded_benchmark


def main(out: Path, source_revision: str, checkpoint_artifact_id: str,
         npz_dir: Path | None = None, packed_library: Path | None = None,
         ckpt: Path | None = None, clips: list[str] | None = None,
         num_envs: int = 1024, device: str = 'cuda:0', zero_policy: bool = False):
    if (npz_dir is None) == (packed_library is None):
        raise ValueError('select exactly one original NPZ directory or original packed library')
    if packed_library is not None and clips is not None:
        raise ValueError('packed library evaluation consumes its complete declared selection')
    files = [str(packed_library)] if packed_library is not None else [str(path) for path in discover_npz(npz_dir, names=clips)]
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
        checkpoint_path=str(ckpt.resolve()) if ckpt is not None else None)


if __name__ == '__main__':
    tyro.cli(main)

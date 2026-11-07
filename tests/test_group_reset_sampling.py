"""CPU comparisons of reset distributions and collection counters, without a simulator."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MDP = Path(__file__).resolve().parents[1] / 'src/supermimic/envs/mdp'
spec = importlib.util.spec_from_file_location('group_reset_sampling', MDP / 'group_reset_sampling.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GroupResetSampling = module.GroupResetSampling


def command_method(name):
    # Import only the numerical method; the environment package imports mjlab.
    cls = next(n for n in ast.parse((MDP / 'commands.py').read_text()).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'torch': torch, 'math': math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(MDP / 'commands.py'), 'exec'), scope)
    return scope[name]


def cfg():
    return SimpleNamespace(sampling_mode='return', uniform_ratio=.5, hard_ratio=.4,
                           memory_ratio=.1, softmax_temp=25., g_max=10., memory_alpha=.01)


def library():
    # Four clips with unequal lengths and unequal bin counts (1, 1, 2, 3).
    starts = torch.tensor([0, 5, 10, 20])
    lens = torch.tensor([5, 5, 10, 15])
    frame_clip = torch.repeat_interleave(torch.arange(4), lens)
    bin_start = torch.arange(0, 35, 5)
    lib = SimpleNamespace(num_clips=4, num_bins=7, num_frames=35, clip_start=starts,
                          frame_clip=frame_clip, bin_start=bin_start, bin_end=bin_start+5)
    lib.last_frame_of = lambda f: (starts+lens-1)[frame_clip[f]]
    return lib


def sampler(lib):
    return GroupResetSampling(lib.num_clips, lib.frame_clip[lib.bin_start],
                              {'BONES': [0, 1, 2], 'ENRAI': [3]},
                              {'ENRAI': .8, 'BONES': .2})


def command(groups=None):
    lib = library()
    return SimpleNamespace(library=lib, cfg=cfg(), device='cpu', reset_groups=groups,
                           bin_return=torch.arange(7).float(), bin_memory=torch.ones(7)/7,
                           metrics={'sampling_entropy': torch.zeros(3)})


def test_group_mass_and_independent_adaptation():
    lib = library()
    groups = sampler(lib)
    returns = torch.arange(7).float()
    memory = torch.ones(7)/7
    for _ in range(10):
        p, memory = groups.distribution(returns, memory, cfg())
        torch.testing.assert_close(p[:4].sum(), torch.tensor(.2))
        torch.testing.assert_close(p[4:].sum(), torch.tensor(.8))
        torch.testing.assert_close(p.sum(), torch.tensor(1.))
        torch.testing.assert_close(memory[:4].sum(), torch.tensor(1.))
        torch.testing.assert_close(memory[4:].sum(), torch.tensor(1.))
    changed = returns.clone()
    changed[4:] -= 10000
    changed_p, _ = groups.distribution(changed, memory, cfg())
    unchanged_p, _ = groups.distribution(returns, memory, cfg())
    assert torch.equal(changed_p[:4], unchanged_p[:4])


def test_within_group_distribution_matches_return_formula():
    groups = sampler(library())
    c = command(groups)
    p, memory = groups.distribution(c.bin_return, c.bin_memory, c.cfg)
    for indices, probability in zip(groups.bin_indices, groups.probabilities):
        hard = torch.softmax(-c.cfg.softmax_temp*c.bin_return[indices]/c.cfg.g_max, 0)
        adaptive = (1-c.cfg.memory_alpha)*torch.ones(len(indices))/len(indices)+c.cfg.memory_alpha*hard
        expected = .5/len(indices)+.4*hard+.1*adaptive/adaptive.sum()
        torch.testing.assert_close(p[indices], probability*expected/expected.sum())
        torch.testing.assert_close(memory[indices], adaptive)


def test_default_seeded_frame_and_memory_parity():
    c = command()
    torch.manual_seed(42)
    frames = command_method('_sample_frames')(c, 1000)
    torch.manual_seed(42)
    baseline = command()
    hard = torch.softmax(-baseline.cfg.softmax_temp*baseline.bin_return/baseline.cfg.g_max, 0)
    memory = (1-baseline.cfg.memory_alpha)*baseline.bin_memory+baseline.cfg.memory_alpha*hard
    p = .5/7+.4*hard+.1*memory/memory.sum()
    p = p/p.sum()
    b = torch.multinomial(p, 1000, replacement=True)
    f = baseline.library.bin_start[b]+(torch.rand(1000)*5).long()
    f = torch.minimum(f, baseline.library.last_frame_of(f)-2).clamp_min(baseline.library.clip_start[baseline.library.frame_clip[f]])
    assert torch.equal(frames, f)
    assert torch.equal(c.bin_memory, memory)


def test_group_frame_membership_and_cumulative_counts():
    lib = library()
    groups = sampler(lib)
    c = command(groups)
    torch.manual_seed(42)
    frames = command_method('_sample_frames')(c, 10000)
    clips = lib.frame_clip[frames]
    groups.record_resets(clips)
    counts = torch.bincount(groups.clip_group[clips], minlength=2)
    assert 7800 < counts[1] < 8200
    assert ((frames >= lib.clip_start[clips]) & (frames <= lib.last_frame_of(frames)-2)).all()
    groups.record_controls(torch.tensor([-1, 0, 0, 1]))
    groups.record_controls(torch.tensor([0, 1]))
    assert groups.statistics() == {
        'BONES': {'probability': .2, 'reset_count': counts[0].item(), 'collected_controls': 3},
        'ENRAI': {'probability': .8, 'reset_count': counts[1].item(), 'collected_controls': 2},
    }


def test_reward_collection_counts_old_group_on_terminal_control():
    groups = sampler(library())
    lib = library()
    lib.bin_of_frame = lambda f: f//5
    c = SimpleNamespace(_env=SimpleNamespace(reward_buf=torch.ones(3)), num_envs=3,
                        device='cpu', reset_groups=groups, _active_reset_group=torch.tensor([-1, 0, 1]),
                        ep_len=torch.zeros(3, dtype=torch.long), max_ep_steps=4,
                        ep_rewards=torch.zeros(3, 4), ep_bins=torch.zeros(3, 4, dtype=torch.long),
                        library=lib, frame=torch.tensor([0, 5, 20]))
    log = command_method('_log_step_rewards')
    log(c, torch.tensor([True, False, True]))  # startup excluded; terminating ENRAI included
    c._active_reset_group[2] = 0  # that environment resets into BONES
    log(c, torch.tensor([False, True, False]))  # other environments' same control
    log(c, torch.tensor([False, True, True]))  # subsequent control
    assert groups.collected_controls.tolist() == [3, 1]


@pytest.mark.parametrize('groups,probabilities', [
    ({'x': [0, 1]}, {'x': 1.}),
    ({'x': [0, 1, 2], 'y': [2, 3]}, {'x': .5, 'y': .5}),
    ({'x': [0, 1, 2, 4]}, {'x': 1.}),
    ({'x': [0, 1, 2, 3]}, {'x': float('nan')}),
    ({'x': [0, 1, 2, 3]}, {'x': .5}),
    ({'x': [0, 1, 2, 3], 'y': []}, {'x': .5, 'y': .5}),
    ({'x': [0, 1, 2, 3]}, {'y': 1.}),
])
def test_invalid_partition_or_probability_refused(groups, probabilities):
    lib = library()
    with pytest.raises(ValueError):
        GroupResetSampling(4, lib.frame_clip[lib.bin_start], groups, probabilities)


def test_zero_probability_group_gets_no_resets():
    lib = library()
    groups = GroupResetSampling(4, lib.frame_clip[lib.bin_start], {'x': [0, 1, 2], 'y': [3]}, {'x': 1., 'y': 0.})
    c = command(groups)
    frames = command_method('_sample_frames')(c, 200)
    assert (lib.frame_clip[frames] != 3).all()

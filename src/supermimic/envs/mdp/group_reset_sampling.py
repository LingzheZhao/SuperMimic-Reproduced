"""Explicit clip-group probabilities and cumulative training exposure."""

from __future__ import annotations

import math

import torch


class GroupResetSampling:
    """Partition clips and preserve return sampling separately inside each group."""

    def __init__(self, num_clips, bin_clip, groups, probabilities):
        if not groups or set(groups) != set(probabilities):
            raise ValueError("Reset groups and probabilities must have identical nonempty names")
        values = list(probabilities.values())
        if any(not math.isfinite(p) or p < 0 for p in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-7):
            raise ValueError("Reset group probabilities must be finite, nonnegative and sum to one")
        self.names = tuple(groups)
        self.probabilities = tuple(float(probabilities[name]) for name in self.names)
        device = bin_clip.device
        self.clip_group = torch.full((num_clips,), -1, dtype=torch.long, device=device)
        seen = set()
        for i, name in enumerate(self.names):
            clips = groups[name]
            if not clips:
                raise ValueError(f"Reset group {name!r} has no clips")
            for clip in clips:
                if type(clip) is not int or not 0 <= clip < num_clips or clip in seen:
                    raise ValueError("Reset groups must partition all clip indices without duplicates")
                seen.add(clip)
            self.clip_group[torch.tensor(clips, device=device)] = i
        if len(seen) != num_clips:
            raise ValueError("Reset groups must include every library clip")
        self.bin_group = self.clip_group[bin_clip]
        self.bin_indices = tuple((self.bin_group == i).nonzero().flatten() for i in range(len(groups)))
        if any(indices.numel() == 0 for indices in self.bin_indices):
            raise ValueError("Every reset group must contain sampling bins")
        self.reset_count = torch.zeros(len(groups), dtype=torch.int64, device=device)
        self.collected_controls = torch.zeros_like(self.reset_count)

    def distribution(self, bin_return, bin_memory, cfg):
        """Return joint bin mass and the within-group adaptive memory."""
        p = torch.empty_like(bin_return)
        memory = torch.empty_like(bin_memory)
        for indices, probability in zip(self.bin_indices, self.probabilities):
            hard = torch.softmax(-cfg.softmax_temp * bin_return[indices] / cfg.g_max, dim=0)
            previous = bin_memory[indices]
            previous = previous / previous.sum()
            retained = (1 - cfg.memory_alpha) * previous + cfg.memory_alpha * hard
            local = cfg.uniform_ratio / indices.numel() + cfg.hard_ratio * hard + cfg.memory_ratio * retained / retained.sum()
            p[indices] = probability * local / local.sum()
            memory[indices] = retained
        return p, memory

    def record_resets(self, clips):
        self.reset_count += torch.bincount(self.clip_group[clips], minlength=len(self.names))

    def record_controls(self, group_ids):
        group_ids = group_ids[group_ids >= 0]
        self.collected_controls += torch.bincount(group_ids, minlength=len(self.names))

    def statistics(self):
        resets = self.reset_count.tolist()
        controls = self.collected_controls.tolist()
        return {
            name: {"probability": probability, "reset_count": resets[i], "collected_controls": controls[i]}
            for i, (name, probability) in enumerate(zip(self.names, self.probabilities))
        }

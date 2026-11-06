"""Compressed measured-control chunks with explicit episode and clip identities."""
from pathlib import Path

import numpy as np


class ControlCapture:
    def __init__(self, output, chunk_rows=4096):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.chunk_rows = chunk_rows
        self.rows = []
        self.initial = []
        self.chunk = 0

    def __deepcopy__(self, memo):
        # Recorder configuration copies share this one owned output stream.
        memo[id(self)] = self
        return self

    def start(self, *, environment, episode, clip, frame, state):
        self.initial.append(dict(environment=environment, episode=episode, clip=clip,
                                 frame=frame, state=np.asarray(state).copy()))

    def append(self, **row):
        self.rows.append({key: np.asarray(value).copy() for key, value in row.items()})
        if len(self.rows) >= self.chunk_rows:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        path = self.output / f'controls-{self.chunk:06d}.npz'
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        self.rows.clear()
        self.chunk += 1

    def close(self):
        self.flush()
        if self.initial:
            with (self.output/'initial-states.npz').open('xb') as stream:
                np.savez_compressed(stream, **{key: np.stack([row[key] for row in self.initial])
                                              for key in self.initial[0]})
            self.initial.clear()


def selection_files(path):
    """Consume a frozen original-NPZ selection without directory filtering."""
    import yaml
    selected = yaml.safe_load(Path(path).read_text())
    sources = selected['sources']
    if (len(sources) != selected['selected_count'] or not sources
            or [row['evaluation_index'] for row in sources] != list(range(len(sources)))
            or len({row['artifact_id'] for row in sources}) != len(sources)):
        raise ValueError('selection requires complete consecutive original clip identities')
    files = []
    for row in sources:
        original = Path(row['npz'])
        if (not original.is_file() or original.stem != row['name']
                or row['native_request_controls'] != row['poses']
                or row['strict_reference_controls'] != row['poses']-1):
            raise ValueError('original NPZ or native/genuine reference denominator differs')
        files.append(str(original))
    return files, selected

import copy

import numpy as np

from supermimic.eval.physical_capture import ControlCapture


def test_chunked_controls_keep_terminal_separate_from_next_initial(tmp_path):
    writer = ControlCapture(tmp_path/'capture', chunk_rows=2)
    assert copy.deepcopy(writer) is writer
    writer.start(environment=0, episode=1, clip=3, frame=10, state=np.zeros(71))
    writer.append(environment=0, episode=1, clip=3, control=0,
                  pre_state=np.zeros(71), post_state=np.ones(71), captured_before_reset=True)
    writer.start(environment=0, episode=2, clip=5, frame=20, state=np.full(71, 8))
    writer.append(environment=0, episode=2, clip=5, control=0,
                  pre_state=np.full(71, 8), post_state=np.full(71, 9), captured_before_reset=False)
    writer.close()
    with np.load(tmp_path/'capture/controls-000000.npz') as rows:
        np.testing.assert_array_equal(rows['episode'], [1, 2])
        np.testing.assert_array_equal(rows['clip'], [3, 5])
        np.testing.assert_array_equal(rows['post_state'][:, 0], [1, 9])
    with np.load(tmp_path/'capture/initial-states.npz') as rows:
        np.testing.assert_array_equal(rows['state'][:, 0], [0, 8])


def test_frozen_selection_keeps_all_originals_and_both_denominators(tmp_path):
    import yaml
    import pytest
    from supermimic.eval.physical_capture import selection_files
    original = tmp_path/'source.npz';original.write_bytes(b'')
    payload = dict(artifact_id='frozen', reference_artifact_id='original', selected_count=1,
        sources=[dict(evaluation_index=0, name='source', artifact_id='original/source', npz=str(original),
                      poses=8, native_request_controls=8, strict_reference_controls=7)])
    manifest = tmp_path/'selection.yaml';manifest.write_text(yaml.safe_dump(payload))
    files, selected = selection_files(manifest)
    assert files == [str(original)] and selected['selected_count'] == 1
    payload['sources'][0]['strict_reference_controls'] = 8
    manifest.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match='denominator'):
        selection_files(manifest)

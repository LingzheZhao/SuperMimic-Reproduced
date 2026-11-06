import numpy as np

from supermimic.eval.physical_capture import ControlCapture


def test_chunked_controls_keep_terminal_separate_from_next_initial(tmp_path):
    writer = ControlCapture(tmp_path/'capture', chunk_rows=2)
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

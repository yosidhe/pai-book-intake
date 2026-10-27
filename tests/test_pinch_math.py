import numpy as np

from pai_book_intake.geometry import quat_to_mat


def _rand_rot(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return quat_to_mat(q)


def test_mats_to_quat_roundtrip_including_near_180_degrees():
    from pai_book_intake.pinch import mats_to_quat

    rng = np.random.default_rng(0)
    m = _rand_rot(rng, 200)
    # 180° 近い回転（本を立てる姿勢で崩れた形）: 手の x = 下、y = 水平、z = x × y
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([0.0, 1.0, 0.0])
    m180 = np.stack([x, y, np.cross(x, y)], axis=-1)[None]
    flip = np.diag([1.0, -1.0, -1.0])[None]  # x 軸まわり 180°
    m = np.concatenate([m, m180, flip])
    assert np.allclose(quat_to_mat(mats_to_quat(m)), m, atol=1e-9)

import numpy as np

# This module focuses on quaternion transformations in 3D space

def normalize_quaternion_xyzw(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")

    return q / norm


def quaternion_xyzw_to_matrix(q):
    qx, qy, qz, qw = normalize_quaternion_xyzw(q)

    return np.array([
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qx * qw),
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx**2 + qy**2),
        ],
    ], dtype=np.float64)


def make_transform_sensor_to_base(
    translation_xyz,
    quaternion_xyzw,
):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_xyzw_to_matrix(
        quaternion_xyzw
    )
    T[:3, 3] = np.asarray(
        translation_xyz,
        dtype=np.float64,
    )
    return T


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv
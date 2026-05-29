import numpy as np
from transform_utils.math_utils import angle_boxminus


def transform_annos_baselink_to_baselink(annos_bl: dict, T: np.ndarray) -> dict:
    """Apply SE(3) T to baselink annos (location + heading_angles). Output remains baselink."""
    if annos_bl is None:
        return None
    out = {"name": [], "dimensions": [], "location": [], "heading_angles": [], "track_id": []}
    if ("name" in annos_bl) and (len(annos_bl["name"]) == 0):
        return out

    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    yaw_rel = float(np.arctan2(R[1, 0], R[0, 0]))

    names = annos_bl.get("name", [])
    dims = annos_bl.get("dimensions", [])
    locs = annos_bl.get("location", [])
    heads = annos_bl.get("heading_angles", [])
    tids = annos_bl.get("track_id", [])

    n = len(names)
    for k in range(n):
        c_prev = np.asarray(locs[k], dtype=np.float64)
        c_curr = R @ c_prev + t

        out["name"].append(names[k])
        out["dimensions"].append(list(dims[k]))
        out["location"].append([float(c_curr[0]), float(c_curr[1]), float(c_curr[2])])
        out["heading_angles"].append(float(heads[k]) + yaw_rel)
        out["track_id"].append(tids[k] if k < len(tids) else "")

    return out

def transform_annos_baselink_to_sensor(annos_bl: dict, T_sensor_from_baselink: np.ndarray, yaw_rad: float) -> dict:
    """
    Convert baselink annos into sensor coords (location + heading_angles).
    """
    if annos_bl is None:
        return None
    out = {"name": [], "dimensions": [], "location": [], "heading_angles": [], "track_id": []}
    if ("name" in annos_bl) and (len(annos_bl["name"]) == 0):
        return out

    T = np.asarray(T_sensor_from_baselink, dtype=np.float64).reshape(4, 4)

    names = annos_bl.get("name", [])
    dims = annos_bl.get("dimensions", [])
    locs = annos_bl.get("location", [])
    heads = annos_bl.get("heading_angles", [])
    tids = annos_bl.get("track_id", [])

    n = len(names)
    for k in range(n):
        xb, yb, zb = locs[k]
        xs, ys, zs, _ = T @ np.array([xb, yb, zb, 1.0], dtype=np.float64)

        hb = float(heads[k])
        hs = float(angle_boxminus(hb, yaw_rad))

        out["name"].append(names[k])
        out["dimensions"].append(list(dims[k]))
        out["location"].append([float(xs), float(ys), float(zs)])
        out["heading_angles"].append(hs)
        out["track_id"].append(tids[k] if k < len(tids) else "")

    return out
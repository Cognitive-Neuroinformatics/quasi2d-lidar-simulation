import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Any


def _to_builtin(x: Any):
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): _to_builtin(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_builtin(v) for v in x]

    except Exception:
        pass
    return x

def is_empty_annos_with_track(ann: dict) -> bool:
    if ann is None:
        return True
    # treat as empty if it has no objects
    return len(ann.get("name", [])) == 0

def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def transform_annos_with_track_se3(T_rel: np.ndarray, ann: dict) -> dict:
        """
        T_rel: (4,4) SE3 mapping prev -> current, in the SAME coord frame as ann['location']
        ann: dict with keys: name, dimensions, location, heading_angles, track_id
        Returns a new dict with transformed location + heading_angles.
        If ann is empty, returns it unchanged.
        """
        if ann is None:
            return {"name": [], "dimensions": [], "location": [], "heading_angles": [], "track_id": []}

        # empty -> return as-is
        if len(ann.get("name", [])) == 0:
            return ann

        R = T_rel[:3, :3]
        t = T_rel[:3, 3]

        locs = np.asarray(ann["location"], dtype=np.float64)         # (N,3)
        dims = ann["dimensions"]                                     # keep original
        yaws = np.asarray(ann["heading_angles"], dtype=np.float64)   # (N,)
        names = ann["name"]
        tids = ann["track_id"]

        if locs.ndim != 2 or locs.shape[1] != 3:
            raise ValueError(f"ann['location'] must be (N,3), got {locs.shape}")
        if yaws.ndim != 1 or yaws.shape[0] != locs.shape[0]:
            raise ValueError(f"ann['heading_angles'] must be (N,), got {yaws.shape} vs N={locs.shape[0]}")

        locs_new = locs @ R.T + t

        delta_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        yaws_new = wrap_to_pi(yaws + delta_yaw)

        return {
            "name": list(names),
            "dimensions": list(dims),
            "location": locs_new.astype(np.float32).tolist(),
            "heading_angles": yaws_new.astype(np.float32).tolist(),
            "track_id": list(tids),
        }

def annos_to_records(annos_with_track: Dict) -> list[dict]:
    names = annos_with_track.get("name", [])
    dims = annos_with_track.get("dimensions", [])
    locs = annos_with_track.get("location", [])
    yaws = annos_with_track.get("heading_angles", [])
    tids = annos_with_track.get("track_id", [])

    n = min(len(names), len(dims), len(locs), len(yaws), len(tids))
    out = []
    for k in range(n):
        l, w, h = dims[k]
        x, y, z = locs[k]
        out.append(
            {
                "name": str(names[k]).strip(),
                "track_id": _to_builtin(tids[k]),
                "location": [float(_to_builtin(x)), float(_to_builtin(y)), float(_to_builtin(z))],
                "dimensions": [float(_to_builtin(l)), float(_to_builtin(w)), float(_to_builtin(h))],
                "heading": float(_to_builtin(yaws[k])),
            }
        )
    return out



class GTTrackJSONWriter:
    def __init__(self, out_dir_json: Path):
        self.root = Path(out_dir_json)
        self.root.mkdir(parents=True, exist_ok=True)

    def _json_path(self, sequence_name: str, sensor_i: int, frame_index: int) -> Path:
        out_dir = self.root 
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{sequence_name}_{sensor_i}_{frame_index:03d}.json"

    def update_multiframe(
        self,
        sequence_name: str,
        sensor_i: int,
        frame_index: int,
        annos_list_in_cur_with_track: List[Optional[Dict]],
    ):
        """
        annos_list_in_cur_with_track: [t, t-1, t-2, t-3] but each is already transformed into CURRENT frame.
        """
        frames = []
        for k, annos in enumerate(annos_list_in_cur_with_track):
            if annos is None:
                continue
            frames.append({"frame_id": int(k), "annos": annos_to_records(annos)})

        payload = {
            "sequence_name": sequence_name,
            "sensor": sensor_i,
            "frame_index": int(frame_index),
            "num_frames": int(len([a for a in annos_list_in_cur_with_track if a is not None])),
            "frames": frames,
        }
        p = self._json_path(sequence_name, sensor_i, frame_index)
        with p.open("w") as f:
            json.dump(payload, f, indent=2)
import os
import numpy as np
from transform_utils.math_utils import angle_boxminus

def write_openpcdet_label_file_multiframe(
        transformer_class,
        i,
        sequence_name,
        filtered_bounding_boxes,
        frame_index,
        out_folder,
        bev_padding=None,
        metadata=None,
        transformation_mode=None,
        return_with_track=False,
        label_subdirs=("labels",),
    ):
        """
        Writes OpenPCDet-style label .txt for one frame.
        writes the SAME label file into multiple sibling directories, for processing later

        """

        fname = f"{sequence_name}_sensor_{i}_{frame_index:03d}.txt"

        # Build output paths and ensure dirs exist
        out_paths = []
        for sub in label_subdirs:
            d = os.path.join(out_folder, sub)
            os.makedirs(d, exist_ok=True)
            out_paths.append(os.path.join(d, fname))

        if transformation_mode == "localsensor":
            yaw_rad = np.deg2rad(metadata["rotation_angle"])
            _, T_sensor_from_baselink = transformer_class.get_sensor_transforms(metadata, bev_padding)
        else:
            yaw_rad = None
            T_sensor_from_baselink = None

        # kept for compatibility (sensor-frame if localsensor)
        annos = {"name": [], "dimensions": [], "location": [], "heading_angles": []}

        # RETURN THIS in BASELINK frame
        annos_with_track_baselink = {
            "name": [],
            "dimensions": [],
            "location": [],         # baselink center
            "heading_angles": [],   # baselink heading
            "track_id": [],
        }

        # Open all files once and write each line to all of them
        files = [open(p, "w") for p in out_paths]
        try:
            for box in filtered_bounding_boxes:
                cat = label_map(box["type"]).strip()
                length, width, height = box["dimensions"]

                # baselink values 
                xb, yb, zb = box["center"]
                heading_b = float(box["heading"])
                tid = str(box.get("track_id", -1))

                # --- write in SENSOR frame if transform is available ---
                if T_sensor_from_baselink is not None:
                    xs, ys, zs, _ = T_sensor_from_baselink @ np.array([xb, yb, zb, 1.0], dtype=np.float64)
                    heading_s = float(angle_boxminus(heading_b, yaw_rad))
                    xw, yw, zw, hw = xs, ys, zs, heading_s
                else:
                    # no localsensor transform requested => write baselink as-is
                    xw, yw, zw, hw = float(xb), float(yb), float(zb), heading_b

                # --- annos (sensor if localsensor) ---
                annos["name"].append(cat)
                annos["dimensions"].append([float(length), float(width), float(height)])
                annos["location"].append([float(xw), float(yw), float(zw)])
                annos["heading_angles"].append(float(hw))

                # --- annos_with_track: BASELINK (for later ego-motion compensation) ---
                annos_with_track_baselink["name"].append(cat)
                annos_with_track_baselink["dimensions"].append([float(length), float(width), float(height)])
                annos_with_track_baselink["location"].append([float(xb), float(yb), float(zb)])
                annos_with_track_baselink["heading_angles"].append(float(heading_b))
                annos_with_track_baselink["track_id"].append(tid)

                line = (
                    f"{xw:.6f} {yw:.6f} {zw:.6f} "
                    f"{float(length):.6f} {float(width):.6f} {float(height):.6f} "
                    f"{float(hw):.6f} {cat}\n"
                )
                for f in files:
                    f.write(line)
        finally:
            for f in files:
                f.close()

        return annos_with_track_baselink if return_with_track else annos

                      


def label_map(label_type):
    if label_type == 1: return 'Vehicle'
    elif label_type == 2: return 'Pedestrian'
    elif label_type == 4: return 'Cyclist'
    return 'Unknown'

import argparse
import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils


def load_frame(tfrecord, frame_idx):
    dataset = tf.data.TFRecordDataset(
        tfrecord,
        compression_type=""
    )

    for i, data in enumerate(dataset):
        if i != frame_idx:
            continue

        frame = open_dataset.Frame()
        frame.ParseFromString(
            bytearray(data.numpy())
        )
        return frame

    raise RuntimeError(
        f"Frame {frame_idx} not found"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tfrecord",
        required=True
    )

    parser.add_argument(
        "--existing-npz",
        required=True
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0
    )

    args = parser.parse_args()

    frame = load_frame(
        args.tfrecord,
        args.frame
    )

    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(
        frame
    )

    existing = np.load(
        args.existing_npz
    )["data"]

    print("=" * 80)
    print("EXISTING PREPROCESSED NPZ")
    print("=" * 80)

    print("shape:", existing.shape)
    print("first rows:")
    print(existing[:5])

    for ri_index in [0, 1]:

        print()
        print("=" * 80)
        print(
            f"WAYMO TOP RETURN {ri_index}"
        )
        print("=" * 80)

        points, cp_points = (
            frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=ri_index,
            )
        )

        # Waymo orders laser calibrations by sensor name.
        # TOP is the first point cloud returned here.
        top_xyz = np.asarray(
            points[0]
        )

        print(
            "XYZ shape:",
            top_xyz.shape
        )

        print(
            "XYZ first rows:"
        )
        print(
            top_xyz[:5]
        )

        # ------------------------------------------------------
        # Extract intensity from TOP range image
        # ------------------------------------------------------

        top_name = (
            open_dataset.LaserName.TOP
        )

        ri = range_images[
            top_name
        ][ri_index]

        ri_tensor = tf.reshape(
            tf.convert_to_tensor(
                ri.data
            ),
            ri.shape.dims
        )

        ri_np = ri_tensor.numpy()

        print(
            "Range image shape:",
            ri_np.shape
        )

        # Channel convention:
        # 0 = range
        # 1 = intensity
        # 2 = elongation
        #
        # Valid pixels are range > 0
        valid_mask = (
            ri_np[..., 0] > 0
        )

        raw_intensity = (
            ri_np[..., 1][
                valid_mask
            ]
        )

        print(
            "valid pixels:",
            valid_mask.sum()
        )

        print(
            "intensity shape:",
            raw_intensity.shape
        )

        candidate = np.concatenate(
            [
                top_xyz,
                raw_intensity[:, None],
            ],
            axis=1,
        )

        print(
            "candidate shape:",
            candidate.shape
        )

        print(
            "candidate first rows:"
        )
        print(
            candidate[:5]
        )

        # ------------------------------------------------------
        # Compare directly if shapes match
        # ------------------------------------------------------

        if candidate.shape == existing.shape:

            diff = np.abs(
                candidate.astype(
                    np.float64
                )
                -
                existing.astype(
                    np.float64
                )
            )

            print()
            print(
                "SHAPE MATCHES EXISTING"
            )

            print(
                "allclose:",
                np.allclose(
                    candidate,
                    existing,
                    atol=1e-6,
                    rtol=0
                )
            )

            print(
                "max abs diff:",
                diff.max()
            )

            print(
                "mean abs diff:",
                diff.mean()
            )

            print(
                "XYZ max diff:",
                diff[:, :3].max()
            )

            print(
                "intensity max diff:",
                diff[:, 3].max()
            )

        else:

            print()
            print(
                "Shape does NOT match existing."
            )


if __name__ == "__main__":
    main()
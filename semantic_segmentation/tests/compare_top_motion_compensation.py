import argparse
import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import range_image_utils


def load_frame(tfrecord, frame_idx):
    dataset = tf.data.TFRecordDataset(
        tfrecord,
        compression_type=""
    )

    for i, data in enumerate(dataset):
        if i == frame_idx:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            return frame

    raise RuntimeError(f"Frame {frame_idx} not found")


def get_top_calibration(frame):
    for c in frame.context.laser_calibrations:
        if c.name == open_dataset.LaserName.TOP:
            return c

    raise RuntimeError("TOP calibration not found")


def extract_top_without_motion_compensation(
    frame,
    range_images,
    ri_index,
):
    """
    Reconstruct TOP LiDAR XYZ directly from the range image,
    intentionally WITHOUT TOP per-pixel motion compensation.

    Output coordinates are in vehicle frame because the TOP extrinsic
    is applied.
    """

    calib = get_top_calibration(frame)

    range_image = range_images[
        open_dataset.LaserName.TOP
    ][ri_index]

    ri = tf.reshape(
        tf.convert_to_tensor(range_image.data),
        range_image.shape.dims
    )

    # ---------------------------------------------------------
    # Beam inclinations
    # ---------------------------------------------------------

    if len(calib.beam_inclinations) > 0:
        beam_inclinations = tf.constant(
            calib.beam_inclinations,
            dtype=tf.float32,
        )
    else:
        beam_inclinations = (
            range_image_utils.compute_inclination(
                tf.constant(
                    [
                        calib.beam_inclination_min,
                        calib.beam_inclination_max,
                    ],
                    dtype=tf.float32,
                ),
                height=range_image.shape.dims[0],
            )
        )

    # Waymo frame_utils reverses the inclination order before
    # range-image -> Cartesian conversion.
    beam_inclinations = tf.reverse(
        beam_inclinations,
        axis=[-1],
    )

    # ---------------------------------------------------------
    # TOP LiDAR -> vehicle extrinsic
    # ---------------------------------------------------------

    extrinsic = np.asarray(
        calib.extrinsic.transform,
        dtype=np.float32,
    ).reshape(4, 4)

    # ---------------------------------------------------------
    # Range -> XYZ
    #
    # IMPORTANT:
    # pixel_pose=None
    # frame_pose=None
    #
    # This deliberately disables TOP motion compensation.
    # ---------------------------------------------------------

    cartesian = (
        range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(
                ri[..., 0],
                axis=0,
            ),
            tf.expand_dims(
                tf.convert_to_tensor(extrinsic),
                axis=0,
            ),
            tf.expand_dims(
                beam_inclinations,
                axis=0,
            ),
            pixel_pose=None,
            frame_pose=None,
        )
    )

    cartesian = tf.squeeze(
        cartesian,
        axis=0,
    )

    valid = ri[..., 0] > 0

    xyz = tf.gather_nd(
        cartesian,
        tf.where(valid),
    ).numpy()

    intensity = tf.gather_nd(
        ri[..., 1],
        tf.where(valid),
    ).numpy()

    return np.concatenate(
        [
            xyz,
            intensity[:, None],
        ],
        axis=1,
    )


def compare(name, candidate, existing):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("candidate:", candidate.shape)
    print("existing :", existing.shape)

    if candidate.shape != existing.shape:
        print("SHAPE MATCH: False")
        return

    diff = (
        candidate.astype(np.float64)
        - existing.astype(np.float64)
    )

    abs_diff = np.abs(diff)

    print(
        "allclose:",
        np.allclose(
            candidate,
            existing,
            atol=1e-5,
            rtol=0,
        )
    )

    print(
        "XYZ max abs diff:",
        abs_diff[:, :3].max()
    )

    print(
        "XYZ mean abs diff:",
        abs_diff[:, :3].mean()
    )

    print(
        "Intensity max abs diff:",
        abs_diff[:, 3].max()
    )

    print("\nExisting first 5:")
    print(existing[:5])

    print("\nCandidate first 5:")
    print(candidate[:5])

    print("\nCandidate - existing XYZ:")
    print(diff[:5, :3])


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tfrecord",
        required=True,
    )

    parser.add_argument(
        "--existing-npz",
        required=True,
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    frame = load_frame(
        args.tfrecord,
        args.frame,
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

    # Return 1
    r0 = extract_top_without_motion_compensation(
        frame,
        range_images,
        ri_index=0,
    )

    print(
        "Return 0:",
        r0.shape
    )

    # Return 2
    r1 = extract_top_without_motion_compensation(
        frame,
        range_images,
        ri_index=1,
    )

    print(
        "Return 1:",
        r1.shape
    )

    combined = np.concatenate(
        [r0, r1],
        axis=0,
    )

    print(
        "Combined:",
        combined.shape
    )

    compare(
        "TOP RETURN 0 + RETURN 1, NO MOTION COMPENSATION",
        combined,
        existing,
    )


if __name__ == "__main__":
    main()
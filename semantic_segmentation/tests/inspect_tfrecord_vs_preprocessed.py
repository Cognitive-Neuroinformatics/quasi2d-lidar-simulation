#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset


def mat4(transform_list):
    return np.asarray(transform_list, dtype=np.float64).reshape(4, 4)


def find_top_calibration(frame):
    for calib in frame.context.laser_calibrations:
        if calib.name == open_dataset.LaserName.TOP:
            return calib

    raise RuntimeError("TOP LiDAR calibration not found.")


def compare_arrays(name, raw, saved, atol=1e-7):
    raw = np.asarray(raw)
    saved = np.asarray(saved)

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print("raw shape   :", raw.shape)
    print("saved shape :", saved.shape)

    if raw.shape != saved.shape:
        print("SHAPE MATCH : False")
        return

    diff = np.abs(raw.astype(np.float64) - saved.astype(np.float64))

    print("allclose    :", np.allclose(raw, saved, atol=atol, rtol=0))
    print("max abs diff:", diff.max())
    print("mean diff   :", diff.mean())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tfrecord",
        required=True,
        help="Raw Waymo TFRecord for this segment.",
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Root of existing waymo_train_small preprocessing.",
    )

    parser.add_argument(
        "--case",
        required=True,
        help="Segment/case name.",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to compare.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Existing preprocessed files
    # ------------------------------------------------------------

    meta_path = (
        f"{args.root}/meta_infos/{args.case}.pkl"
    )

    calib_path = (
        f"{args.root}/laser_calibrations/{args.case}/"
        "laser_calibrations/laser_calibrations.npz"
    )

    beam_path = (
        f"{args.root}/temp/{args.case}/"
        "beam_inclinations/beam_inclinations.npy"
    )

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    saved_calib = np.load(calib_path)

    saved_beams_single = (
        np.load(beam_path)
        .astype(np.float64)
    )

    # ------------------------------------------------------------
    # Read requested raw TFRecord frame
    # ------------------------------------------------------------

    dataset = tf.data.TFRecordDataset(
        args.tfrecord,
        compression_type="",
    )

    raw_frame = None

    for i, data in enumerate(dataset):

        if i != args.frame:
            continue

        raw_frame = open_dataset.Frame()

        raw_frame.ParseFromString(
            bytearray(data.numpy())
        )

        break

    if raw_frame is None:
        raise RuntimeError(
            f"Frame {args.frame} not found."
        )

    saved_frame = meta["frames"][args.frame]

    # ------------------------------------------------------------
    # 1. Timestamp
    # ------------------------------------------------------------

    raw_timestamp = int(
        raw_frame.timestamp_micros
    )

    saved_timestamp = int(
        saved_frame["log_time_stamp"]
    )

    print("\n" + "=" * 80)
    print("1. TIMESTAMP")
    print("=" * 80)

    print("TFRecord timestamp_micros :", raw_timestamp)
    print("PKL log_time_stamp        :", saved_timestamp)
    print(
        "MATCH                     :",
        raw_timestamp == saved_timestamp
    )

    # ------------------------------------------------------------
    # 2. Vehicle -> world frame pose
    # ------------------------------------------------------------

    raw_pose = mat4(
        raw_frame.pose.transform
    )

    saved_pkl_pose = np.asarray(
        saved_frame["lidar2world"],
        dtype=np.float64,
    )

    saved_npz_pose = np.asarray(
        saved_calib["frame_pose"][args.frame],
        dtype=np.float64,
    )

    compare_arrays(
        "2A. TFRecord frame.pose vs PKL lidar2world",
        raw_pose,
        saved_pkl_pose,
    )

    compare_arrays(
        "2B. TFRecord frame.pose vs calibration frame_pose",
        raw_pose,
        saved_npz_pose,
    )

    print("\nRaw TFRecord frame.pose:")
    print(raw_pose)

    print("\nPKL lidar2world:")
    print(saved_pkl_pose)

    # ------------------------------------------------------------
    # 3. TOP LiDAR calibration
    # ------------------------------------------------------------

    top_calib = find_top_calibration(
        raw_frame
    )

    raw_extrinsic = mat4(
        top_calib.extrinsic.transform
    )

    saved_extrinsic = np.asarray(
        saved_calib["extrinsic"][args.frame],
        dtype=np.float64,
    )

    compare_arrays(
        "3. TOP LiDAR extrinsic: TFRecord vs saved NPZ",
        raw_extrinsic,
        saved_extrinsic,
    )

    print("\nRaw TOP extrinsic:")
    print(raw_extrinsic)

    print("\nSaved TOP extrinsic:")
    print(saved_extrinsic)

    # ------------------------------------------------------------
    # 4. TOP beam inclinations
    # ------------------------------------------------------------

    raw_beams = np.asarray(
        top_calib.beam_inclinations,
        dtype=np.float64,
    )

    saved_beams_frame = np.asarray(
        saved_calib[
            "beam_inclinations"
        ][args.frame],
        dtype=np.float64,
    )

    print("\nRaw TOP beam count:", len(raw_beams))

    compare_arrays(
        "4A. TOP beams: TFRecord vs laser_calibrations.npz",
        raw_beams,
        saved_beams_frame,
        atol=1e-10,
    )

    compare_arrays(
        "4B. TOP beams: TFRecord vs standalone beam_inclinations.npy",
        raw_beams,
        saved_beams_single,
        atol=1e-6,
    )

    print("\nTFRecord beams [rad]:")
    print(raw_beams)

    print("\nTFRecord beams [deg]:")
    print(np.rad2deg(raw_beams))

    # ------------------------------------------------------------
    # 5. Check that calibration is constant across saved frames
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("5. SAVED CALIBRATION CONSISTENCY")
    print("=" * 80)

    extrinsics = saved_calib["extrinsic"]
    beams_all = saved_calib[
        "beam_inclinations"
    ]

    print(
        "All extrinsics equal frame 0:",
        np.allclose(
            extrinsics,
            extrinsics[0][None, :, :],
        ),
    )

    print(
        "All beam rows equal frame 0:",
        np.allclose(
            beams_all,
            beams_all[0][None, :],
        ),
    )


if __name__ == "__main__":
    main()
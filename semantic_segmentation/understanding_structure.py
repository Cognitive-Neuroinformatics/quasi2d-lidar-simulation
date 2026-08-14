import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils


TFRECORD = (
    "/media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/raw_data/"
    "segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord"
)

dataset = tf.data.TFRecordDataset(TFRECORD, compression_type="")

for frame_idx, data in enumerate(dataset):
    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    if dataset_pb2.LaserName.TOP not in segmentation_labels:
        continue

    labels = segmentation_labels[dataset_pb2.LaserName.TOP]

    if len(labels) == 0:
        continue

    print(
        f"Frame {frame_idx:03d}: "
        f"{len(labels)} return(s) with semantic labels"
    )

    for return_idx, label_tensor in enumerate(labels):
        print(
            f"    return {return_idx}: "
            f"shape = {label_tensor.shape}"
        )
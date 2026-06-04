# Quasi-2D LiDAR Dataset Generation Framework

A dataset generation framework for transforming Waymo Open Dataset recordings into quasi-2D pointclouds that emulate the LiDAR sensor characteristics of our real-world autonomous driving platforms.

The framework uses the real mounting positions, orientations, and scan configurations of our Ibeo Scala Gen 1 and Valeo Scala Gen 2 sensor setups to simulate how Waymo scenes would appear from our research vehicles. By combining these sensor-specific simulations with the scale and high-quality annotations of the Waymo Open Dataset, the framework generates annotated training data for developing and evaluating deep learning models for sparse LiDAR perception.

The framework supports:

* Multi-sensor quasi-2D point cloud simulation
* BEV image generation with handcrafted input encodings
* Temporal point cloud fusion (2 frames and 4 frames)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) compatible label generation
* Multi-frame ground-truth generation for temporal database sampling and data augmentation

---

# Overview

Modern autonomous driving datasets are typically collected using high-resolution LiDAR sensors that provide dense 3D observations of the environment. In contrast, compact automotive LiDAR systems often employ narrow vertical fields of view and only a small number of scan layers.

Such sensors generate sparse **quasi-2D point clouds**, making reliable perception significantly more challenging. The reduced vertical information can negatively impact object detection, tracking, and scene understanding.

This repository transforms raw Waymo Open Dataset recordings into sensor-specific quasi-2D LiDAR observations suitable for research on:

* Sparse LiDAR perception
* BEV-based perception
* Multi-frame temporal fusion and 3D object detection

The generated outputs are directly compatible with OpenPCDet-based training pipelines.

---

# Processing Pipeline

The framework transforms dense Waymo Open Dataset point clouds into quasi-2D LiDAR observations that emulate the sensor configurations used on our research vehicles.

For each Waymo frame, the raw range images are first converted into a dense 3D point cloud. Virtual Ibeo Scala Gen 1 or Valeo Scala Gen 2 sensors are then simulated using the configured mounting positions, orientations, and scan geometries. A Bresenham3D raycasting procedure reproduces the sampling characteristics of the target sensor, generating sparse quasi-2D point clouds and corresponding sensor-specific annotations.

The resulting point clouds and annotations are transformed into the local sensor coordinate frame. Previous observations are ego-motion compensated and fused with the current frame to generate 2-frame and 4-frame point clouds together with their corresponding multi-frame ground-truth annotations for temporal database sampling and data augmentation.

The framework can additionally export BEV image representations using handcrafted height, overlap, and combined height-overlap encodings.

```text
Waymo TFRecords
       │
       ▼
Dense Point Cloud Extraction
       │
       ▼
Virtual Sensor Simulation
       │
       ▼
Bresenham3D Raycasting
       │
       ▼
Quasi-2D Point Cloud
       │
       ▼
Sensor Frame Transformation
       │
       ├── OpenPCDet Labels
       ├── BEV Images
       │     ├── Height
       │     ├── Overlap
       │     └── Height + Overlap
       │
       └── Temporal Fusion
             ├── 1f Point Clouds
             ├── 2f Point Clouds + GT DB Sampling
             └── 4f Point Clouds + GT DB Sampling
```

---

# Installation

Create a dedicated Conda environment:

```bash
conda create -n quasi2d-lidar python=3.10
conda activate quasi2d-lidar

pip install -r requirements.txt
```

---

# Dataset Preparation

Download the Waymo Open Dataset and organize the TFRecord files in a directory accessible to the preprocessing pipeline.

Example:

```text
raw_data/
├── segment-0000.tfrecord
├── segment-0001.tfrecord
├── segment-0002.tfrecord
└── ...
```

Dataset splits are defined through:

```text
train.txt
val.txt
test.txt
```

which are referenced through the configuration file.

---

## Usage

### Example 1: Generate BEV Image Dataset

The following command generates BEV image representations for all simulated sensors together with the corresponding labels.

```bash
python waymo_to_quasi2d_pointcloud_processor.py \
    --config config/parameters.yaml \
    --load_dir /data/waymo/raw_data \
    --save_dir /data/simulated_data \
    --sensor scala \
    --run_train_and_val \
    --selected_sensors \
        front_left front_center front_right \
        rear_left rear_center rear_right \
    --save_outputs labels images_overlap \
    --num_proc 30
```

### Example 2: Generate Single-Frame Point Cloud Dataset

```bash
python waymo_to_quasi2d_pointcloud_processor.py \
    --config config/parameters.yaml \
    --load_dir /data/waymo/raw_data \
    --save_dir /data/simulated_data \
    --sensor scala \
    --run_train_and_val \
    --selected_sensors front_center \
    --save_outputs labels points_1f \
    --num_proc 30
```

### Example 3: Generate 2-Frame Fusion Dataset

```bash
python waymo_to_quasi2d_pointcloud_processor.py \
    --config config/parameters.yaml \
    --load_dir /data/waymo/raw_data \
    --save_dir /data/simulated_data \
    --sensor scala \
    --run_train_and_val \
    --selected_sensors front_center \
    --save_outputs labels_2f points_2f gt_2f \
    --num_proc 30
```

### Example 4: Generate 4-Frame Fusion Dataset

```bash
python waymo_to_quasi2d_pointcloud_processor.py \
    --config config/parameters.yaml \
    --load_dir /data/waymo/raw_data \
    --save_dir /data/simulated_data \
    --sensor scala \
    --run_train_and_val \
    --selected_sensors front_center \
    --save_outputs labels_4f points_4f gt_4f \
    --num_proc 30
```

## Command Line Arguments

| Argument                | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| `--config`              | Path to YAML configuration file                        |
| `--load_dir`            | Directory containing Waymo TFRecords                   |
| `--save_dir`            | Output directory                                       |
| `--sensor`              | Sensor configuration to simulate (`scala` or `scala2`) |
| `--prefix`              | Dataset split (`train`, `val`, `test`)                 |
| `--run_train_and_val`   | Process train and validation sets sequentially         |
| `--selected_sensors`    | Subset of sensors to generate                          |
| `--save_outputs`        | Output modalities to save                              |
| `--num_proc`            | Number of worker processes                             |
| `--transformation_mode` | Coordinate transformation mode                         |

---


# Output Directory Structure

The generated dataset is organized by dataset split and sensor view. Each sensor is processed independently and can be used as a standalone dataset.

```text
output/
├── training/
│   └── front_center/
│       ├── points_concat_1f/
│       ├── points_concat_2f/
│       ├── points_concat_4f/
│       ├── labels/
│       ├── labels_2f/
│       ├── labels_4f/
│       ├── images_height/
│       ├── images_overlap/
│       ├── images_overlap_height/
│       └── labels_gt_sampler_json/
│           ├── 2f/
│           └── 4f/
│
└── validation/
    └── front_center/
        ├── points_concat_1f/
        ├── points_concat_2f/
        ├── points_concat_4f/
        ├── labels/
        ├── labels_2f/
        ├── labels_4f/
        ├── images_height/
        ├── images_overlap/
        ├── images_overlap_height/
        └── labels_gt_sampler_json/
            ├── 2f/
            └── 4f/
```

Each sensor is processed independently and stored in its own directory.

---

# Dataset Variants

Different outputs are intended for different downstream tasks.

## BEV-Based Models

```text
images_height/
images_overlap/
images_overlap_height/
labels/
```

The standard `labels/` directory contains the annotations required for BEV image training.

## Single-Frame Point Cloud Models

```text
points_concat_1f/
labels/
```

## Two-Frame Temporal Fusion Models

```text
points_concat_2f/
labels_2f/
labels_gt_sampler_json/2f/
```

These directories should always be used together.

## Four-Frame Temporal Fusion Models

```text
points_concat_4f/
labels_4f/
labels_gt_sampler_json/4f/
```

These directories should always be used together.

---

# Configuration

The preprocessing pipeline is controlled through a YAML configuration file:

```bash
--config config/parameters.yaml
```

The configuration contains three main sections:

1. Dataset configuration
2. BEV generation parameters
3. Sensor simulation parameters

## Dataset Configuration

```yaml
dataset:
  data_percent: 100
  sets_dir: "/data/waymo/ImageSets"
```

The `sets_dir` directory contains the train/validation/test split files used during dataset generation.

## BEV Configuration

```yaml
bev:
  x_size: 1024
  y_size: 1024
  voxel_size: 0.1
  bev_padding: 3.7
  z_min: -1
  z_max: 2
```

These parameters control the generation of BEV image representations.

## Sensor Configuration

The framework currently supports two research-vehicle sensor configurations:

| Configuration | Sensor Type       |
| ------------- | ----------------- |
| `scala`       | Ibeo Scala Gen 1  |
| `scala2`      | Valeo Scala Gen 2 |

Each configuration specifies:

* Sensor mounting positions
* Sensor orientations
* Detection range
* Horizontal field of view
* Angular sampling pattern
* Vertical scan geometry

Sensor selection is performed through:

```bash
--sensor scala
```

or

```bash
--sensor scala2
```

---




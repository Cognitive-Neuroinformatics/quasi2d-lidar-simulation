# Semantic-aware MLS surface reconstruction

The upstream `waymo_preprocessing.py` pipeline performs:

```text
TFRecord → accumulation/label propagation → static filtering → densification
```

and writes one canonical static input:

```text
<dataset-root>/recon_related/<case>/static_recon_labels.npz
```

## Pipeline owned by this repository

```text
recon_related/<case>/static_recon_labels.npz
                     ↓
             semantic PCL MLS
                     ↓
        CPU or CUDA SCALA-2 raycast
```

Dynamic-object inputs under:

```text
<dataset-root>/temp/<case>/occ/preproc/dynamic/objects/
```

## Current dataset defaults

```text
DATASET_ROOT=/data/waymo/surface_reconstruction
```

## Build PCL MLS

```bash
./run_semantic_static_mls.sh build
```

## Run MLS only

```bash
./run_semantic_static_mls.sh reconstruct
```

## overall MLS+Raycaster


```bash
./run_one_scene_unified.sh reconstruct
```



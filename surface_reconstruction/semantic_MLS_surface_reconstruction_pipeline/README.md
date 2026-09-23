# Semantic MLS — unified CPU/CUDA rasterizer pipeline

1. CPU static-ground densification (TO DO!! what to do with curb? buildings? vegetation?)
2. CPU semantic-aware PCL MLS reconstruction
3. selectable SCALA2 rasterization:
   - CPU: original optimized CPU rasterizer
   - CUDA: original PyTorch-CUDA rasterizer


## Main entry point

```bash
./run_one_scene_unified.sh \
  --dataset-root /path/to/road_reconstruction_study \
  --caseid segment-... \
  --rasterizer cpu
```

For CUDA:

```bash
./run_one_scene_unified.sh \
  --dataset-root /path/to/road_reconstruction_study \
  --caseid segment-... \
  --rasterizer cuda \
  --cuda-devices 0 1 2 \
  --cuda-precision float32
```

## Switch rasterizers without rerunning densification and MLS

After a reconstruction has already been produced, render it with CUDA only:

```bash
./run_one_scene_unified.sh \
  --dataset-root /path/to/road_reconstruction_study \
  --caseid segment-... \
  --rasterizer cuda \
  --skip-densify \
  --skip-reconstruct \
  --cuda-devices 0 1 2
```

Or render the same reconstruction with CPU only:

```bash
./run_one_scene_unified.sh \
  --dataset-root /path/to/road_reconstruction_study \
  --caseid segment-... \
  --rasterizer cpu \
  --skip-densify \
  --skip-reconstruct
```

## Useful controls

Run only selected sensors:

```bash
--raycast-sensors front_center rear_center
```

Run selected frames; end is exclusive:

```bash
--raycast-start-frame 17 --raycast-end-frame 20
```

CPU sensor parallelism:

```bash
--sensor-workers 3
```

CUDA GPU selection:

```bash
--cuda-devices 0 1 2
```

CUDA cache budget:

```bash
--cuda-gpu-cache-gb 5
```

## Validation

CPU reference validation remains available:

```bash
python validate_cpu_reference.py /path/to/reference_cpu /path/to/new_cpu --sensors front_center
```

CPU versus CUDA validation remains available under `diagnostics/`:

```bash
python diagnostics/validate_cpu_vs_cuda.py \
  --cpu-root /path/to/scala2_raycast_cpu_optimized \
  --cuda-root /path/to/scala2_raycast_cuda_float64 \
  --sensors front_center \
  --start-frame 17 \
  --end-frame 18 \
  --range-atol 1e-5 \
  --xyz-atol 1e-5
```

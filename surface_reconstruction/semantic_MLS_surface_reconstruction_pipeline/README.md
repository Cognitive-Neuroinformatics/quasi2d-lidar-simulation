# Semantic static MLS v3 — unified CPU/CUDA rasterizer pipeline

This folder combines the two uploaded strict-baseline packages into one runnable pipeline:

1. CPU static-ground densification (TO DO what to do with curb?)
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

The default backend is `cpu`.

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

This is the cleanest way to compare the two rasterizers because they consume the same reconstruction.

## Important baseline defaults

The unified launcher preserves the full-pipeline settings used by the uploaded CPU/CUDA packages:

- intersection mode: `tangent_patch`
- patch radius: `0.03 m`
- hit radius: `0.03 m`
- minimum range: `0.5 m`
- maximum range: `80 m`
- first mirror side: `0`
- exact structured SCALA2 lookup on CPU
- CPU rasterizer default point batch: `500000`
- CUDA rasterizer default point batch: `2000000`
- CUDA default precision: `float32`
- CUDA geometry cache: `5 GiB` per GPU
- NPZ mode in the full benchmark launcher: `stored`



## CUDA precision

The original CUDA package defines two modes:

- `--cuda-precision float32`: accelerated CUDA baseline. This is the unified default for CUDA.
- `--cuda-precision float64`: reference-oriented CUDA mode for closer numerical comparison with the mostly-float64 CPU geometry path.


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

For the strictest comparison, first validate CUDA float64 against CPU. Then separately evaluate the intended CUDA float32 accelerator mode.

# CUDA SCALA2 raycaster (using tangent patch)

This package keeps the CPU-optimized densification and PCL-MLS pipeline and adds a PyTorch-CUDA implementation of the final SCALA2 rendering stage.

## What is on the GPU

For each visible static MLS tile and active dynamic MLS object, CUDA performs:

1. world -> SCALA2 sensor transform
2. range rejection
3. exact structured 16 x 653 SCALA2 angular candidate generation
4. tangent-plane intersection
5. finite patch-radius validation
6. nearest positive hit per ray with the same support-distance tie-break as CPU
7. retention of the exact winning MLS source row

Only the final source indices (at most 10,448 rays per sensor) return to CPU. Intensity, semantic ID, instance ID, ground ID and all extra point-aligned provenance are resolved lazily on CPU from those winning source rows.

Static geometry is cached as `xyz + normal` on each GPU across frames. Dynamic object *local* geometry is also cached; its frame pose is applied on GPU.

## Output layout

```
<output-root>/
  front_left/points/000.npz
  front_center/points/000.npz
  front_right/points/000.npz
  rear_left/points/000.npz
  rear_center/points/000.npz
  rear_right/points/000.npz
  raycast_summary.json
```

## Files added

- `raycaster/cuda_scala2_backend.py` - CUDA geometry cache, structured candidate lookup and per-ray reduction
- `raycaster/raycast_mls_scala2_cuda.py` - complete static + dynamic CUDA renderer
- `check_cuda_environment.py` - verifies CUDA-enabled PyTorch and lists GPUs
- `validate_cpu_vs_cuda.py` - CPU/CUDA output comparison
- `run_cuda_raycast_benchmark.sh` - full benchmark with `/usr/bin/time` and `nvidia-smi` sampling

## 1. Verify CUDA PyTorch

```bash
cd /home/samanti/Documents/Uni_Bremen/PhD/my_workspace/lidar_surface_reconstruction/semantic_static_mls_v3_cuda_raycast_v1
python check_cuda_environment.py
```

You need `CUDA available: True`.

## 2. First validation: one sensor, one frame, float64

Use float64 first because it is the closest numerical comparison with the validated NumPy CPU path.

```bash
cd /home/samanti/Documents/Uni_Bremen/PhD/my_workspace/lidar_surface_reconstruction/semantic_static_mls_v3_cuda_raycast_v1 && \
python raycaster/raycast_mls_scala2_cuda.py \
  --dataset-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study \
  --caseid segment-17791493328130181905_1480_000_1500_000_with_camera_labels \
  --reconstruction-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study/semantic_aware_mls/semantic_static_mls/segment-17791493328130181905_1480_000_1500_000_with_camera_labels \
  --output-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study/semantic_aware_mls/semantic_static_mls/segment-17791493328130181905_1480_000_1500_000_with_camera_labels/scala2_raycast_cuda_fp64_test \
  --sensors front_center \
  --start-frame 17 \
  --end-frame 18 \
  --first-mirror-side 0 \
  --minimum-range 0.5 \
  --max-range 80 \
  --intersection-mode tangent_patch \
  --patch-radius 0.03 \
  --hit-radius 0.03 \
  --point-batch-size 1000000 \
  --gpu-cache-gb 5 \
  --devices 0 \
  --precision float64 \
  --npz-compression stored \
  --overwrite
```

Then compare with the validated CPU output:

```bash
python validate_cpu_vs_cuda.py \
  --cpu-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study/semantic_aware_mls/semantic_static_mls/segment-17791493328130181905_1480_000_1500_000_with_camera_labels/scala2_raycast_cpu_exact_v3 \
  --cuda-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study/semantic_aware_mls/semantic_static_mls/segment-17791493328130181905_1480_000_1500_000_with_camera_labels/scala2_raycast_cuda_fp64_test \
  --sensors front_center \
  --start-frame 17 \
  --end-frame 18 \
  --range-atol 1e-5 \
  --xyz-atol 1e-5
```

## 3. Fast CUDA benchmark: float32

Float32 is the intended accelerator mode. TF32 is explicitly disabled so geometry matrix products use normal FP32 rather than silently reducing mantissa precision.

```bash
python raycaster/raycast_mls_scala2_cuda.py \
  --dataset-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study \
  --caseid segment-17791493328130181905_1480_000_1500_000_with_camera_labels \
  --reconstruction-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study/semantic_aware_mls/semantic_static_mls/segment-17791493328130181905_1480_000_1500_000_with_camera_labels \
  --output-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study/semantic_aware_mls/semantic_static_mls/segment-17791493328130181905_1480_000_1500_000_with_camera_labels/scala2_raycast_cuda_fp32_test \
  --sensors front_center \
  --start-frame 17 \
  --end-frame 20 \
  --max-range 80 \
  --intersection-mode tangent_patch \
  --patch-radius 0.03 \
  --hit-radius 0.03 \
  --point-batch-size 2000000 \
  --gpu-cache-gb 5 \
  --devices 0 \
  --precision float32 \
  --npz-compression stored \
  --overwrite
```

Compare it too. Float32 can differ at numerically near-tied patch boundaries, so validate source/ray identity rather than assuming bit-for-bit equality.

## 4. Overnight: all six sensors, all frames, three GPUs

The benchmark script samples GPU utilization every two seconds and records `/usr/bin/time -v` resource usage.

```bash
cd /home/samanti/Documents/Uni_Bremen/PhD/my_workspace/lidar_surface_reconstruction/semantic_static_mls_v3_cuda_raycast_v1 && \
./run_cuda_raycast_benchmark.sh \
  --dataset-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study \
  --caseid segment-17791493328130181905_1480_000_1500_000_with_camera_labels \
  --reconstruction-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study/semantic_aware_mls/semantic_static_mls/segment-17791493328130181905_1480_000_1500_000_with_camera_labels \
  --output-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study/semantic_aware_mls/semantic_static_mls/segment-17791493328130181905_1480_000_1500_000_with_camera_labels/scala2_raycast_cuda_all_sensors \
  --devices 0,1,2 \
  --precision float32 \
  --point-batch-size 2000000 \
  --gpu-cache-gb 5 \
  --npz-compression stored
```

With 3 GPUs the six sensors are assigned round-robin and execute in two waves per frame:

```
GPU0: front_left  -> rear_left
GPU1: front_center -> rear_center
GPU2: front_right -> rear_right
```

Each device has its own persistent geometry cache. The same static MLS geometry is copied to a given GPU only on first use, then reused across later frames while it remains inside the configured LRU budget.

If there is only one GPU, use `--devices 0`. If VRAM is insufficient, reduce `--point-batch-size` first, then `--gpu-cache-gb`.

## Precision / equivalence

The existing CPU implementation performs most geometric arithmetic in float64. Therefore:

- `--precision float64`: reference-oriented CUDA mode; expected to match CPU much more closely but may be substantially slower on GPUs with weak FP64 throughput.
- `--precision float32`: accelerator mode; usually much faster and uses half the geometry VRAM. Exact ray/source identity must be validated because near-ties can flip under FP32 rounding.

The renderer never changes the semantic/instance/intensity values itself. Once a winning source row is selected, properties are copied exactly from the original MLS NPZ.

## Important: what CUDA does NOT accelerate yet

This package accelerates **raycasting only**. Densification remains NumPy/SciPy CPU and MLS remains PCL CPU. These stay as the reference for the next stages of CUDA conversion.

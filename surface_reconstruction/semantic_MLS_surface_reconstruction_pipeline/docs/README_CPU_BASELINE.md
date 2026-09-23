# Semantic static MLS v3 — CPU-optimized reference before CUDA

This package is the CPU reference pipeline for:

1. static-ground densification
2. semantic + instance-aware PCL MLS reconstruction
3. six-sensor SCALA2 tangent-patch raycasting

The scientific geometry/settings are intentionally kept separate from the implementation optimizations so this can be used as the CPU baseline for the later CUDA version.

## Known raycast baseline from the validated scene

For `segment-17791493328130181905_1480_000_1500_000_with_camera_labels`, front-center frame 17:

- original exact CPU raycaster: **110.5 s**
- structured exact candidate lookup: **46.1 s**
- tile cull + lazy properties, warm-cache frames: **24.3–25.1 s / frame / sensor**

The user-side validation reported identical ray indices, semantic IDs, instance IDs, source types, ranges and XYZ between the exact reference and the optimized v3 raycast output.

The additional v4 changes in this folder should be validated once on the same frame before treating them as the new frozen CPU reference.

## CPU bottlenecks found in the uploaded code

### Densification

The expensive parts are the large cKDTree queries used by generic coverage/plane fitting and metadata-source mapping, plus very large NPZ writes. The original implementation also repeatedly scanned all points by ring sector and used Python loops/set-based deduplication.

CPU changes in this package:

- semantic-family lookup table instead of repeated `np.isin`
- sort/group ring sectors once instead of scanning the full cloud for every sector
- vectorized neighbouring-sector Z evaluation
- vectorized generated-point deduplication
- avoid loading unused intensity during geometry analysis
- compute generated→original metadata source mapping only once
- avoid eager full-cloud float64 duplication during metadata mapping
- **stream the merged densified NPZ one attribute/chunk at a time**, avoiding several-GB peak temporary merged arrays
- optional `stored` NPZ mode: identical arrays, much faster I/O, larger files
- detailed per-stage timing JSON
- densified PCD is optional because MLS consumes the NPZ, not the PCD

### PCL MLS reconstruction

The dominant work is expected to remain PCL MLS itself, followed by PCD exchange and output→source nearest-neighbour attribute transfer.

CPU changes in this package:

- independent PCL jobs can run concurrently (`--mls-workers`)
- each PCL job has an explicit thread count (`--pcl-threads`)
- default full-pipeline split for the 24-logical-CPU workstation: **3 MLS workers × 8 PCL threads**
- BLAS thread pools are kept at 1 to avoid oversubscription
- only XYZ is materialized before each PCL call; intensity/semantic/ground/instance metadata stay in the original arrays until the final nearest source row is known
- background semantic family is classified once per tile rather than repeatedly rescanned for every group
- reconstructed static objects are binned into output tiles once instead of being cropped against every tile repeatedly
- dynamic object MLS models are processed concurrently
- optional uncompressed/stored NPZ output
- exact 3D tile bounds are written directly into `static_manifest.json`, eliminating the raycaster's old first-run bounds-learning pass
- temporary PCL PCD files automatically use `/dev/shm` when at least 8 GiB is free; this removes physical-disk I/O for those temporary files
- reconstruction report records stage timings and summed PCL/PCD/attribute-transfer timings

### SCALA2 raycasting

The original dominant cost was a generic SciPy radius lookup for tens of millions of MLS points per frame.

CPU changes in this package:

- exact structured candidate lookup exploiting the native **16 × 653 SCALA2 ray lattice** instead of a generic cKDTree radius query
- conservative 3D tile-FOV culling before geometry processing
- tile 3D bounds are consumed directly from the optimized reconstruction manifest
- point-level FOV rejection is fused into structured exact candidate generation instead of performed as a separate azimuth/elevation pass
- tangent normals are transformed only for actual point↔ray candidates
- static tiles are no longer duplicated wholesale from float32→float64 before batching
- world XYZ/normals are no longer copied for every range-surviving point; they are gathered only for candidates/winners
- nearest-hit reduction is O(N) over fixed ray IDs instead of lexicographically sorting all candidates
- source properties are loaded lazily only for NPZ files that own final winning ray hits
- dynamic object local→world transforms are computed once per frame and shared across all sensors
- multiple sensors can render concurrently (`--sensor-workers`)
- output remains exactly `sensor/points/frame.npz`, e.g. `front_left/points/017.npz`
- optional stored NPZ output avoids compression CPU cost

## One complete CPU benchmark

The benchmark writes separate `_cpu_optimized` outputs; it does not overwrite the original reconstruction root.

```bash
cd /home/samanti/Documents/Uni_Bremen/PhD/my_workspace/lidar_surface_reconstruction/semantic_static_mls_v3_cpu_optimized_v4

./run_one_scene_cpu_optimized.sh \
  --clean \
  --dataset-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study \
  --caseid segment-17791493328130181905_1480_000_1500_000_with_camera_labels
```

Defaults for the benchmark:

- densification: uploaded pipeline settings, `fill-spacing-mode=local_ring_density`
- NPZ mode: `stored` for CPU timing
- no densified PCD unless `--write-densified-pcd` is supplied
- MLS: `--mls-workers 3 --pcl-threads 8 --attribute-workers 1`
- raycast: all six sensors, all frames
- raycast: `--sensor-workers 3`
- exact structured lookup, tangent-patch radius 0.03 m, range 80 m

If the final densification experiment should use the newer along-ring estimator instead of the uploaded launcher's `local_ring_density`, add:

```bash
--fill-spacing-mode along_ring_density
```

The benchmark creates:

```text
<dataset>/semantic_aware_mls/cpu_benchmarks/<case>/<timestamp>/
├── densify.log
├── densify.resources.txt
├── densify_timing.json
├── reconstruct.log
├── reconstruct.resources.txt
├── raycast.log
├── raycast.resources.txt
├── lscpu.txt
├── memory.txt
├── nvidia_smi.txt
└── cpu_pipeline_benchmark.json
```

At the end it prints total stage time plus the largest densification, MLS and raycast sub-stages. This is the CPU baseline to compare against CUDA.

## Tune six-sensor CPU parallelism before the full raycast

If the optimized reconstruction already exists, benchmark sensor concurrency over frames 17–19:

```bash
python tune_raycast_cpu.py \
  --dataset-root /media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study \
  --caseid segment-17791493328130181905_1480_000_1500_000_with_camera_labels \
  --reconstruction-root /path/to/semantic_static_mls_cpu_optimized/<case> \
  --workers 1 2 3 6 \
  --start-frame 17 \
  --end-frame 20
```

Use the fastest measured `sensor-workers` value for the full scene. This should be measured rather than assumed because the raycaster is heavily memory-bandwidth limited.

## Validate exact CPU raycast output

```bash
python validate_cpu_reference.py \
  /path/to/reference_raycast_root \
  /path/to/new_raycast_root \
  --sensors front_center
```

With the CPU exact path, use the default `--atol 0 --rtol 0`. For a future CUDA implementation, tolerances can be specified explicitly while discrete arrays should still match.

## Why `stored` NPZ is used for the timing baseline

`np.savez_compressed` can spend a large amount of CPU time compressing tens of millions of points. Compression changes storage size, not the scientific arrays. The optimized CPU benchmark therefore defaults to `stored` NPZ so the timing measures densification/MLS/raycast computation rather than ZIP deflation.

The tradeoff is substantially higher disk usage. Use `--npz-compression compressed` if disk space is more important than runtime.

## Next step: CUDA

After one full CPU run, use `cpu_pipeline_benchmark.json` to decide what moves first. The expected first CUDA target is the raycaster hot path:

- world→sensor transform
- range test
- structured SCALA2 candidate generation
- tangent-patch intersection
- nearest-hit reduction

The CPU code in this package is intended to remain the validation reference for that CUDA implementation.

#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-root", required=True, type=Path)
    p.add_argument("--cuda-root", required=True, type=Path)
    p.add_argument("--sensors", nargs="+", default=["front_center"])
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True, help="Exclusive")
    p.add_argument("--range-atol", type=float, default=1e-5)
    p.add_argument("--xyz-atol", type=float, default=1e-5)
    return p.parse_args()

def neq_count(a, b, equal_nan=False):
    if a.shape != b.shape:
        return None
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        if equal_nan:
            eq = (a == b) | (np.isnan(a) & np.isnan(b))
        else:
            eq = a == b
        return int(np.count_nonzero(~eq))
    return int(np.count_nonzero(a != b))

def main():
    args = parse_args()
    overall = True
    print("=" * 110)
    print("DETAILED CPU vs CUDA SCALA2 VALIDATION")
    print("=" * 110)

    for sensor in args.sensors:
        for frame in range(args.start_frame, args.end_frame):
            cpu_path = args.cpu_root / sensor / "points" / f"{frame:03d}.npz"
            cuda_path = args.cuda_root / sensor / "points" / f"{frame:03d}.npz"
            if not cpu_path.is_file() or not cuda_path.is_file():
                print(f"{sensor} frame {frame:03d}: MISSING cpu={cpu_path.is_file()} cuda={cuda_path.is_file()}")
                overall = False
                continue

            with np.load(cpu_path, allow_pickle=False) as a, np.load(cuda_path, allow_pickle=False) as b:
                ncpu, ngpu = len(a["ray_index"]), len(b["ray_index"])
                same_rays = np.array_equal(a["ray_index"], b["ray_index"])
                print(f"\n{sensor} frame {frame:03d}: hits CPU/CUDA = {ncpu}/{ngpu}, same rays={same_rays}")

                fields = [
                    ("semantic_id", False),
                    ("instance_id", False),
                    ("source_type", False),
                    ("source_object_id", False),
                    ("ground_id", False),
                    ("intensity", True),
                ]
                all_fields = True
                for name, equal_nan in fields:
                    if name not in a.files or name not in b.files:
                        print(f"  {name:24s}: missing")
                        all_fields = False
                        continue
                    same = np.array_equal(a[name], b[name], equal_nan=equal_nan) if equal_nan else np.array_equal(a[name], b[name])
                    mismatches = neq_count(a[name], b[name], equal_nan=equal_nan) if a[name].shape == b[name].shape else None
                    print(f"  {name:24s}: same={same} mismatches={mismatches}")
                    all_fields &= same

                geometry_ok = False
                if same_rays and len(a["range_m"]) == len(b["range_m"]):
                    dr = np.abs(a["range_m"].astype(np.float64) - b["range_m"].astype(np.float64))
                    dxyz = np.linalg.norm(a["xyz"].astype(np.float64) - b["xyz"].astype(np.float64), axis=1)
                    geometry_ok = bool(np.all(dr <= args.range_atol) and np.all(dxyz <= args.xyz_atol))
                    print(f"  range max/mean           : {dr.max(initial=0):.12g} / {dr.mean() if len(dr) else 0:.12g}")
                    print(f"  xyz max/mean             : {dxyz.max(initial=0):.12g} / {dxyz.mean() if len(dxyz) else 0:.12g}")
                    print(f"  geometry within tolerance: {geometry_ok}")

                    # Show worst rays.
                    if len(dr):
                        worst = np.argsort(dr)[-5:][::-1]
                        print("  worst range differences:")
                        for i in worst:
                            if dr[i] == 0:
                                continue
                            print(
                                f"    ray={int(a['ray_index'][i])} "
                                f"CPU={float(a['range_m'][i]):.9f} "
                                f"CUDA={float(b['range_m'][i]):.9f} "
                                f"diff={float(dr[i]):.9g}"
                            )
                else:
                    print("  geometry comparison       : ray sets differ")

                # If ray sets are identical, inspect source-sample geometry/provenance.
                if same_rays:
                    for name in ("surface_xyz_world", "surface_xyz_source", "surface_distance_to_ray_m"):
                        if name in a.files and name in b.files and a[name].shape == b[name].shape:
                            aa = a[name].astype(np.float64)
                            bb = b[name].astype(np.float64)
                            if aa.ndim == 2:
                                dd = np.linalg.norm(aa - bb, axis=1)
                            else:
                                dd = np.abs(aa - bb)
                            print(f"  {name:24s}: max diff={dd.max(initial=0):.12g}, nonzero={int(np.count_nonzero(dd))}")

                frame_ok = (ncpu == ngpu and same_rays and all_fields and geometry_ok)
                overall &= frame_ok
                print(f"  RESULT                    : {'PASS' if frame_ok else 'DIFF'}")

    print("\n" + "=" * 110)
    print("OVERALL:", "PASS" if overall else "DIFFERENCES FOUND")
    raise SystemExit(0 if overall else 2)

if __name__ == "__main__":
    main()
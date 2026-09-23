#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

ROOT="/media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/waymo_lidargs_mls_study"

SCENES={
    "dense":"segment-10203656353524179475_7625_000_7645_000_with_camera_labels",
    "dynamic_rich":"segment-17791493328130181905_1480_000_1500_000_with_camera_labels"
}

def load_poses(case):
    path=f"{ROOT}/laser_calibrations/{case}/laser_calibrations/laser_calibrations.npz"

    with np.load(path,allow_pickle=False) as d:
        print("\nFile:",path)
        print("Keys:",d.files)
        poses=np.asarray(d["frame_pose"],dtype=np.float64)

    if poses.ndim==2 and poses.shape[1]==16:
        poses=poses.reshape(-1,4,4)

    if poses.ndim!=3 or poses.shape[1:]!=(4,4):
        raise ValueError(f"Unexpected frame_pose shape: {poses.shape}")

    return poses

def analyse(name,case):
    poses=load_poses(case)

    # Translation part of vehicle -> world transform.
    xyz=poses[:,:3,3]
    xy=xyz[:,:2]

    # Distance travelled between consecutive frames.
    step_xyz=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
    step_xy=np.linalg.norm(np.diff(xy,axis=0),axis=1)

    cumulative=np.concatenate([[0.0],np.cumsum(step_xy)])

    print("\n"+"="*72)
    print(name.upper())
    print("="*72)
    print(f"Frames                    : {len(poses):,}")
    print(f"XY trajectory length      : {step_xy.sum():.3f} m")
    print(f"3D trajectory length      : {step_xyz.sum():.3f} m")
    print(f"XY start -> end distance  : {np.linalg.norm(xy[-1]-xy[0]):.3f} m")
    print(f"World X extent            : {np.ptp(xy[:,0]):.3f} m")
    print(f"World Y extent            : {np.ptp(xy[:,1]):.3f} m")

    print("\nConsecutive XY displacement")
    print("--------------------------------")
    print(f"mean                      : {np.mean(step_xy):.4f} m/frame")
    print(f"median                    : {np.median(step_xy):.4f} m/frame")
    print(f"p10                       : {np.percentile(step_xy,10):.4f} m/frame")
    print(f"p25                       : {np.percentile(step_xy,25):.4f} m/frame")
    print(f"p75                       : {np.percentile(step_xy,75):.4f} m/frame")
    print(f"p90                       : {np.percentile(step_xy,90):.4f} m/frame")
    print(f"p95                       : {np.percentile(step_xy,95):.4f} m/frame")
    print(f"max                       : {np.max(step_xy):.4f} m/frame")

    # Waymo is normally 10 Hz, but keep this explicitly labelled as an
    # estimate rather than using it for the geometric conclusion.
    assumed_hz=10.0
    speed=step_xy*assumed_hz
    print("\nApproximate speed assuming 10 Hz")
    print("--------------------------------")
    print(f"median                    : {np.median(speed)*3.6:.2f} km/h")
    print(f"p90                       : {np.percentile(speed,90)*3.6:.2f} km/h")
    print(f"max                       : {np.max(speed)*3.6:.2f} km/h")

    return {
        "name":name,
        "poses":poses,
        "xy":xy,
        "step_xy":step_xy,
        "cumulative":cumulative
    }

results={name:analyse(name,case) for name,case in SCENES.items()}

print("\n\n"+"="*72)
print("DIRECT COMPARISON")
print("="*72)
print(f"{'Metric':<30}{'DENSE':>18}{'DYNAMIC-RICH':>18}")
print("-"*66)

a=results["dense"]
b=results["dynamic_rich"]

metrics=[
    ("Frames",len(a["poses"]),len(b["poses"]),".0f"),
    ("Trajectory [m]",a["step_xy"].sum(),b["step_xy"].sum(),".2f"),
    ("Median step [m]",np.median(a["step_xy"]),np.median(b["step_xy"]),".4f"),
    ("P90 step [m]",np.percentile(a["step_xy"],90),np.percentile(b["step_xy"],90),".4f"),
    ("P95 step [m]",np.percentile(a["step_xy"],95),np.percentile(b["step_xy"],95),".4f"),
    ("Maximum step [m]",np.max(a["step_xy"]),np.max(b["step_xy"]),".4f")
]

for label,x,y,fmt in metrics:
    print(f"{label:<30}{format(x,fmt):>18}{format(y,fmt):>18}")

# Plot trajectory shape.
plt.figure(figsize=(10,8))
for name,r in results.items():
    xy=r["xy"]
    plt.plot(xy[:,0],xy[:,1],label=name)
    plt.scatter(xy[0,0],xy[0,1],s=50)
    plt.scatter(xy[-1,0],xy[-1,1],s=50)

plt.xlabel("World X [m]")
plt.ylabel("World Y [m]")
plt.title("Ego-vehicle trajectories")
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.savefig("ego_trajectory_comparison.png",dpi=200)
plt.close()

# Plot displacement frame by frame.
plt.figure(figsize=(12,6))
for name,r in results.items():
    plt.plot(np.arange(1,len(r["poses"])),r["step_xy"],label=name)

plt.xlabel("Frame")
plt.ylabel("XY displacement from previous frame [m]")
plt.title("Consecutive ego displacement")
plt.legend()
plt.tight_layout()
plt.savefig("ego_frame_displacement.png",dpi=200)
plt.close()

print("\nSaved:")
print("  ego_trajectory_comparison.png")
print("  ego_frame_displacement.png")
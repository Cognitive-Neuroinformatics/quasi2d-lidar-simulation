#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT="/media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/waymo_lidargs_mls_study/recon_related"

SCENES={
    "dense":"segment-10203656353524179475_7625_000_7645_000_with_camera_labels",
    "dynamic_rich":"segment-17791493328130181905_1480_000_1500_000_with_camera_labels"
}

ROAD_ID=18
GRID_SIZES=[0.10,0.20]
NN_SAMPLE=2_000_000
HEATMAP_GRID=0.20

def load_road(name,case):
    path=f"{ROOT}/{case}/static_recon_labels.npz"
    print("\n"+"="*80)
    print(name.upper())
    print("="*80)
    print("Input:",path)

    with np.load(path,allow_pickle=False) as d:
        print("Keys:",d.files)
        xyz=np.asarray(d["xyz"],dtype=np.float64)
        semantic=np.asarray(d["semantic_id"],dtype=np.int16)

    road=xyz[semantic==ROAD_ID]
    finite=np.all(np.isfinite(road),axis=1)
    road=road[finite]

    print(f"Total static points : {len(xyz):,}")
    print(f"ROAD points         : {len(road):,}")
    print(f"ROAD fraction       : {100*len(road)/len(xyz):.2f}%")

    return road

def nearest_neighbor_stats(xyz):
    # XY is intentional: we are measuring ground sampling density.
    xy=xyz[:,:2]

    if len(xy)>NN_SAMPLE:
        rng=np.random.default_rng(42)
        idx=rng.choice(len(xy),NN_SAMPLE,replace=False)
        query=xy[idx]
        print(f"NN query sample     : {len(query):,}/{len(xy):,}")
    else:
        query=xy
        print(f"NN query sample     : {len(query):,}")

    print("Building ROAD XY cKDTree...")
    tree=cKDTree(xy)

    print("Querying nearest neighbours...")
    distances,_=tree.query(query,k=2,workers=-1)
    nn=distances[:,1]
    nn=nn[np.isfinite(nn)]

    result={
        "median":np.percentile(nn,50),
        "p90":np.percentile(nn,90),
        "p95":np.percentile(nn,95),
        "p99":np.percentile(nn,99),
        ">0.05":100*np.mean(nn>0.05),
        ">0.10":100*np.mean(nn>0.10),
        ">0.20":100*np.mean(nn>0.20),
        ">0.30":100*np.mean(nn>0.30)
    }

    return result

def grid_stats(xyz,resolution):
    xy=xyz[:,:2]
    minimum=np.min(xy,axis=0)

    cells=np.floor((xy-minimum)/resolution).astype(np.int64)
    unique_cells=np.unique(cells,axis=0)

    occupied=len(unique_cells)
    points_per_cell=len(xy)/occupied

    return {
        "resolution":resolution,
        "occupied_cells":occupied,
        "occupied_area":occupied*resolution**2,
        "points_per_occupied_cell":points_per_cell
    }

def make_heatmap(xyz,name):
    xy=xyz[:,:2]

    xmin,ymin=np.min(xy,axis=0)
    xmax,ymax=np.max(xy,axis=0)

    xedges=np.arange(xmin,xmax+HEATMAP_GRID,HEATMAP_GRID)
    yedges=np.arange(ymin,ymax+HEATMAP_GRID,HEATMAP_GRID)

    hist,xe,ye=np.histogram2d(
        xy[:,0],
        xy[:,1],
        bins=[xedges,yedges]
    )

    # log1p lets both low-density and high-density regions remain visible.
    image=np.log1p(hist.T)

    plt.figure(figsize=(12,10))
    plt.imshow(
        image,
        origin="lower",
        extent=[xe[0],xe[-1],ye[0],ye[-1]],
        aspect="equal"
    )
    plt.colorbar(label="log(1 + ROAD points per 20 cm cell)")
    plt.xlabel("World X [m]")
    plt.ylabel("World Y [m]")
    plt.title(f"ROAD sampling density — {name}")
    plt.tight_layout()

    output=f"road_density_{name}.png"
    plt.savefig(output,dpi=200)
    plt.close()

    print("Heatmap             :",output)

def print_nn(stats):
    print("\nNearest-neighbour distance in XY")
    print("--------------------------------")
    print(f"median              : {stats['median']:.4f} m")
    print(f"p90                 : {stats['p90']:.4f} m")
    print(f"p95                 : {stats['p95']:.4f} m")
    print(f"p99                 : {stats['p99']:.4f} m")
    print(f"NN >  5 cm          : {stats['>0.05']:.3f}%")
    print(f"NN > 10 cm          : {stats['>0.10']:.3f}%")
    print(f"NN > 20 cm          : {stats['>0.20']:.3f}%")
    print(f"NN > 30 cm          : {stats['>0.30']:.3f}%")

def analyse(name,case):
    road=load_road(name,case)

    nn=nearest_neighbor_stats(road)
    print_nn(nn)

    grids={}
    print("\nOccupied ROAD grid")
    print("------------------")

    for resolution in GRID_SIZES:
        stats=grid_stats(road,resolution)
        grids[resolution]=stats

        print(f"\nGrid {resolution:.2f} m")
        print(f"occupied cells      : {stats['occupied_cells']:,}")
        print(f"occupied XY area    : {stats['occupied_area']:,.2f} m²")
        print(f"points/occupied cell: {stats['points_per_occupied_cell']:.2f}")

    make_heatmap(road,name)

    return {
        "road_points":len(road),
        "nn":nn,
        "grids":grids
    }

results={}

for name,case in SCENES.items():
    results[name]=analyse(name,case)

print("\n\n"+"="*80)
print("DIRECT COMPARISON")
print("="*80)

dense=results["dense"]
sparse=results["dynamic_rich"]

print(f"{'Metric':<28}{'DENSE':>18}{'DYNAMIC-RICH':>18}")
print("-"*64)
print(f"{'ROAD points':<28}{dense['road_points']:>18,}{sparse['road_points']:>18,}")
print(f"{'NN median [m]':<28}{dense['nn']['median']:>18.4f}{sparse['nn']['median']:>18.4f}")
print(f"{'NN p95 [m]':<28}{dense['nn']['p95']:>18.4f}{sparse['nn']['p95']:>18.4f}")
print(f"{'NN p99 [m]':<28}{dense['nn']['p99']:>18.4f}{sparse['nn']['p99']:>18.4f}")
print(f"{'NN >10 cm [%]':<28}{dense['nn']['>0.10']:>18.3f}{sparse['nn']['>0.10']:>18.3f}")
print(f"{'NN >20 cm [%]':<28}{dense['nn']['>0.20']:>18.3f}{sparse['nn']['>0.20']:>18.3f}")

for resolution in GRID_SIZES:
    d=dense["grids"][resolution]
    s=sparse["grids"][resolution]
    label=f"occupied {int(resolution*100)}cm cells"
    print(f"{label:<28}{d['occupied_cells']:>18,}{s['occupied_cells']:>18,}")

print("\nDone.")
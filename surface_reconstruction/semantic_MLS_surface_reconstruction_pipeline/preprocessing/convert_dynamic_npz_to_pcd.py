import os,struct
import numpy as np

ROOT="/data/waymo/waymo_lidargs_mls_study/temp/segment-17791493328130181905_1480_000_1500_000_with_camera_labels/occ/preproc/dynamic/objects"

def save_pcd(path,xyz,intensity,semantic,ground,instance):
    n=len(xyz)
    header=f"""# .PCD v0.7
VERSION 0.7
FIELDS x y z intensity semantic_id ground_id instance_id
SIZE 4 4 4 4 2 1 4
TYPE F F F F I I I
COUNT 1 1 1 1 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA binary
"""
    with open(path,"wb") as f:
        f.write(header.encode())
        for i in range(n):
            f.write(struct.pack("<ffffhbi",float(xyz[i,0]),float(xyz[i,1]),float(xyz[i,2]),float(intensity[i]),int(semantic[i]),int(ground[i]),int(instance[i])))

count=0
for name in sorted(os.listdir(ROOT)):
    d=os.path.join(ROOT,name)
    src=os.path.join(d,"stitch_labeled.npz")
    dst=os.path.join(d,"stitch_labeled.pcd")
    if not os.path.isfile(src): continue

    with np.load(src,allow_pickle=False) as x:
        xyz=x["xyz"]
        intensity=x["intensity"]
        semantic=x["semantic_id"]
        ground=x["ground_id"]
        instance=x["instance_id"]

    save_pcd(dst,xyz,intensity,semantic,ground,instance)
    print(f"{name}: {len(xyz)} points")
    count+=1

print(f"\nConverted {count} objects")
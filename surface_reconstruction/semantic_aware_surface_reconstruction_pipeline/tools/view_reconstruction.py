#!/usr/bin/env python3
"""View a tiled hybrid static reconstruction before raycasting."""
import argparse,json
from pathlib import Path
import numpy as np
SEM={1:[.9,.1,.1],2:[.75,.2,.1],3:[.7,.1,.25],4:[.65,.25,.2],5:[1,.45,0],6:[1,.7,0],7:[.6,.1,.8],8:[.95,.85,.1],9:[1,.35,.35],10:[.45]*3,11:[1,.3,0],12:[0,.7,.9],13:[.2,.4,1],14:[.95,.75,.1],15:[.1,.65,.1],16:[.35,.2,.1],17:[.65,.35,.25],18:[.3]*3,19:[1]*3,20:[.55,.5,.4],21:[.75,.5,.4],22:[.7,.25,.2]}
SURF={10:[.15,.35,.95],11:[.2,.7,.9],12:[.7,.35,.85],20:[.95,.45,.1],30:[.45,.45,.45]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--reconstruction-root',required=True,type=Path);p.add_argument('--mode',choices=['semantic','surface_type','uniform'],default='semantic');p.add_argument('--max-points',type=int,default=3000000);p.add_argument('--point-size',type=float,default=1.5);a=p.parse_args()
 try:import open3d as o3d
 except ImportError as e:raise RuntimeError('Install open3d') from e
 root=a.reconstruction_root.expanduser().resolve();m=json.loads((root/'static_manifest.json').read_text());rng=np.random.default_rng(13);xyz=[];attr=[];counts=np.asarray([int(t.get('point_count',0)) for t in m['tiles']],np.int64);total=max(int(counts.sum()),1)
 for t,c in zip(m['tiles'],counts):
  with np.load(root/t['file'],allow_pickle=False) as d:
   x=np.asarray(d['xyz'],np.float32);v=np.asarray(d['semantic_id'] if a.mode=='semantic' else d['surface_type'] if a.mode=='surface_type' and 'surface_type' in d.files else np.zeros(len(x),np.uint8))
  q=max(1,round(a.max_points*c/total)) if a.max_points>0 else len(x)
  if len(x)>q:i=rng.choice(len(x),q,False);x=x[i];v=v[i]
  xyz.append(x);attr.append(v)
 x=np.concatenate(xyz);v=np.concatenate(attr)
 colors=np.tile([.2,.55,.95],(len(x),1)) if a.mode=='uniform' else np.asarray([(SEM if a.mode=='semantic' else SURF).get(int(k),[.15]*3) for k in v],float)
 cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(x.astype(float));cloud.colors=o3d.utility.Vector3dVector(colors);vis=o3d.visualization.Visualizer();vis.create_window('Hybrid reconstruction',1500,950);vis.add_geometry(cloud);opt=vis.get_render_option();opt.background_color=np.asarray([1.,1.,1.]);opt.point_size=a.point_size;print(f'Displaying {len(x):,} points from {len(m["tiles"])} tiles | mode={a.mode}');vis.run();vis.destroy_window()
if __name__=='__main__':main()

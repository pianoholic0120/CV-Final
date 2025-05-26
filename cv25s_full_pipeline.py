import argparse, sys, os
from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2, open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Constants
RGB_SUFFIX = ".color.png"
DEPTH_PROJ_SUFFIX = ".depth.proj.png"
POSE_TXT = "poses_final.txt"
INTRINSIC_TUP = (525, 525, 320, 240)

# File utilities
def find_seq_dirs(data_root: Path) -> List[Path]:
    seq_dirs=[]
    for scene in sorted(data_root.iterdir()):
        if not scene.is_dir(): continue
    #     for split in ("train","test"):
    #         sub=scene/split
    #         if sub.exists(): seq_dirs+=list(sorted(sub.glob("seq-*")))
    #     seq_dirs+=list(sorted(scene.glob("seq-*")))
    # return seq_dirs
        test_sub = scene / "test"
        if test_sub.exists():
            seq_dirs += list(sorted(test_sub.glob("seq-*")))
    return seq_dirs

def list_frames(seq_dir: Path) -> List[str]:
    return sorted(f.name.replace(RGB_SUFFIX,"") for f in seq_dir.glob(f"*{RGB_SUFFIX}"))

# I/O and preprocessing
def imread(path: Path, flags=cv2.IMREAD_UNCHANGED):
    img=cv2.imread(str(path),flags)
    if img is None: raise IOError(f"Cannot load {path}")
    if flags==cv2.IMREAD_COLOR: img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def preprocess_depth(raw: np.ndarray) -> np.ndarray:
    d=raw.astype(np.float32)
    invalid=(d==0)|(d==65535)
    d[invalid]=np.nan; d*=1e-3
    return d

# Load poses
def load_poses(seq_dir: Path, frames: List[str]) -> Dict[str,np.ndarray]:
    poses={}; pose_file = seq_dir / "poses_final.txt"
    if pose_file.exists():
        for ln in pose_file.open():
            tokens = ln.split()
            if len(tokens) < 8:
                continue
            name = tokens[0]
            qw, qx, qy, qz = map(float, tokens[1:5])
            tx, ty, tz         = map(float, tokens[5:8])
            # Rcw = R.from_quat([qx, qy, qz, qw]).as_matrix()
            Rwc = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T   = np.eye(4, dtype=np.float32)
            # T[:3,:3] = Rcw
            T[:3,:3] = Rwc
            T[:3, 3] = [tx, ty, tz]
            if len(tokens) == 9:
                T = np.linalg.inv(T)
            poses[name.replace(RGB_SUFFIX, "")] = T
        return poses
    for f in frames:
        p=seq_dir/f"{f}.pose.txt"
        # if p.exists(): poses[f]=np.linalg.inv(np.loadtxt(p).astype(np.float32))
        if p.exists(): poses[f]=np.loadtxt(p).astype(np.float32)
    if not poses: raise FileNotFoundError(f"No pose files in {seq_dir}")
    return poses

# Odometry & pose refinement
def refine_poses_rgbd(frames: List[str], rgbds: List[o3d.geometry.RGBDImage],
                      intrinsic: o3d.camera.PinholeCameraIntrinsic,
                      init_poses: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    pg = o3d.pipelines.registration.PoseGraph()
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    for i in range(len(frames)-1):
        f1, f2 = frames[i], frames[i+1]
        T1_cw = init_poses[f1]  
        T2_cw = init_poses[f2] 
        T1_to_T2 = np.linalg.inv(T2_cw) @ T1_cw
        
        result = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbds[i], rgbds[i+1], intrinsic, T1_to_T2,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm())
        success, trans = result[0], result[1]
        if not success:
            print(f"Odometry failed {i}->{i+1}, using init")
            trans = T1_to_T2
        pc1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbds[i], intrinsic)
        pc2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbds[i+1], intrinsic)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            pc1, pc2, 0.05, trans)
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            i, i+1, trans, info, False))
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(trans)))
    opt = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.05, edge_prune_threshold=0.25,
        preference_loop_closure=0.1, reference_node=0)
    
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        opt)
    refined_poses = {}
    for i, f in enumerate(frames):
        T_rel = pg.nodes[i].pose  
        T_ref_to_world = init_poses[frames[0]] 
        refined_poses[f] = T_ref_to_world @ T_rel
    
    return refined_poses

# TSDF weight
def compute_weight(depth: np.ndarray, voxel: float) -> np.ndarray:
    w=np.exp(-depth/2.0); w[np.isnan(depth)]=0
    return w.astype(np.float32)

# Integrate sequence
def integrate_seq(seq_dir: Path, voxel: float, kf_every: int,
                  depth_trunc: float=3.5, trunc_mult: float=6.0) -> o3d.geometry.PointCloud:
    frames=list_frames(seq_dir)[::kf_every]
    poses=load_poses(seq_dir,frames)
    fx,fy,cx,cy=INTRINSIC_TUP
    intrinsic=o3d.camera.PinholeCameraIntrinsic(640,480,fx,fy,cx,cy)
    rgbds, valid = [], []
    for f in frames:
        dpath, rpath = seq_dir/f"{f}{DEPTH_PROJ_SUFFIX}", seq_dir/f"{f}{RGB_SUFFIX}"
        if not dpath.exists() or not rpath.exists(): continue
        depth=preprocess_depth(imread(dpath)); color=imread(rpath,cv2.IMREAD_COLOR)
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        rgbds.append(rgbd); valid.append(f)
    frames, poses = valid, {f:poses[f] for f in valid}
    poses_ref=refine_poses_rgbd(frames, rgbds, intrinsic, poses)
    tsdf=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=max(trunc_mult*voxel,0.04),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    use_w=hasattr(tsdf, 'integrate_with_weights')
    for f, rgbd in zip(frames, rgbds):
        depth_np=np.asarray(rgbd.depth)
        if use_w:
            tsdf.integrate_with_weights(rgbd, intrinsic, poses_ref[f], compute_weight(depth_np,voxel))
        else:
            tsdf.integrate(rgbd, intrinsic, poses_ref[f])
    mesh=tsdf.extract_triangle_mesh(); mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(300000)

# Reconstruction & sweep
def run_reconstruction(data_root: Path, out_root: Path, voxel: float, kf_every: int):
    for seq in tqdm(find_seq_dirs(data_root), desc="Seqs"):
        try:
            pcd=integrate_seq(seq,voxel,kf_every)
        except Exception as e:
            print(f"[skip] {seq}: {e}"); continue
        pcd=pcd.voxel_down_sample(0.0075)
        scene=seq.parent.parent.name if seq.parent.name in ("train","test") else seq.parent.name
        out=out_root/f"{scene}-{seq.name}-v{voxel*1000:.1f}-k{kf_every}.ply"
        out.parent.mkdir(parents=True,exist_ok=True)
        o3d.io.write_point_cloud(str(out),pcd)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data_root",required=True)
    p.add_argument("--output_dir",default="results")
    p.add_argument("--voxels",default="0.0025,0.003")
    p.add_argument("--kfs",default="10,20")
    p.add_argument("--debug",action="store_true")
    args=p.parse_args()
    print(f"Open3D {o3d.__version__}")
    vs=[float(v) for v in args.voxels.split(",")]
    ks=[int(k) for k in args.kfs.split(",")]
    root=Path(args.data_root).expanduser()
    out=Path(args.output_dir).expanduser(); out.mkdir(exist_ok=True)
    for v in vs:
        for k in ks:
            print(f"=== Sweep: voxel={v} kf={k} ===")
            run_reconstruction(root,out/f"v{v*1000:.1f}_k{k}",v,k)

if __name__=='__main__': 
    main()

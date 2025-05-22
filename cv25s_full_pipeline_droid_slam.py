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
    seq_dirs = []
    for scene in sorted(data_root.iterdir()):
        if not scene.is_dir(): continue
        test_sub = scene / "test"
        if test_sub.exists():
            seq_dirs += list(sorted(test_sub.glob("seq-*")))
    return seq_dirs

def list_frames(seq_dir: Path) -> List[str]:
    return sorted(f.name.replace(RGB_SUFFIX, "") for f in seq_dir.glob(f"*{RGB_SUFFIX}"))

# I/O and preprocessing
def imread(path: Path, flags=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(str(path), flags)
    if img is None:
        raise IOError(f"Cannot load {path}")
    if flags == cv2.IMREAD_COLOR:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess_depth(raw: np.ndarray) -> np.ndarray:
    d = raw.astype(np.float32)
    invalid = (d == 0) | (d == 65535)
    d[invalid] = np.nan
    d *= 1e-3
    return d

# Load poses (ACE0 / official)
def load_poses(seq_dir: Path, frames: List[str]) -> Dict[str, np.ndarray]:
    poses = {}
    ace = seq_dir / POSE_TXT
    if ace.exists():
        for ln in ace.open():
            name, qw, qx, qy, qz, x, y, z, *_ = ln.split()
            Rcw = R.from_quat([float(qx),float(qy),float(qz),float(qw)]).as_matrix()
            t = np.array([x,y,z], dtype=np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3,:3] = Rcw; T[:3,3] = t
            poses[name.replace(RGB_SUFFIX, "")] = T
        return poses
    for f in frames:
        p = seq_dir / f"{f}.pose.txt"
        if p.exists():
            T_wc = np.loadtxt(p).astype(np.float32)
            poses[f] = np.linalg.inv(T_wc)
    if not poses:
        raise FileNotFoundError(f"No pose files in {seq_dir}")
    return poses

# Load DROID-SLAM poses
def load_droid_frames_and_poses(pose_file: Path) -> Dict[str, np.ndarray]:
    poses = {}
    with open(pose_file) as f:
        for ln in f:
            tok = ln.strip().split()
            name = tok[0]
            vals = list(map(float, tok[1:]))
            T = np.eye(4, dtype=np.float32)
            T[:3,:4] = np.array(vals, dtype=np.float32).reshape(3,4)
            poses[name] = T
    return poses

# Pose refinement (RGBD odometry + pose graph)
def refine_poses_rgbd(frames: List[str], rgbds: List[o3d.geometry.RGBDImage],
                      intrinsic: o3d.camera.PinholeCameraIntrinsic,
                      init_poses: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    pg = o3d.pipelines.registration.PoseGraph()
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    for i in range(len(frames)-1):
        f1, f2 = frames[i], frames[i+1]
        init = init_poses[f2] @ np.linalg.inv(init_poses[f1])
        success, trans = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbds[i], rgbds[i+1], intrinsic, init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm())
        if not success:
            trans = init
        pc1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbds[i], intrinsic)
        pc2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbds[i+1], intrinsic)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            pc1, pc2, 0.05, trans)
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            i, i+1, trans, info, False))
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(trans)))
    opt = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.05,
        edge_prune_threshold=0.25,
        preference_loop_closure=0.1,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        opt)
    refined = {f: np.linalg.inv(pg.nodes[i].pose) for i,f in enumerate(frames)}
    return refined

# TSDF weight function
def compute_weight(depth: np.ndarray, voxel: float) -> np.ndarray:
    w = np.exp(-depth/2.0)
    w[np.isnan(depth)] = 0
    return w.astype(np.float32)

# Main integration per sequence
def integrate_seq(seq_dir: Path,
                  voxel: float,
                  kf_every: int,
                  pose_source: str,
                  droid_pose_file: str,
                  depth_trunc: float,
                  trunc_mult: float) -> o3d.geometry.PointCloud:
    """
    Integrate TSDF for one sequence, supports ACE0 or per-sequence DROID poses.
    """
    scene = seq_dir.parent.parent.name if seq_dir.parent.name in ("train","test") else seq_dir.parent.name
    seq_name = seq_dir.name  # e.g. seq-03-test

    # 1. Load poses & frames
    if pose_source == "droid":
        # expect one pose file per sequence named <scene>-<seq>.txt
        if droid_pose_file is None:
            raise ValueError("droid_pose_file must be a directory path when using droid source")
        # build sequence-specific pose file path
        pose_txt = Path(droid_pose_file) / f"{scene}-{seq_name}-poses.txt"
        if not pose_txt.exists():
            print(f"[warning] DROID pose file not found for {scene}/{seq_name}, fallback to ACE0")
            frames = list_frames(seq_dir)[::kf_every]
            poses  = load_poses(seq_dir, frames)
        else:
            # load only valid lines with 12 floats
            poses_all = load_droid_frames_and_poses(pose_txt)
            frames = []
            poses = {}
            for name, T in poses_all.items():
                color_file = seq_dir / f"{name}{RGB_SUFFIX}"
                depth_file = seq_dir / f"{name}{DEPTH_PROJ_SUFFIX}"
                if color_file.exists() and depth_file.exists():
                    frames.append(name)
                    poses[name] = T
    else:
        frames = list_frames(seq_dir)[::kf_every]
        poses  = load_poses(seq_dir, frames)

    # 2. Prepare camera intrinsics
    fx,fy,cx,cy = INTRINSIC_TUP
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640,480,fx,fy,cx,cy)

    # 3. Read RGB-D for selected frames
    rgbds, valid = [], []
    for f in frames:
        c = seq_dir / f"{f}{RGB_SUFFIX}"
        d = seq_dir / f"{f}{DEPTH_PROJ_SUFFIX}"
        if not c.exists() or not d.exists():
            continue
        depth = preprocess_depth(imread(d))
        color = imread(c, cv2.IMREAD_COLOR)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False)
        rgbds.append(rgbd)
        valid.append(f)

    # 4. Sync valid frames & poses
    frames = valid
    poses = {f: poses[f] for f in frames if f in poses}

    # 5. Pose refinement (skip for DROID)
    if pose_source == "droid":
        poses_ref = poses
    else:
        poses_ref = refine_poses_rgbd(frames, rgbds, intrinsic, poses)

    # 6. TSDF integration
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=max(trunc_mult*voxel,0.04),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    use_w = hasattr(tsdf, 'integrate_with_weights')
    for f, rgbd in zip(frames, rgbds):
        depth_np = np.asarray(rgbd.depth)
        if use_w:
            tsdf.integrate_with_weights(rgbd, intrinsic, poses_ref[f], compute_weight(depth_np, voxel))
        else:
            tsdf.integrate(rgbd, intrinsic, poses_ref[f])

    # 7. Extract mesh and sample
    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(300000)

# Reconstruction and sweep
def run_reconstruction(data_root: Path, out_root: Path,
                       voxel: float, kf_every: int,
                       pose_source: str, droid_pose_file: str,
                       depth_trunc: float, trunc_mult: float):
    for seq in tqdm(find_seq_dirs(data_root), desc="Sequences"):
        try:
            pcd = integrate_seq(seq, voxel, kf_every,
                                pose_source, droid_pose_file,
                                depth_trunc, trunc_mult)
        except Exception as e:
            print(f"[skip] {seq}: {e}")
            continue
        pcd = pcd.voxel_down_sample(0.0075)
        scene = seq.parent.parent.name if seq.parent.name in ("train","test") else seq.parent.name
        out_path = out_root / f"{scene}-{seq.name}-v{voxel*1000:.1f}-k{kf_every}.ply"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_path), pcd)

# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--voxels", default="0.0025,0.003")
    parser.add_argument("--kfs", default="10,20")
    parser.add_argument("--pose_source", choices=["ace0","droid"], default="ace0")
    parser.add_argument("--droid_pose_file", default=None)
    parser.add_argument("--depth_trunc", type=float, default=3.5)
    parser.add_argument("--trunc_mult", type=float, default=6.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Open3D version: {o3d.__version__}")
    voxels = [float(v) for v in args.voxels.split(",")]
    kfs    = [int(k)   for k in args.kfs.split(",")]
    data_root = Path(args.data_root).expanduser()
    out_root  = Path(args.output_dir).expanduser()
    out_root.mkdir(exist_ok=True, parents=True)

    for v in voxels:
        for k in kfs:
            print(f"=== Sweep: voxel={v}  kf={k} ===")
            run_reconstruction(data_root, out_root / f"v{v*1000:.1f}_k{k}",
                               v, k,
                               args.pose_source, args.droid_pose_file,
                               args.depth_trunc, args.trunc_mult)

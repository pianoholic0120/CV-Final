import cv2
import os
from pathlib import Path
import open3d as o3d
import numpy as np
import trimesh

def save_cameras(pose_file, sparse_dir):
    # assume all images are the same
    rgb_files = []
    rots = []
    translations = []
    focal_lengths = []
    confidences = []
    with open(pose_file, 'r') as f:
        for line in f:
            tokens = line.split()
            rgb_files.append(tokens[0])
            qw, qx, qy, qz = [float(t) for t in tokens[1:5]]
            rots.append([qw, qx, qy, qz])
            tx, ty, tz = [float(t) for t in tokens[5:8]]
            translations.append([tx, ty, tz])
            focal_length = float(tokens[8])
            confidence = float(tokens[9])
            focal_lengths.append(focal_length)
            confidences.append(confidence)

    rgb_image = cv2.imread(rgb_files[0])
    imgs_shape = rgb_image.shape #(H,W,C)

    print(imgs_shape)
    target_file=os.path.join(sparse_dir, 'cameras.txt')
    with open(target_file, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        for i in range(len(rgb_files)):
            f.write(f"{i} PINHOLE {imgs_shape[1]} {imgs_shape[0]} {focal_lengths[i]} {focal_lengths[i]} {imgs_shape[1]/2} {imgs_shape[0]/2}\n")

def save_images_txt(pose_file, sparse_dir):
    images_file =  os.path.join(sparse_dir, 'images.txt')

    rgb_files = []
    rots = []
    translations = []
    focal_lengths = []
    confidences = []
    with open(pose_file, 'r') as f:
        for line in f:
            tokens = line.split()
            rgb_files.append(tokens[0])
            qw, qx, qy, qz = [float(t) for t in tokens[1:5]]
            rots.append([qw, qx, qy, qz])
            tx, ty, tz = [float(t) for t in tokens[5:8]]
            translations.append([tx, ty, tz])
            focal_length = float(tokens[8])
            confidence = float(tokens[9])
            focal_lengths.append(focal_length)
            confidences.append(confidence)

    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(len(rgb_files)):
            name = Path(rgb_files[i]).stem
            qw, qx, qy, qz = rots[i]
            tx, ty, tz = translations[i]
            f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {name}.png\n\n")

def save_point_cloud(pt_file, sparse_dir):
    cloud = o3d.io.read_point_cloud(pt_file, format='xyzrgb')
    pts=np.asarray(cloud.points).reshape(-1,3)[::3]
    colors=np.asarray(cloud.colors).reshape(-1,3)[::3]
    colors = colors.astype(np.uint8)
    # clamp colors in [0,255]
    colors = np.clip(colors, 0, 255)

    save_path = os.path.join(sparse_dir, 'points3D.txt')
    header = """#3D point list with one line of data per point:
#POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
#Number of points: 3, mean track length: 3.3334
"""
    with open(save_path, 'w') as f:
        f.write(header)
        for i, (pt, color) in enumerate(zip(pts, colors)):
            f.write(f"{i} {pt[0]} {pt[1]} {pt[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")

    # save as ply
    save_path = os.path.join(sparse_dir, 'points3D.ply')
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    pct = trimesh.PointCloud(pts, colors=colors)
    pct.vertices_normal = normals
    default_normal = [0, 1, 0]
    vertices = pct.vertices
    colors = pct.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))
    with open(save_path, 'w') as f:
        f.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {int(color[0])} {int(color[1])} {int(color[2])} {normal[0]} {normal[1]} {normal[2]}\n")


if __name__ == '__main__':
    src_dir='./result'
    pose_file=os.path.join(src_dir, 'poses_final.txt')
    pt_file=os.path.join(src_dir, 'point_cloud_out.txt')

    sparse_dir=os.path.join(src_dir, 'sparse')
    if not os.path.exists(sparse_dir):
        os.makedirs(sparse_dir)

    save_cameras(pose_file, sparse_dir)
    save_images_txt(pose_file, sparse_dir)
    save_point_cloud(pt_file, sparse_dir)
import torch  
import numpy as np  
import os  
from scipy.spatial.transform import Rotation as R  
  
data = torch.load('/home/arthur/storage/7SCENES/reconstruction/stairs-seq-01.pth', weights_only=True)  
poses = data['poses']  
tstamps = data['tstamps']  
  
imagedir = '/home/arthur/storage/7SCENES/stairs/test/seq-01-test/'  
stride = 1  
image_list = sorted(os.listdir(imagedir))[::stride]  
  
poses = poses.numpy() if hasattr(poses, 'numpy') else np.array(poses)  
tstamps = tstamps.numpy() if hasattr(tstamps, 'numpy') else np.array(tstamps)  
  
with open('/home/arthur/storage/7SCENES/reconstruction/stairs-seq-01-poses.txt', 'w') as f:  
    for i, (pose, tstamp) in enumerate(zip(poses, tstamps)):  
        original_frame_idx = int(tstamp) 
        original_filename = image_list[original_frame_idx] if original_frame_idx < len(image_list) else f"frame-{original_frame_idx}"  
          
        t = pose[:3]       # tx, ty, tz  
        q = pose[3:]       # qx, qy, qz, qw  
          
        r = R.from_quat(q)  
        R_mat = r.as_matrix()    
          
        pose_mat = np.hstack((R_mat, t.reshape(3,1)))  
        line_vals = pose_mat.flatten()  
          
        frame_name = f"{original_filename.split('.')[0]}" if '.' in original_filename else f"frame-{original_frame_idx:06d}"  
        line_str = ' '.join(f"{v:.6f}" for v in line_vals)  
        f.write(f"{frame_name} {line_str}\n")
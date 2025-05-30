import argparse
import os
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from reloc3r.image_retrieval.topk_retrieval import TopkRetrieval, PREPROCESS_FOLDER, DB_DESCS_FILE_MASK, PAIR_INFO_FILE_MASK
from reloc3r.reloc3r_relpose import Reloc3rRelpose, setup_reloc3r_relpose_model, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.datasets.sevenscenes_retrieval import *  
from reloc3r.datasets.cambridge_retrieval import *  
from eval_relpose import build_dataset
from reloc3r.utils.metric import *
from reloc3r.utils.device import to_numpy

from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser(description='evaluation code for visual localization')

    # model
    parser.add_argument('--model', type=str, 
        default='Reloc3rRelpose(img_size=512)')
    parser.add_argument('--resolution', 
        default=(512,384))

    # test set: process the database
    parser.add_argument('--dataset_db', type=str, 
        default="CambridgeRetrieval(scene='{}', split='train')")
    parser.add_argument('--dataset_q', type=str, 
        default="CambridgeRetrieval(scene='{}', split='test')")
    parser.add_argument('--db_step', type=int, 
        default=1, help='process all database images or skip every db_step images') 
    parser.add_argument('--topk', type=int, 
        default=10, help='topk similar images for motion averaging')
    parser.add_argument('--cache_folder', type=str, default=PREPROCESS_FOLDER)
    parser.add_argument('--db_descs_file_mask', type=str, default=DB_DESCS_FILE_MASK)
    parser.add_argument('--pair_info_file_mask', type=str, default=PAIR_INFO_FILE_MASK)

    # test set: relpose
    parser.add_argument('--dataset_relpose', type=str, 
        default="CambridgeRelpose(scene='{}', pair_id={}, resolution={}, seed=777)")
    parser.add_argument('--batch_size', type=int,
        default=1)
    parser.add_argument('--num_workers', type=int,
        default=10)

    parser.add_argument('--scene', type=str, 
        default='KingsCollege')  
    parser.add_argument('--amp', type=int, 
        default=0,
        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")

    parser.add_argument('--focal', type=float,
        default=525.0,
        help='focal length to append to output lines') 

    return parser


def test(args):
    assert args.scene in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs', 
        'GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch'] 

    if not os.path.exists(args.cache_folder):
        os.mkdir(args.cache_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    args.device = device

    # process database and retrieval
    args.pair_info_available = False
    if not args.pair_info_available:
        args.load_bd_desc = False 
        args.dataset_db = args.dataset_db.format(args.scene)
        args.dataset_q = args.dataset_q.format(args.scene)
        runner = TopkRetrieval(args)
        dataset_db = eval(args.dataset_db)
        runner.build_database(dataset_db)
        dataset_q = eval(args.dataset_q)
        all_retrieved = runner.retrieve_topk(dataset_db, dataset_q)
        pair_info_path = f"{args.cache_folder}/{args.pair_info_file_mask}".format(dataset_q.scene, args.db_step, args.topk)
        np.save(pair_info_path, all_retrieved, allow_pickle=True)
        print(f'Database-query pairs saved to {pair_info_path}.')

    # infer relative poses
    args.relative_pose_available = False
    if not args.relative_pose_available:
        reloc3r_relpose = setup_reloc3r_relpose_model(args.model, device)
        data_loader_test = {
            f"{args.dataset_relpose.split('(')[0]} pair_id={pid}":
            build_dataset(
                args.dataset_relpose.format(args.scene, pid, args.resolution),
                args.batch_size,
                args.num_workers,
                test=True
            )
            for pid in range(args.topk)
        }
        for test_name, testset in data_loader_test.items():
            print(f'Testing {test_name}')
            pose_folder = f"{args.cache_folder}/poses_{testset.dataset.scene}_pair-id={testset.dataset.pair_id}"
            if not os.path.exists(pose_folder):
                os.mkdir(pose_folder)
            with torch.no_grad():
                for batch in tqdm(testset):
                    pose = inference_relpose(batch, reloc3r_relpose, device, use_amp=args.amp)
                    view1, view2 = batch
                    for sid in range(len(pose)):
                        Rt = np.eye(4, dtype=np.float32)
                        Rt[:3,:3] = to_numpy(pose[sid][:3,:3])
                        Rt[:3,3] = to_numpy(pose[sid][:3,3])
                        name_label = view2['label'][sid].split('_')[-1]
                        name_inst = view2['instance'][sid].split('.')[0].split('-')[-1]
                        np.savetxt(f"{pose_folder}/{name_label}_{name_inst}_pose-q2d.txt", Rt)
                        np.savetxt(f"{pose_folder}/{name_label}_{name_inst}_pose-db.txt", to_numpy(view1['camera_pose'][sid]))
                        np.savetxt(f"{pose_folder}/{name_label}_{name_inst}_pose-gt.txt", to_numpy(view2['camera_pose'][sid]))

    # infer absolute poses and save final predictions
    reloc3r_visloc = Reloc3rVisloc()
    rerrs, terrs = [], []
    out_pred = os.path.join(args.cache_folder, f'poses_final.txt')
    fout = open(out_pred, 'w')

    if 'SevenScenes' in args.dataset_q:
        seqs = stat_7Scenes['seqs_test'][args.scene]
        beg, end = 0, stat_7Scenes['n_frames'][args.scene] - 1
        mask_gt_db_cam = mask_gt_db_cam_7Scenes
        mask_q2d_cam = mask_q2d_cam_7Scenes
        mask_gt_q_cam = mask_gt_q_cam_7Scenes
    else:  # Cambridge
        seqs = stat_Cambridge[args.scene]['seq']
        mask_gt_db_cam = mask_gt_db_cam_Cambridge
        mask_q2d_cam = mask_q2d_cam_Cambridge
        mask_gt_q_cam = mask_gt_q_cam_Cambridge

    from scipy.spatial.transform import Rotation as R_scipy

    for seq in seqs:
        if 'Cambridge' in args.dataset_q:
            beg, end = stat_Cambridge[args.scene]['range'][str(seq)]
        for fid in tqdm(range(beg, end+1)):
            if args.scene == 'GreatCourt' and seq == 4 and fid == 73:
                continue
            pose_db, pose_q2d = [], []
            for pid in range(args.topk):
                path_db = f"{args.cache_folder}/poses_{args.scene}_pair-id={pid}/{mask_gt_db_cam}".format(seq, fid)
                path_q2d = f"{args.cache_folder}/poses_{args.scene}_pair-id={pid}/{mask_q2d_cam}".format(seq, fid)
                pose_db.append(np.loadtxt(path_db).reshape(4,4))
                pose_q2d.append(np.loadtxt(path_q2d))
            # gt_q = np.loadtxt(f"{args.cache_folder}/poses_{args.scene}_pair-id={pid}/{mask_gt_q_cam}".format(seq, fid))

            Rt_pred = reloc3r_visloc.motion_averaging(pose_db, pose_q2d)
            # extract quaternion (x,y,z,w) then reorder to qw,qx,qy,qz
            q_xyzw = R_scipy.from_matrix(Rt_pred[:3,:3]).as_quat()
            qx, qy, qz, qw = q_xyzw.tolist()
            tx, ty, tz = Rt_pred[:3,3].tolist()
            filename = f"frame-{fid:06d}.color.png"
            fout.write(
                f"{filename} "
                f"{qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                f"{tx:.10f} {ty:.10f} {tz:.10f} "
                f"{args.focal:.6f}\n"
            )
            # compute error (optional)
            # rerrs.append(get_rot_err(Rt_pred[:3,:3], gt_q[:3,:3]))
            # terrs.append(np.linalg.norm(Rt_pred[:3,3] - gt_q[:3,3]))

    fout.close()
    if rerrs and terrs:
        print(f'Scene {args.scene} median pose error: {np.median(terrs):.2f} m {np.median(rerrs):.2f} deg')
    else:
        print(f'Predictions saved to {out_pred}. No GT errors computed.')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    test(args)

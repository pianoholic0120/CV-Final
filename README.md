# CV-Final

![Pipeline - training data needed](./images/Screenshot%202025-06-07%20at%2018.04.00.png)
![Pipeline - only testing data needed](./images/Screenshot%202025-06-07%20at%2018.05.26.png)

## Table of Contents

- [Requirements & Installation](#requirements--installation)
  - [Reloc3r - generate poses](#reloc3r---generate-poses)
    - [Create the environment using conda](#create-the-environment-using-conda)
    - [Compile cuda kernels for RoPE](#compile-cuda-kernels-for-rope)
  - [ACE0 - generate poses](#ace0---generate-poses)
    - [Create the environment using conda](#create-the-environment-using-conda-1)
    - [C++/Python binding](#cpython-binding)
  - [Main body - 2D images & poses to 3D point cloud](#main-body---2d-images--poses-to-3d-point-cloud)
    - [Create the environment using conda](#create-the-environment-using-conda-2)
- [How to run (examples)](#how-to-run-examples)
  - [To run reloc3r to generate poses](#to-run-reloc3r-to-generate-poses)
  - [To reconstruct based on reloc3r-generated poses](#to-reconstruct-based-on-reloc3r-generated-poses)
  - [To run ACE0 to generate poses](#to-run-ace0-to-generate-poses)
    - [Data Preparation (for a specific sequence)](#data-preparation-for-a-specific-sequence)
    - [ACE0 poses (and point cloud) generation](#ace0-poses-and-point-cloud-generation)
  - [To sweep parameters (voxels, kf) of reconstruction based on ace0-generated poses](#to-sweep-parameters-voxels-kf-of-reconstruction-based-on-ace0-generated-poses)
  - [To refine pose using ACE0](#to-refine-pose-using-ace0)
    - [Data preparation (for a specific sequence)](#data-preparation-for-a-specific-sequence-1)
    - [Poses file converting](#poses-file-converting)
    - [Refinement](#refinement)
  - [To compute accuracy w.r.t GT](#to-compute-accuracy-wrt-gt)
- [Reproduce](#reproduce)
  - [Data Preparation](#data-preparation)
  - [Reconstruction](#reconstruciton)
  - [Further Improvement](#further-improvement)
- [Directory Highlights](#directory-highlights)
- [File Highlights](#file-highlights)
- [Ablation](#ablation)
  - [parameters markdown](#parameters-markdown)
- [External Dependences](#external-dependences)
- [Cited](#cited)

## Requirements & Installation

You should prepare the following two environments (at least two) to run our code.

### Reloc3r - generate poses
 - You can directly follow the desciption of the original [reloc3r github codebase](https://github.com/ffrivera0/reloc3r) for detailed installation setup. 
 
 - However, we made slight modification to the file: 
 eval_visloc_pose.py, ./reloc3r/datasets/sevenscenes_retrieval.py, and ./reloc3r/datasets/sevenscenes.py

#### Create the environment using conda

    conda create -n reloc3r python=3.11 cmake=3.14.0

    conda activate reloc3r 

    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system

    cd ./reloc3r/

    pip install -r requirements.txt

    pip install -r requirements_optional.txt    # install this as well

#### Compile cuda kernels for RoPE

    cd croco/models/curope/

    python setup.py build_ext --inplace

    cd ../../../

- The pre-trained model weights will automatically download when running the evaluation and demo code below.
- Note that the pre-trained weights does not include 7scenes, which is crucial for valid comparison and testing.

### ACE0 - generate poses

- You can directly follow the desciption of the original [ACE0 github codebase](https://github.com/nianticlabs/acezero) for detailed installation setup. 

#### Create the environment using conda

    conda env create -f environment.yml

    conda activate ace0

#### C++/Python binding 

(In order to register cameras to the scene, it relies on the RANSAC implementation of the DSAC* paper, which is written in C++)

    cd ./ace0/acezero/dsacstar/

    python setup.py install

    cd ..

### Main body - 2D images & poses to 3D point cloud

#### Create the environment using conda

    conda env create -f environment.yml -n {scenes310}

    conda activate {scenes310}

    pip install -r requirements.txt


## How to run (examples)

### To run reloc3r to generate poses

First change directory to ./reloc3r/

    cd ./reloc3r/

Specify train/test split under ./reloc3r/datasets/sevenscenes_retrieval.py

Create desired scene under ./data/7scenes/{scene}/ with all seq-{seq_index}/ with *.color.png, *.depth.png, and *.poses.txt, The folder ./data/7scenes/{scene}/ should contain all training sequences and "single" testing sequence to be estimated.

- Note: Testing sequences doesn't need pose.txt of each images.

Run reloc3r pipeline to generate poses_final.py

    python eval_visloc.py --model "Reloc3rRelpose(img_size=512)" --dataset_db "SevenScenesRetrieval(scene='{}', split='train')" --dataset_q "SevenScenesRetrieval(scene='{}', split='test')" --dataset_relpose "SevenScenesRelpose(scene='{}', pair_id={}, resolution={})" --scene "{scene}" --topk 10

- if you come across ModuleNotFoundError: No module named 'h5py', you may pip install it or depend on conda where our version is 3.13.0

- if you come across the issue saying that data must be a sequence (got NoneType) during pose generation of sparse sequences, this is probably due to the error of your data naming format under the testing sequence folder. For instance:

        |-- chess
            |-- seq-05 (sparse, please note the index!!!!!!!!!!!!!)
                |-- frame-000000.color.png
                |-- frame-000000.depth.png
                |-- frame-000000.depth.proj.png
                |-- frame-000000.pose.txt
                |-- frame-000001.color.png
                |-- frame-000001.depth.png
                |-- frame-000001.depth.proj.png
                |-- frame-000002.color.png
                ...


Put the generated poses_final.py into the folder of testing sequence to be evaluated

    cd ..

### To reconstruct based on reloc3r-generated poses:
(the sequence should contain poses_final.txt specifying all the poses in that sequence.)

run:

    python cv25s_full_pipeline_reloc3r.py --seq_dir /path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}/ --output /path/to/the_file/of/{scenes}-seq-{sequence_index}.ply

It will take several minutes to an hour, depends on your number of frames.
### To run ACE0 to generate poses

#### Data Preparation (for a specific sequence)

Make two directory respectively for rgb images and depth(optional) images from the same sequence under same root directory.

e.g. 

    cd ./path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}/

    mkdir ../seq-{sequence_index}-images/

    find . -type f -name "*.color.png" -exec cp {} ../seq-{sequence_index}-images/ \;
    
    # Optional
    mkdir ../seq-{sequence_index}-depth/ 

    find . -type f -name "*.depth.proj.png" -exec cp {} ../seq-{sequence_index}-depth/ \;  

#### ACE0 poses (and point cloud) generation
First change directory to ./ace0/acezero/

    cd ./ace0/acezero/

Run (with & withput depth): 

    # without depth information
    python ace_zero.py "/path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}-images/*.png" ./result/  --export_point_cloud True

    # with depth information
    python ace_zero.py "/path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}-images/*.png" ./result/  --depth_files "/path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}-depth/*.png" --export_point_cloud True

This will generate poses_final.txt and corresponding .ply file of the reconstruction.
- note that it's also feasable to add argument     

        --use_external_focal_length 525.0 
    
    to specify focal length or 
    
        --render_visualization True 
    
    to see visualization of the reconstruction process offline. (note that this visualization argument will sometimes pose errors when using together with the argument --export_point_cloud True)


### To sweep parameters (voxels, kf) of reconstruction based on ace0-generated poses:

    python cv25s_full_pipeline.py --data_root /path/to/your_file/with/7scenes/ --output_dir /path/to/save/your_results/ --voxels 0.0025,0.003 --kf 1,5

### To refine pose using ACE0:

#### Data preparation (for a specific sequence):

Make a directory of rgb images from the desired sequence under same root directory.

e.g. 

    cd ./path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}/

    mkdir ../seq-{sequence_index}-test/

    find . -type f -name "*.color.png" -exec cp {} ../seq-{sequence_index}-images/ \;

#### Poses file converting
To convert your poses_final.txt to each-frame pose estimation to the format following 7scenes dataset.

    python reloc3r_to_ace0.py --poses_dir ./reloc3r_poses/ --output_root /path/to/the_folder/7scenes

#### Refinement
First, train ACE, and freeze the initial estimation of calibrations for 5000 iterations for more stable training.

    python train_ace.py "./path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}-test/*.png" ./results/  --pose_file "./path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}/*.txt" --pose_refinement mlp --pose_refinement_wait 5000 --use_external_focal_length 525.0 --refine_calibration False

Then, generate the final refined pose using:

    python register_mapping.py "./path/to/the_folder/7scenes/{scenes}/test/seq-{sequence_index}-test/*.png" ./results/ --use_external_focal_length 525.0 --session ace_network

### To compute accuracy w.r.t GT:
    
    python acc_comp.py --pred_folder /path/to/your_file/with/point_clouds/ --gt_folder /path/to/your_file/with/gt7scenes/

## Directory Highlights

| Name | Utiliation | 
| :----- | :---- |
ace0_poses | poses txt files generated by ace0 |
ours_best | point cloud we generated without poses imformation, with the highest performance | 
reloc3r_poses | poses txt files generated by reloc3r |
reloc3r_ace0_refined_poses | poses txt files generated by first estimating poses by reloc3r than refining them using ACE0 (the results are not better than solely utilizing reloc3r) | 
sparse_seq_poses | poses txt file for sparse sequences |
sparse_seq_point_cloud | .ply file for point clouds of sparse sequences |

## File Highlights

| Name | Utiliation | 
| :----- | :---- |
cv35s_full_pipelline.py | main function for generated desired point clouds with 2d images and a estimated pose txt files, for ace0 |
cv35s_full_pipelline_reloc3r.py (generate_ply_files.sh) | main function for generated desired point clouds with 2d images and a estimated poses txt file, for reloc3r (single poses_final.txt is needed for each specific sequence)|
generate_gt.py (generate_gt.sh) | for generating gt point cloud, which requires 2D images and corresponding pose txt files (each frame requires a corresponding pose.txt file, which can be generated using reloc3r_to_ace0.py and corresponding poses_final.txt) |
reloc3r_to_ace0.py | can be utilized to generate pose for each frames with poses files in ./ace0_poses/ and ./reloc3r_poses/ |
point_cloud_down_sample.py | down sample point cloud with tunable parameters |
ply_combine.py (ply_combine.sh) | can be used to combine two point clouds with pymeshlab (pymeshlab should be installed additionally) |

## Ablation

| ID | Method | Pre-processing | Post-processing | Focal Length | Depth |Property | Accuracy | Completeness | 
| :-----| ----: | :----: | :-----| ----: | :----: |:----: | :----: |:----: |
| 291723 | GT | ✅  | ✅ | ✅ |✅ | -  |  0.0 | 0.51 |  
| 295646 | GT | ❌  | ❌ | ✅ |✅ | -  | 0.0  | 0.51  |
| 286791 | ACE0 | ❌ | ❌ | ❌ |❌ | Sparse  | 0.22  | 0.31  |
| 287379 | ACE0 | ❌  | ❌ | ✅ | ❌| Dense  |  0.53 | 0.18  |
| 287534 | ACE0 | ❌  | ❌| ✅ |❌ | Sparse  |  0.27 |  0.31 |
| 291376 | COLMAP | ❌  | ❌ | ❌ |❌| Sparse  |  4.28 | 0.4  |
| 291970 | ACE0 | ❌  | ❌ | ❌ | ✅| Sparse  |  0.2 |  0.34 |
| 292226 | ACE0 | ❌  | ✅ (Adaptive)| ❌ | ✅| Sparse  |  0.27 |  0.35 |
| 292839 | ACE0 | ❌  | ✅ (7.5e-3) | ❌ | ✅| Sparse  | 0.23  | 0.34  |
| 295372 | ACE0 | ✅ (20)  | ❌ | ❌ | ❌| Sparse  | 0.24  |  0.38 |
| 295908 | ACE0 | ✅ (Q-align)  | ❌ | ❌ | ❌| Sparse  |  0.26 |  0.34 |
| 296158 | cv25s + ACE0 | ✅ (15)  | ✅ (2.5e-3) | ✅ | ✅| Sparse  | 0.25  |  0.19 |
| 296420 | Combined | ✅  | ✅ | ✅ | ✅| Sparse  | 0.16  |  0.16 |
| 297182 | PCN Refined | ❌  | ✅ | ❌ | ✅| Sparse  | 0.20  |  0.34 |
| 297998 | cv25s + Reloc3r | ❌  | ✅(5e-3)| ✅ | ✅| - (300000) | 0.02  |  0.01 |
| 298700 | spann3r | ❌  | ✅(4e-3)| ❌ | ❌| - (300000) | 0.21  |  0.53 |
| 299961 | cv25s + Reloc3r | ❌  | ✅(2.5e-3)| ✅ | ✅| - (300000) | 0.01  |  0.01 |
| 300466 | cv25s + Reloc3r | ❌  | ✅(2e-3)| ✅ | ✅| - (300000) | 0.01  |  0.01 |
| 300486 | combine 300466 & 291723 | ❌  | ✅(2e-3)| ✅ | ✅| - (300000) | 0.01  |  0.01 |
| 300797 | gt_generate + Reloc3r | ❌  | ✅(1e-2)| ✅ | ✅| -  | 0.01  |  0.01 |

* Pre-processing: kf_every, Q-align
* Post-processing: voxel_grid_size (down-sample), PCN Refinement


### parameters markdown

| ID | Sor_k | Sor_std | r_radius | r_min | voxel_down_sample | # of points | Accuracy | Completeness | 
| :-----| ----: | :----: | :-----| ----: | :----: |:----: | :----: |:----: |
| 297998 | 20  | 2.0| 0.05 | 8| 0.005 |300000 | 0.02  |  0.01 |
| 299961 | 20  | 0.8| 0.03 | 12| 0.0025 |300000 | 0.01  |  0.01 |
| 300466 | 25  | 0.6| 0.025 | 16| 0.002 |300000 | 0.01  |  0.01 |


## External Dependences
1. ACE0
    https://github.com/nianticlabs/acezero
2. Q-Align
    https://github.com/Q-Future/Q-Align
3. COLMAP SfM
    https://github.com/colmap/colmap
4. Reloc3r
    https://github.com/ffrivera0/reloc3r
5. PCN (Point Completion Network)
    https://github.com/wentaoyuan/pcn?tab=readme-ov-file
6. Spann3r
    https://github.com/HengyiWang/spann3r

## Reproduce

### Data Preparation

You should prepare data for reloc3r follow the original [reloc3r github codebase](https://github.com/ffrivera0/reloc3r) and [description](#to-run-reloc3r-to-generate-poses) in my codebase.

Note that you must specify the sequence number for which index of sequences are training data and which is for testing in ./reloc3r/datasets/sevenscenes_retrieval.py: Each time, only one index of testing sequence is given, and only that sequence data (along with all training sequences data) can be put under the directory /home/username/this_repository/reloc3r/data/7scenes/{scene_name}/. Also note the number of frames specifying in the file, you're required to modify when estimated sparse sequences.

    'seqs_train':{
        'chess': [1,2,4,6],
        'fire': [1,2],
        'heads': [2],
        'office': [1,3,4,5,8,10],
        'pumpkin': [2,3,6,8],
        'redkitchen': [1,2,5,7,8,11,13],
        'stairs': [2,3,5,6]
        },
    'seqs_test':{
        'chess': [3], # 5(sparse)
        'fire': [3], # 4(sparse)
        'heads': [1],
        'office': [2], # 6, 7, 9
        'pumpkin': [1], # 7(sparse)
        'redkitchen': [3], # 4, 6, 12, 14
        'stairs': [1] # 4(sparse)
        },
    'n_frames': { # number of frames in training sequence
        'chess': 1000, 
        'fire': 1000, 
        'heads': 1000, 
        'office': 1000, 
        'pumpkin': 1000, 
        'redkitchen': 1000, 
        'stairs': 500
        },
    'n_frames_test': { # number of frames in testing sequence
        'chess': 1000, # 10 for sparse case
        'fire': 1000,  # 10 for sparse case
        'heads': 1000, 
        'office': 1000, 
        'pumpkin': 1000,  # 10 for sparse case
        'redkitchen': 1000, 
        'stairs': 500 # 10 for sparse case
        }

For example, only seq-03 is testing sequence for chess scene, seq-01, seq-02, seq-04, and seq-06 are all training data which have pose ground truth.

    /home/username/this_repository/
    |-- reloc3r
        |-- data/7scenes
            |-- chess
                |-- seq-01 (train structure)
                |-- seq-02 (train structure)
                |-- seq-03 (test structure)
                |-- seq-04 (train structure)
                |-- seq-06 (train structure)
            |-- fire
            |-- heads
            |-- office
            |-- pumpkin
            |-- redkitchen
            |-- stairs

Then, you should be able to generate ./reloc3r/_db-q_pair_info/poses_final.txt for all 14 sequence desired one by one, as [here](#to-run-reloc3r-to-generate-poses) shows. You can also refer to the folder ./reloc3r_poses/ under this repository for all 14 estimated pose for these sequences.

You should be to, then, put these txt files (change their names to poses_final.txt) into their corresponding testing sequences under your storage with 7scenes dataset (You don't actually need to put images under /home/username/storage/7scenes/, this is just an example. Only make sure the directory structure under /home/username/storage/7scenes/ is the same to us is fine).  That is, we assume the 7scenes data you prepared is as follow:

    /home/username/storage/7scenes/
    |-- chess
        |-- test
            |-- seq-03
                |-- frame-000000.color.png
                |-- frame-000000.depth.png
                |-- frame-000000.depth.proj.png
                |-- frame-000000.pose.txt
                |-- frame-000001.color.png
                |-- frame-000001.depth.png
                |-- frame-000001.depth.proj.png
                |-- frame-000002.color.png
                ...
                |-- poses_final.txt
            |-- seq-05 (sparse, please note the index!!!!!!!!!!!!!)
                |-- frame-000000.color.png
                |-- frame-000000.depth.png
                |-- frame-000000.depth.proj.png
                |-- frame-000000.pose.txt
                |-- frame-000001.color.png
                |-- frame-000001.depth.png
                |-- frame-000001.depth.proj.png
                |-- frame-000002.color.png
                ...
                |-- poses_final.txt
        |-- train
            |-- seq-01
                |-- frame-000000.color.png
                |-- frame-000000.depth.png
                |-- frame-000000.depth.proj.png
                |-- frame-000000.pose.txt
                |-- frame-000001.color.png
                |-- frame-000001.depth.png
                |-- frame-000001.depth.proj.png
                |-- frame-000001.pose.txt
                |-- frame-000002.color.png
                ...
            |-- seq-02
            |-- seq-04
            |-- seq-06
        TestSplit.txt
        TrainSplit.txt
    |-- fire
        |-- test
            |-- seq-03
            |-- seq-04 (sparse)
        |-- train
            |-- seq-01
            |-- seq-02
        TestSplit.txt
        TrainSplit.txt
    |-- heads
        |-- test
            |-- seq-01
        |-- train
            |-- seq-02
        TestSplit.txt
        TrainSplit.txt
    |-- office
        |-- test
            |-- seq-02
            |-- seq-06
            |-- seq-07
            |-- seq-09
        |-- Train
            |-- seq-01
            |-- seq-03
            |-- seq-04
            |-- seq-05
            |-- seq-08
            |-- seq-10
        TestSplit.txt
        TrainSplit.txt
    |-- pumpkin
        |-- test
            |-- seq-01
            |-- seq-07 (sparse)
        |-- train
            |-- seq-02
            |-- seq-03
            |-- seq-06
            |-- seq-08
        TestSplit.txt
        TrainSplit.txt
    |-- redkitchen
        |-- test
            |-- seq-03
            |-- seq-04
            |-- seq-06
            |-- seq-12
            |-- seq-14
        |-- train
            |-- seq-01
            |-- seq-02
            |-- seq-05
            |-- seq-07
            |-- seq-08
            |-- seq-11
            |-- seq-13
        TestSplit.txt
        TrainSplit.txt
    |-- stairs
        |-- test
            |-- seq-01
            |-- seq-04 (sparse)
        |-- train
            |-- seq-02
            |-- seq-03
            |-- seq-05
            |-- seq-06
        TestSplit.txt
        TrainSplit.txt

### Reconstruciton

Based on [this](#to-reconstruct-based-on-reloc3r-generated-poses), you can now generated the point cloud one by one. Alternatively, you can modify the bash file (generate_ply_files.sh and generate_sparse_ply_files.sh) for proper paths of yours. Then simply run them to get the 14 testing point clouds and 4 sparse testing point cloud respectively. (📍 be careful to check that in each sequence directory, the file "poses_final.txt" exists.)

    bash generate_ply_files.sh
    bash generate_sparse_ply_files.sh 

This will get 0.01/0.01 in the accuracy/completeness performance results.
### Further Improvement

We can further improve a little by combining points clouds generated above and the ones generated by depth backprojection.

First, generate point clouds with depth backprojection:

    # one-by-one

    python generate_gt.py --sequence_path /path/to/your-files/7scenes/{scene}/test/seq-{sequence-index}/ --ply_path /path/to/file_of/ground_truth/{scene}-seq-{sequence-index}.ply

    # All in one, you should first modify the path in generate_gt.sh

    bash generate_gt.sh

Then, combine the two by running:

    # one-by-one

    python ply_combine.py --input1 /path/to/your_results/test/{scene}-seq-{sequence-index}.ply --input2 /path/to/your_results/gt_generated_in_previous_step/{scene}-seq-{sequence-index}.ply --output /path/to/place/to/store/results/{scene}-seq-{sequence-index}.ply

    # All in one, you should first modify the path in generate_gt.sh

    bash ply_combine.sh

## Cited
ACE0:

    @article{bhat2023zoedepth,
    title={Zoe{D}epth: Zero-shot transfer by combining relative and metric depth},
    author={Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and M{\"u}ller, Matthias},
    journal={arXiv},
    year={2023}
    }

Q-Align:

    @article{wu2023qalign,
    title={Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels},
    author={Wu, Haoning and Zhang, Zicheng and Zhang, Weixia and Chen, Chaofeng and Li, Chunyi and Liao, Liang and Wang, Annan and Zhang, Erli and Sun, Wenxiu and Yan, Qiong and Min, Xiongkuo and Zhai, Guangtai and Lin, Weisi},
    journal={arXiv preprint arXiv:2312.17090},
    year={2023},
    institution={Nanyang Technological University and Shanghai Jiao Tong University and Sensetime Research},
    note={Equal Contribution by Wu, Haoning and Zhang, Zicheng. Project Lead by Wu, Haoning. Corresponding Authors: Zhai, Guangtai and Lin, Weisi.}
    }

COLMAP:

    @inproceedings{schoenberger2016sfm,
        author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title={Structure-from-Motion Revisited},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }

Reloc3r:

    @article{reloc3r,
    title={Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization},
    author={Dong, Siyan and Wang, Shuzhe and Liu, Shaohui and Cai, Lulu and Fan, Qingnan and Kannala, Juho and Yang, Yanchao},
    journal={arXiv preprint arXiv:2412.08376},
    year={2024}
    }

PCN:

    @inProceedings{yuan2018pcn,
    title     = {PCN: Point Completion Network},
    author    = {Yuan, Wentao and Khot, Tejas and Held, David and Mertz, Christoph and Hebert, Martial},
    booktitle = {3D Vision (3DV), 2018 International Conference on},
    year      = {2018}
    }

Spann3r:

    @article{wang20243d,
    title={3D Reconstruction with Spatial Memory},
    author={Wang, Hengyi and Agapito, Lourdes},
    journal={arXiv preprint arXiv:2408.16061},
    year={2024}
    }
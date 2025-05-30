# CV-Final

## By Spann3r

### Installation

1. Create conda environment

   ```
   conda create -n spann3r python=3.9 cmake=3.14.0
   conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia  # use the correct version of cuda for your system
   
   pip install -r requirements.txt
   
   # Open3D has a bug from 0.16.0, please use dev version
   pip install -U -f https://www.open3d.org/docs/latest/getting_started.html open3d
   ```

3. Compile cuda kernels for RoPE

   ```
   cd croco/models/curope/
   python setup.py build_ext --inplace
   cd ../../../
   ```

4. Download the DUSt3R checkpoint

   ```
   mkdir checkpoints
   cd checkpoints
   # Download DUSt3R checkpoints
   wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
   ```

5. Download our [checkpoint](https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy?usp=sharing) and place it under `./checkpoints`

### Config
```
using pretrained model:
1. DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
2. spann3r_101.pth
```
### Run
- Prepare Dataset: \
Place 7SCENES test data in data directory like
```
data
|-- chess
   |-- seq-03
   |-- seq-05
   |-- TestSplit.txt
| -- fire
| -- heads
| -- office
| -- pumpkin
| -- redkitchen
| -- stairs
```
- Running code
```
python eval.py
```

- Output \
Output will occur in the checkpoints directory
```
checkpoints
|-- ckpt_best
   |-- 7scenes
      |-- chess_seq-03-gt.ply    -> psudo ground truth
      |-- chess_seq-03-mask.ply  -> predict
      |-- chess_seq-03.npy
      .
      .
      .
      | logs.txt
```


### Citation

```
@article{wang20243d,
  title={3D Reconstruction with Spatial Memory},
  author={Wang, Hengyi and Agapito, Lourdes},
  journal={arXiv preprint arXiv:2408.16061},
  year={2024}
}
```


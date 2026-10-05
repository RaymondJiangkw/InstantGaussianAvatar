## Instant Expressive Gaussian Head Avatars at Over 100 FPS <br><sub>Official PyTorch implementation of the IEEE ECCV 2026 paper</sub>

<img src="./docs/teaser.jpg" alt="Teaser" style="width:100%;"/>

**Instant Expressive Gaussian Head Avatars at Over 100 FPS**<br>
Kaiwen Jiang, Xueting Li, Seonwook Park, Ravi Ramamoorthi, Shalini De Mello, Koki Nagano<br>

[**Paper**](https://arxiv.org/abs/2512.16893) | [**Project**](https://research.nvidia.com/labs/amri/projects/instant4d) | [**Video**](https://youtu.be/uiUFoE5SoYo) | [**Checkpoints**](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/k1jiang_ucsd_edu/IQCpqR0vq10NToG8hw3r-4obAfLq7OoW68z_2lmIW-SRDZo?e=dinyEt)

Abstract: *Portrait animation has witnessed tremendous quality improvements thanks to recent advances in video diffusion models. However, these 2D methods often compromise 3D consistency and speed, limiting their applicability in real-world scenarios, such as digital twins or telepresence. In contrast, 3D-aware feedforward facial animation methods -- built upon 3D representations, such as neural radiance fields or Gaussian splatting -- ensure 3D consistency and achieve faster inference speed, but come with inferior expression details. In this paper, we address this portrait animation trilemma (speed, 3D consistency, and expressiveness) and propose a pipeline that instantly converts an in-the-wild single image into a 3D-consistent, fast yet expressive animatable representation via a feed-forward encoder. Unlike previous computationally intensive global fusion mechanisms (e.g., multiple attention layers) for fusing 3D structural and animation information, our design employs an efficient lightweight local fusion strategy to achieve high animation expressivity. Furthermore, our animation representation is decoupled from the face's 3D representation and learns motion implicitly from data, eliminating the dependency on pre-defined parametric models that often constrain animation capabilities. Our method runs at 107.31 FPS for animation and pose control, representing a 3-4 order of magnitude speedup versus the state of the art while achieving comparable animation quality, thus surpassing alternative designs that trade speed for quality or vice versa.*

## Requirements
- We have done all the experiments on the Linux platform with NVIDIA 6000 Ada GPUs.
- Dependencies: see [environment.yml](./environment.yml) for exact library dependencies. You need to use the following commands with Miniconda3 to create and activate your Python environment.
```bash
$ conda env create -f environment.yml
$ conda activate instanthead
$ pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3 --no-build-isolation
$ pip install git+https://github.com/mohammadasim98/met3r --no-build-isolation
# met3r overrides packages. Therefore, need to fix that after installation.
$ pip install numpy==1.26.4 opencv-python==4.10.0.84 opencv-python-headless==4.10.0.84 timm==0.9.7
```

## ⭐ Highlights
Besides the checkpoint that reproduces the results reported in the paper, we also provide a newer checkpoint with improved image quality, a wider cropping window, and faster inference.

We recommend using this newer checkpoint by default. The original checkpoint remains available for reproducibility and can be selected by passing the `--legacy` flag.

## Getting started
Pre-trained models could be downloaded from the [link](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/k1jiang_ucsd_edu/IQCpqR0vq10NToG8hw3r-4obAfLq7OoW68z_2lmIW-SRDZo?e=dinyEt). Please put the directory `checkpoints` directly under the main folder.
Besides, you need to download models from the following links to the directory `checkpoints` as well for inference and metrics calculation:
- Download [link](https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view).
- Download [link](https://drive.google.com/file/d/1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H/view).
- Download [link](https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat).
- Download [link](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view) as `face2seg.pth`.
- Download `backbone_ir50_ms1m_epoch120.pth` from [link](https://drive.google.com/drive/folders/1omzvXV_djVIW2A7I09DWMe9JR-9o_MYh).
- Download `emonet_5.pth` and `emonet_8.pth` from [link](https://github.com/face-analysis/emonet/tree/master/pretrained).
- Download `3dmm_assets`, and `pretrained_models/retinaface_resnet50_2020-07-20_old_torch.pth` as `retinaface_resnet50_2020-07-20_old_torch.pth`, `pretrained_models/segment_face.pb` as `segment_face.pb`, `pretrained_models/hrn_v1.1` as `hrn_v1.1`, `pretrained_models/de-retouching.pth` as `de-retouching.pth`, `pretrained_models/large_base_net.pth` as `hrn.pth` from [link](https://drive.google.com/drive/folders/1qI3-5GKxEDbQBSUCwDWmH15uWU3Ka5LS). Copy `3dmm_assets/BFM_model_front.mat` into `BFM_model_front.mat`.
- Download `BBRegressorParam_r.mat` as `3dmm_assets/BBRegressorParam_r.mat` from [link](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/master/util).
- Download the zip following the instructions of [link](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads) and unzip all the content into `3dmm_assets/`.
- Download `weights/mb1_120x120.pth` as `mb1_120x120.pth`, `configs/param_mean_std_62d_120x120.pkl` as `param_mean_std_62d_120x120.pkl`, `configs/bfm_noneck_v3.pkl` as `bfm_noneck_v3.pkl`, `configs/tri.pkl` as `tri.pkl`, `configs/bfm_noneck_v3.pkl` as `bfm_noneck_v3.pkl`, and `configs/mb1_120x120.yml` as `mb1_120x120.yml` from [link](https://github.com/cleardusk/3DDFA_V2/tree/master).

### Quick Example
```python
import torch
from src.utils import *
from src.model import *
from src.third_party.motion_encoder import *

device = "cuda"

state_dict = torch.load("/path/to/checkpoint", map_location=device, weights_only=True)

M = MotEncoder(device)
E = InstantGaussianAvatar().to(device).eval().requires_grad_(False)

M.load_state_dict(state_dict["M_state_dict"])
E.load_state_dict(state_dict["E_state_dict"])

src_image = ... # Tensor with shape (1, 3, 512, 512) in value range (-1, 1)
dri_image = ... # Tensor in value range (-1, 1)

pose = ... # Tensor with shape (1, 25). EG3D Convention.

out = E(src_image, pose) # Reconstrution
rgb = animate_gaussians(E, out['gaussian_dict'], pose, M(src_image), M(dri_image)) # Animation
```

### Inference
Please inspect the `inference.py` script for proper inference, including generating animated images given the source image and driving images with pose control, and measuring the speed.

Example for using the demo images (source image comes from FFHQ, and the driving images come from VOODOO-XP test set):
```bash
$ python inference.py --measure_speed --assume_fixed_crop_box_for_driving_images
```

### Testing
We integrate the calculation of metrics in the `metrics_cal.py` script. We test on the [VOODOO-XP test set](https://github.com/mbzuai-metaverse/voodooxp-official). Inside the `checkpoints` folder, we share the lists `voodooxp_self_reenactment.txt` and `voodooxp_cross_reenactment.txt` for reproducing our experiments. Each file contains a list of pairs of video names. The source image is taken from the first frame of the first video, while the driving images are taken from the frames of the second video.

After establishing the triplet of source image, driving images and predicted images, we provide the `eval.py` script to calculate the metrics.

## Citation
```bibtex
@inproceedings{jiang2026instant,
  title={Instant Expressive Gaussian Head Avatar via 3D-Aware Expression Distillation},
  author={Jiang, Kaiwen and Li, Xueting and Park, Seonwook and Ramamoorthi, Ravi and De Mello, Shalini and Nagano, Koki},
  booktitle={European Conference on Computer Vision},
  year={2026}, 
  organization={Springer}
}
```

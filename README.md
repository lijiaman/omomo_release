# Object Motion Guided Human Motion Synthesis (SIGGRAPH Asia 2023) 
This is the official implementation for the SIGGRAPH Asia 2023 (TOG) [paper](https://arxiv.org/abs/). For more information, please check the [project webpage](https://lijiaman.github.io/projects/omomo/).

![OMOMO Teaser](omomo_teaser.jpg)

## Environment Setup
> Note: This code was developed on Ubuntu 20.04 with Python 3.8, CUDA 11.3 and PyTorch 1.11.0.

Clone the repo.
```
git clone https://github.com/lijiaman/omomo_release.git
cd omomo_release/
```
Create a virtual environment using Conda and activate the environment. 
```
conda create -n omomo_env python=3.8
conda activate omomo_env 
```
Install PyTorch. 
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Install PyTorch3D. 
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```
Install human_body_prior. 
```
git clone https://github.com/nghorbani/human_body_prior.git
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior/
python setup.py develop
```
Install other dependencies. 
```
pip install -r requirements.txt 
```

### Quick Start 
First, download pretrained [models](https://drive.google.com/drive/folders/1llKvkTg0v-eqXGlIrEJYNqUqXAE0x8GK?usp=sharing) and put ```pretrained_models/``` to the root folder.  

If you would like to generate visualizations, please download [Blender](https://www.blender.org/download/) first. And put blender path to blender_path. Replace the blender_path in line 45 of ```omomo_release/manip/vis/blender_vis_mesh_motion.py```. 

Please download [SMPL-H](https://mano.is.tue.mpg.de/download.php) (select the extended SMPL+H model) and put the model to ```smpl_models/smplh_amass/```. If you have a different folder path for SMPL-H model, please modify the path in line 13 of ```egoego/data/amass_diffusion_dataset.py```.

Then run EgoEgo pipeline on the testing data. This will generate corresponding visualization results in folder ```test_data_res/```. To disable visualizations, please remove ```--gen_vis```. 
```
sh scripts/test_egoego_pipeline.sh
```

### Training 
Train stage 1 (generating hand joint position from object geometry).
```
sh scripts/train_stage1.sh
```
Train stage 2 (generating full-body motion from hand joint position). 
```
sh scripts/train_stage2.sh
```

### Evaluation

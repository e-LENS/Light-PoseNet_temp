# SCKD


## Requirements
* LINUX
* Python 3
* CPU or NVIDIA GPU

## Installation
```
git clone https://github.com/e-LENS/SCKD.git
cd SCKD
pip install -r requirements.txt
```

## Dataset
* Outdoor dataset

  Download [King's College dataset](https://www.repository.cam.ac.uk/handle/1810/251342) under `datasets` folder
  
* Indoor dataset

  Download [7 Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) under `datasets` folder

```
datasets/KingsCollege
datasets/Chess
datasets/Fire
datasets/Heads
datasets/Office
datasets/Pumpkin
datasets/RedKitchen
datasets/Stairs
```

* Compute the mean image for each dataset

```
python util/compute_image_mean.py --dataroot datasets/[데이터셋이름] --height 256 --width 455 --save_resized_imgs
```

* 6DoF 전처리

> 7Scenes dataset은 Position & Orientation label을 4x4 matrix 로 제공하나 
> 위 PoseNet은 position(X,Y,Z) & Orientation quaternion(W,P,Q,R) 의 7dimension vector label을 사용해 label값을 변환하는 dataset preprocessing이 필요하다.

해당 github의
`
posenet-pytorch/7scenes_preprocessing.py
`
파일을 이용해 

4x4 => position(X,Y,Z) & Orientation quaternion(W,P,Q,R) 로 label을 변환한다. 


## SCKD

Train & Test the PoseNet model on each dataset


### Pretrained model
    'resnet18im': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34im': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50im': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101im': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152im': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',


### Train Teacher Model 
```
!python train.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name [model]/[Dataset]/[beta500_ex] --beta 500 --gpu 0 --niter 500 --batchSize32 --lr 0.001
```

### Test Teacher Model
```
!python test.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name [model]/[Dataset]/[beta500_ex] --gpu 0
```

### Train Student Model
```
!python KD_train.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name  [student model]/[Dataset]/[beta500_bt_lr_m#1_m#2_scaling] --beta 500 --gpu 0 --niter 500 --T_model [ resnet34 | resnet50 | resnet101]  --T_path [TeacherModel_Path] --save_epoch_freq 5 --SCmodule 3 4 --hintmodule 5 [--SCKL]
```

### Test Student Model
```
!python KD_test.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name Student/[Dataset]/[beta500_LossFunction] --beta 500 --gpu 0 
```

* 학습된 모델 및 학습 결과는 `./checkpoints/[name]`에 저장됨
* 학습된 모델의 테스트 결과는 `./results/[name]`에 저장됨



---
### reference

참고한 repo
- https://github.com/hazirbas/poselstm-pytorch
- https://github.com/HobbitLong/RepDistiller



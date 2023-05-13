# Light-PoseNet


### Train

```
!python train.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name [모델명]/[Dataset]/[beta500_ex] --beta 500 --gpu 0 --niter 500 --batchSize32 --lr 0.001
```

### Test

```
!python test.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name [모델명]/[Dataset]/[beta500_ex] --gpu 0
```

### KD_train

```
!python KD_train.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name  [S_모델명]/[Dataset]/[beta500_bt_lr_m#1_m#2_scaling] --beta 500 --gpu 0 --niter 500 --T_model [ resnet34 | resnet50 | resnet101]  --T_path [TeacherModel_Path] --save_epoch_freq 5 --CSmodule 3 4 --hintmodule 5 [--CSKL]
```

### KD_test

```
!python KD_test.py --model [resnet18 | resnet34 | resnet50 | resnet101] --dataroot [DATAROOT] --name Student/[Dataset]/[beta500_LossFunction] --beta 500 --gpu 0 
```



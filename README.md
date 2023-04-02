# Light-PoseNet



### Train

```
!python train.py --model [posenet/poselstm/onlyStudent] --dataroot [DATAROOT] --name [모델명]/[Dataset]/[beta500_ex] --beta 500 --gpu 0 --niter 300
```

### Test

```
!python test.py --model [posenet/poselstm/onlyStudent] --dataroot [DATAROOT] --name [모델명]/[Dataset]/[beta500_ex] --gpu 0
```

### KD_train

```
!python KD_train.py --model Student --dataroot [DATAROOT] --name  Student/[Dataset]/[beta500_LossFunction] --LossF [KD_MSE/KD_Gaussian] --beta 500 --gpu 0 --niter 300 --T_path [TeacherModel_Path] --save_epoch_freq 5
```

### KD_test

```
!python KD_test.py --model Student --dataroot [DATAROOT] --name Student/[Dataset]/[beta500_LossFunction] --beta 500 --gpu 0 
```

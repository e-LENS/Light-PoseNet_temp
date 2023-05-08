import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import resPoseNet
import pickle
import numpy
from torchinfo import summary


class DistillKL_Feature(nn.Module):
    # Distilling the Knowledge in a ResNet Module (Hint / Guided Module)
    def __init__(self, layerTrans):
        super(DistillKL_Feature, self).__init__()

    def forward(self, feature_S, feature_T):
        loss = 0
        for i in range(len(feature_T)):
            print("feature_S[", i, "].shape :", feature_S[i].shape)
            if feature_T[i].shape[1] != feature_S[i].shape[1]:
                Trans = resPoseNet.Trans_Student(feature_S[i].shape[1], feature_T[i].shape[1])
                feature_S[i] = Trans(feature_S[i])
                print("Trans_feature_S[", i, "].shape :", feature_S[i].shape)

            feature_S[i] = F.normalize(feature_S[i], p=2, dim=1)
            feature_T[i] = F.normalize(feature_T[i], p=2, dim=1)
            feature_S[i] = F.log_softmax(feature_S[i], dim=1)
            feature_T[i] = F.softmax(feature_T[i], dim=1)
            loss += F.kl_div(feature_S[i], feature_T[i], reduction='batchmean')

        # if self.layerTrans=="attention":

        return loss


class Distill_CS(nn.Module):
    def __init__(self, isKL=False):
        super(Distill_CS, self).__init__()
        self.isKL = isKL
        self.MSE = torch.nn.MSELoss()

    def forward(self, CS_T, CS_S):
        loss = 0
        if self.isKL:
            for i in range(len(CS_T)):
                CS_S[i] = F.log_softmax(CS_S[i], dim=1)
                CS_T[i] = F.softmax(CS_T[i], dim=1)
                loss += F.kl_div(CS_S, CS_T, reduction='batchmean')

        else:
            for i in range(len(CS_T)):
                loss += self.MSE(CS_S[i], CS_T[i])
        return loss


class PoseNetModel(BaseModel):
    def name(self):
        return 'PoseNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.isKD = opt.isKD

        self.pretrained = opt.pretrained

        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc)

        # load/define networks
        resnet_weights = None
        if self.isTrain and opt.init_weights != None:
            resnet_file = open(opt.init_weights, "rb")
            resnet_weights = pickle.load(resnet_file, encoding="bytes")
            resnet_file.close()
            print('initializing the weights from ' + opt.init_weights)
        self.mean_image = np.load(os.path.join(opt.dataroot, 'mean_image.npy'))

        self.netG = resPoseNet.define_network(opt.input_nc, opt.model, pretrained=self.pretrained,
                                              init_from=resnet_weights, isTest=not self.isTrain,
                                              isKD=self.isKD,
                                              gpu_ids=self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            if not opt.isKD:  # 얘 추가
                self.load_network(self.netG, 'G', opt.which_epoch)

        summary(self.netG, (1, 3, 224, 224))

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            criterion_MSE = torch.nn.MSELoss()

            if self.isKD:
                self.isKL = opt.KLCS
                self.layerTrans = opt.layerTrans
                criterion_KLfeatures = DistillKL_Feature(opt.layerTrans)
                criterion_CS = Distill_CS(self.isKL)
                criterion = torch.nn.ModuleList([])
                criterion.append(criterion_MSE)
                criterion.append(criterion_KLfeatures)
                criterion.append(criterion_CS)

                self.hintmodule = opt.hintmodule
                self.CSmodule = opt.CSmodule

            else:
                criterion = torch.nn.MSELoss()

            self.criterion = criterion
            self.criterion.cuda()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=0.0625,
                                                betas=(self.opt.adambeta1, self.opt.adambeta2))
            self.optimizers.append(self.optimizer_G)
            # for optimizer in self.optimizers:
            #     self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if self.isKD:

            print("###########  Set Teacher model   ###########")

            self.Teacher = resPoseNet.define_network(opt.input_nc, opt.T_model, pretrained=False,
                                                     init_from=False, isTest=True,
                                                     isKD=self.isKD,
                                                     gpu_ids=self.gpu_ids)
            self.Teacher.load_state_dict(torch.load(opt.T_path))
            summary(self.Teacher, (1, 3, 224, 224))

            # self.hintmodule = opt.hintmodule
            # self.CSmodule = opt.CSmodule
            if isinstance(opt.hintmodule, int):
                self.hintmodule = list(opt.hintmodule)
            else:
                self.hintmodule = opt.hintmodule

            if isinstance(opt.CSmodule, int):
                self.CSmodule = list(opt.CSmodule)
            else:
                self.CSmodule = opt.CSmodule

        print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        # print('-----------------------------------------------')

    def set_Tfeature(self):

        feat_CS = []
        feat_guided = []
        with torch.no_grad():
            self.Teacher.eval()
            feat_t, output_t = self.Teacher(self.input_A)

            for i in self.CSmodule:
                feat_CS.append(feat_t[i - 1])
            for j in self.hintmodule:
                feat_guided.append(feat_t[j - 1])

            self.Tfeat = [f.detach() for f in feat_CS]
            self.Tfeat_hint = [k.detach() for k in feat_guided]


    def set_Sfeature(self):
        feat_CS = []
        feat_guided = []

        for i in self.CSmodule:
            feat_CS.append(self.feat_s[i - 1])
        for j in self.hintmodule:
            feat_guided.append(self.feat_s[j - 1])

        self.Sfeat = feat_CS
        self.Sfeat_guided = feat_guided

    def set_CS(self):
        SS_Tfeat = util.getSelfSimilarity(self.Tfeat)
        SS_Sfeat = util.getSelfSimilarity((self.Sfeat))
        SS_Gt = util.getSelfSimilarity([self.input_B])

        CS_T_Gt = util.getSelfCrossSimilarity(SS_Tfeat, SS_Gt)
        CS_S_Gt = util.getSelfCrossSimilarity(SS_Sfeat, SS_Gt)

        self.CS_T_Gt = CS_T_Gt
        self.CS_S_Gt = CS_S_Gt

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.image_paths = input['A_paths']
        self.batch_index = input['index_N']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def forward(self):
        if self.isKD:
            self.feat_s, self.pred_B = self.netG(self.input_A)
            self.set_Tfeature()
            self.set_Sfeature()
            self.set_CS()
        else:
            self.pred_B = self.netG(self.input_A)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.netG.eval()
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward(self):
        self.loss_Gt = 0
        self.loss_pos = 0
        self.loss_ori = 0

        if self.isKD:
            self.loss_feature = 0
            self.loss_CS = 0

            criterion_MSE, criterion_KL, criterion_CS = self.criterion
            # Gt_Loss
            mse_pos = criterion_MSE(self.pred_B[0], self.input_B[:, 0:3])
            ori_gt = F.normalize(self.input_B[:, 3:], p=2, dim=1)
            mse_ori = criterion_MSE(self.pred_B[1], ori_gt)
            self.loss_Gt = (mse_pos + mse_ori * self.opt.beta)
            self.loss_pos = mse_pos.item()
            self.loss_ori += mse_ori.item() * self.opt.beta

            # Feature_Loss
            self.loss_feature = criterion_KL(self.Sfeat_guided, self.Tfeat_hint)

            # Cross-Similarity Loss
            self.lossCS = criterion_CS(self.CS_T_Gt, self.CS_S_Gt)

            self.loss_G = self.loss_Gt + (self.loss_feature * self.opt.alpha) + (self.lossCS * self.opt.sigma)

        else:
            criterion_MSE = self.criterion
            mse_pos = criterion_MSE(self.pred_B[0], self.input_B[:, 0:3])
            ori_gt = F.normalize(self.input_B[:, 3:], p=2, dim=1)
            mse_ori = criterion_MSE(self.pred_B[1], ori_gt)
            self.loss_G = (mse_pos + mse_ori * self.opt.beta)
            self.loss_pos = mse_pos.item()
            self.loss_ori += mse_ori.item() * self.opt.beta

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def get_current_loss(self):
        if self.opt.isKD:
            return OrderedDict([('loss_Gt', self.loss_Gt),
                           ('loss_feature', self.loss_feature),
                           ('loss_CS', self.lossCS)])
        else:
            return OrderedDict([('loss_Gt', self.loss_Gt)])

    def get_current_errors(self):
        if self.opt.isTrain:
            return OrderedDict([('pos_err', self.loss_pos),
                                ('ori_err', self.loss_ori),
                                ])

        pos_err = torch.dist(self.pred_B[0], self.input_B[:, 0:3])
        ori_gt = F.normalize(self.input_B[:, 3:], p=2, dim=1)
        abs_distance = torch.abs((ori_gt.mul(self.pred_B[1])).sum())
        ori_err = 2 * 180 / numpy.pi * torch.acos(abs_distance)
        return [pos_err.item(), ori_err.item()]

    def get_current_pose(self):
        return numpy.concatenate((self.pred_B[0].data[0].cpu().numpy(),
                                  self.pred_B[1].data[0].cpu().numpy()))

    def get_current_visuals(self):
        input_A = util.tensor2im(self.input_A.data)
        # pred_B = util.tensor2im(self.pred_B.data)
        # input_B = util.tensor2im(self.input_B.data)
        return OrderedDict([('input_A', input_A)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
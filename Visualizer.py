import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
#from util.util import getSelfSimilarity
import numpy as np
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from os.path import join as jpath


#import torchshow as ts


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 300  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.isKD = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

model = create_model(opt)
testepoch = 32


def getSelfSimilarity(input_m):
    # shape : N*256*256*3 => N*(256*256*3)
    sim = []
    temp = (input_m.view([input_m.shape[0], -1]).squeeze())
    # F.pdist => 1*(_nC_2) //
    sim.append(pairwise_cosine_similarity(temp))  # N*N

    return sim

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = value - leftMin/leftSpan

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


for i, data in enumerate(dataset):

    model.netG.load_state_dict(torch.load("./TeacherBestWeight/Teacher_Resnet34/Heads_Resnet34.pth"))

    model.set_input(data)
    model.test()
    feature, pred = model.pred_B
    print(len(feature[0]))

    if (i == 0):

        input_B = model.input_B.cpu()
        act_1 = feature[0].cpu()
        act_2 = feature[1].cpu()
        act_3 = feature[2].cpu()
        act_4 = feature[3].cpu()
        act_5 = feature[4].cpu()
        act_6 = feature[5].cpu()
        print(pred[0].shape)
        print(pred[1].shape)
        pred = torch.cat([pred[0], pred[1]], dim = 1)
        print(pred.shape)

        # print(len(input_B))
        # img_Max = torch.max(model.input_A[0]).cpu()
        # img_Min = torch.min(model.input_A[0]).cpu()
        # gt_Max = torch.max(model.input_B).cpu()
        # gt_Min = torch.min(model.input_B).cpu()
        #input_B = translate(input_B, gt_Min, gt_Max, img_Min, img_Max)

        image_Similarity = getSelfSimilarity(model.input_A)
        gt_Similarity = getSelfSimilarity(input_B)
        act_1_Similarity = getSelfSimilarity(act_1)
        act_2_Similarity = getSelfSimilarity(act_2)
        act_3_Similarity = getSelfSimilarity(act_3)
        act_4_Similarity = getSelfSimilarity(act_4)
        act_5_Similarity = getSelfSimilarity(act_5)
        act_6_Similarity = getSelfSimilarity(act_6)
        pred_Similarity = getSelfSimilarity(pred)

        img_Sim = image_Similarity[0]
        gt_Sim = gt_Similarity[0]
        act_1_Sim = act_1_Similarity[0]
        act_2_Sim = act_2_Similarity[0]
        act_3_Sim = act_3_Similarity[0]
        act_4_Sim = act_4_Similarity[0]
        act_5_Sim = act_5_Similarity[0]
        act_6_Sim = act_6_Similarity[0]
        pred_Sim = pred_Similarity[0]

        #img_Max = torch.max(img_Sim).cpu()
        #img_Min = torch.min(img_Sim).cpu()

        #gt_Max = torch.max(gt_Sim).cpu()
        #gt_Min = torch.min(gt_Sim).cpu()

        #scaled_gt_Sim = translate(gt_Sim, img_Min, img_Max, gt_Min, gt_Max)

        #print(scaled_gt_Sim)

        img_Sim = img_Sim.cpu().numpy()
        #scaled_gt_Sim = scaled_gt_Sim.cpu().numpy()

        np.save(jpath('./Image/heads', 'img.npy'), img_Sim)
        np.save(jpath('./Image/heads', 'gt.npy'), gt_Sim)
        np.save(jpath('./Image/heads', 'act1.npy'), act_1_Sim)
        np.save(jpath('./Image/heads', 'act2.npy'), act_2_Sim)
        np.save(jpath('./Image/heads', 'act3.npy'), act_3_Sim)
        np.save(jpath('./Image/heads', 'act4.npy'), act_4_Sim)
        np.save(jpath('./Image/heads', 'act5.npy'), act_5_Sim)
        np.save(jpath('./Image/heads', 'act6.npy'), act_6_Sim)
        np.save(jpath('./Image/heads', 'pred.npy'), pred_Sim)

        break










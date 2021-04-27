# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import cv2
import torch.nn.functional as F
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/ddrnet23_slim.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        # model_state_file = os.path.join(final_output_dir, 'best_0.7589.pth')
        model_state_file = os.path.join(final_output_dir, 'best.pth')    
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load('/home/kong/Documents/DDRNet.Pytorch/DDRNet.Pytorch/output/face/ddrnet23_slim/checkpoint.pth.tar')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

  #  print(pretrained_dict.keys())
    new_state = {k:v for k,v in pretrained_dict.items() if k in model.state_dict()}

    model.load_state_dict(new_state)
    model = model.cuda()

    torch.save(model.state_dict(), 'model_best_bacc.pth.tar', _use_new_zipfile_serialization=False)


   # gpus = list(config.GPUS)
  #  model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    print(test_size)
    img = cv2.imread("/home/kong/Downloads/731509a96420ef3dd0cffe869a4a53cb.jpeg")
 
    img = cv2.resize(img,(512,512))
    image = img.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std

    image = image.transpose((2,0,1))
    image = torch.from_numpy(image)
    
  #  image = image.permute((2, 0, 1))
    
    print(image.shape)
    image = image.unsqueeze(0)

    image = image.cuda()
    start = time.time()
    print("input : ",image)
    for i in range(1):
        out= model(image)
    end = time.time()
    print("Cuda 1000 images inference time : ",1000.0/(end - start))
    outadd = out[0]*1.0 + out[1]*0.4
    out0 = out[0].squeeze(dim=0)
    out1 = out[1].squeeze(dim=0)

    print(out0.size(),out0[0,1,1],out0[1,1,1])
    print("out:",out0)


    outadd = outadd.squeeze(dim=0)
    out0 = F.softmax(out0,dim=0)
    out1 = F.softmax(out1,dim=0)
    outadd = F.softmax(outadd,dim=0)

    out0 = torch.argmax(out0,dim=0)
    out1 = torch.argmax(out1,dim=0)
    outadd = torch.argmax(outadd,dim=0)

    pred0 = out0.detach().cpu().numpy()
    pred1 = out1.detach().cpu().numpy()
    predadd = outadd.detach().cpu().numpy()
    pred0 = pred0*255
    pred1 = pred1*255
    predadd = predadd*255

    pred_ch = np.zeros(pred0.shape)
    pred_rgb0 = np.array([pred_ch,pred_ch,pred0])
    pred_rgb1 = np.array([pred_ch,pred_ch,pred1])
    pred_rgbadd = np.array([predadd,pred_ch,predadd])
    pred_rgb0 = pred_rgb0.transpose(1,2,0)
    pred_rgb1 = pred_rgb1.transpose(1,2,0)
    pred_rgbadd = pred_rgbadd.transpose(1,2,0)
    pred_rgb0 = cv2.resize(pred_rgb0,(img.shape[1],img.shape[0]))
    pred_rgb1 = cv2.resize(pred_rgb1,(img.shape[1],img.shape[0]))
    pred_rgbadd = cv2.resize(pred_rgbadd,(img.shape[1],img.shape[0]))
    dst=cv2.addWeighted(img,0.7,pred_rgb0.astype(np.uint8),0.3,0)
    dst1=cv2.addWeighted(img,0.7,pred_rgb1.astype(np.uint8),0.3,0)
    dstadd=cv2.addWeighted(img,0.7,pred_rgbadd.astype(np.uint8),0.3,0)
    
    imgadd = np.vstack((img,pred_rgb0,dst,pred_rgb1, dst1,pred_rgbadd, dstadd))


    cv2.imwrite("a242.jpg",imgadd)




if __name__ == '__main__':
    main()

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

def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """
 
    b_channel, g_channel, r_channel = cv2.split(img) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道
 
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    return img_new
 
def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    """ 将png透明图像与jpg图像叠加 
        y1,y2,x1,x2为叠加位置坐标值
    """
    
    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
    
    '''
    当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
    这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
    '''
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]
 
    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]
 
    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[yy1:yy2,xx1:xx2,3] / 255.0
    alpha_jpg = 1 - alpha_png
    
    # 开始叠加
    for c in range(0,3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg*jpg_img[y1:y2,x1:x2,c]) + (alpha_png*png_img[yy1:yy2,xx1:xx2,c]))
 
    return jpg_img

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
    img = cv2.imread("/home/kong/Documents/DDRNet.Pytorch/DDRNet.Pytorch/images/418cd0c0b416d93bc5a129834537f2e1.jpeg")
 
    img = cv2.resize(img,(512,512))
    image = img.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std

    image = image.transpose((2,0,1))
    image = torch.from_numpy(image)
    
  #  image = image.permute((2, 0, 1))
    
 #   print(image.shape)
    image = image.unsqueeze(0)

    image = image.cuda()
    start = time.time()
 #   print("input : ",image)
    for i in range(1):
        out= model(image)
    end = time.time()
    #print("Cuda 1000 images inference time : ",1000.0/(end - start))
    outadd = out[0]*1.0 + out[1]*0.4
    out0 = out[0].squeeze(dim=0)
    out1 = out[1].squeeze(dim=0)

   # print(out0.size(),out0[0,1,1],out0[1,1,1])
  #  print("out:",out0)


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


####================= alpha channel =========================#
    print("pred0:",pred0.shape, img.shape)
    pred0 = np.array(pred0,np.uint8)
    pred0up = cv2.resize(pred0,(512,512))
    png = np.dstack((img,pred0up))
    bg = cv2.imread("/home/kong/Downloads/4aeb26a89778f73261ccef283e70992f.jpeg")

    addpng = merge_img(bg,png,500,1012,100,612)
    cv2.imwrite("png.png",addpng)

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
    
    imgadd = np.hstack((img,dstadd))


    cv2.imwrite("a242.jpg",imgadd)




if __name__ == '__main__':
    main()

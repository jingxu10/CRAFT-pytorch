"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT
from CLEval.script import validate

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

class CRAFT_Dataset(Dataset):
    def __init__(self, test_folder, canvas_size, mag_ratio):
        self.image_list, _, _ = file_utils.get_files(test_folder)
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = imgproc.loadImage(self.image_list[idx])
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        # x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        return x,1

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='models/craft_ic15_20k.pth', type=str, help='pretrained model')
parser.add_argument('--gt_file', default='CLEval/gt/gt_IC15.zip', type=str, help='ground truth files')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--quant', default=False, action='store_true', help='tune with int8')
parser.add_argument('--int8', default=False, action='store_true', help='quantize with INT8')
parser.add_argument('--jit', default=False, action='store_true', help='run with jit')
parser.add_argument('--ipex', default=False, action='store_true', help='run with ipex')

args = parser.parse_args()
refine_net = None

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None, conf=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    t00 = time.time()
    # forward pass
    with torch.no_grad():
        if args.ipex:
            x = x.to('dpcpp')
            if conf:
                with ipex.AutoMixPrecision(conf, running_mode='inference'):
                    y, feature = net(x)
            else:
                y, feature = net(x)
        else:
            y, feature = net(x)
    t_delta = time.time() - t00

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text, t_delta

def inference(net, conf=None):
    t_sum = 0
    count = 0
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text, t_delta = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net, conf)
        t_sum = t_sum + t_delta

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
        count = count + 1
    print("ave inf time/f: {}s/f | {}".format(t_sum/count, count))

def ilit_test(net):
    inference(net)
    outZip = zipfile.ZipFile('result.zip', mode='w', allowZip64=True)
    for file in os.listdir('result'):
        if file.endswith('.txt'):
            outZip.write(os.path.join('result/', file), file, zipfile.ZIP_DEFLATED)
    outZip.close()
    resDict = validate(args.gt_file)
    os.remove('result.zip')
    precision = resDict['method']['Detection']['precision']
    recall = resDict['method']['Detection']['recall']
    f1 = 2 * (precision*recall) / (precision+recall)
    return f1

if __name__ == '__main__':
    if args.ipex:
        try:
            import intel_pytorch_extension as ipex
            print("ipex imported")
        except:
            print("importing ipex failed. Quit.")
            exit(1)
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    if args.quant:
        from lpot import Quantization
        dataset = CRAFT_Dataset(args.test_folder, args.canvas_size, args.mag_ratio)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        if args.ipex:
            quantizer = Quantization("./config_ipex.yaml")
        else:
            net.eval()
            net.fuse()
            quantizer = Quantization("./config.yaml")
        net = quantizer(net, dataloader, eval_func=ilit_test)
        exit(1)

    if args.ipex:
        net = net.to('dpcpp')
        if args.int8:
            ipex_config_path = os.path.join(os.path.expanduser('./lpot_workspace/pytorch_ipex/craft_ocr/checkpoint/'), "best_configure.json")
            conf = ipex.AmpConf(torch.int8, configure_file=ipex_config_path)
        else:
            conf = ipex.AmpConf(None)
    if args.jit:
        try:
            net = torch.jit.script(net)
        except:
            d = torch.randn(1, 3, 720, 1280)
            if args.ipex:
                d = d.to('dpcpp')
            net = torch.jit.trace(net, d)

    # LinkRefiner
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()
    inference(net, conf)
    print("elapsed time : {}s".format(time.time() - t))

import argparse
import time
from pathlib import Path

import math
import pytesseract
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.deskew_plate import deskew

from IPython.display import Image
from matplotlib import pyplot as plt

import cv2
import sys
import numpy as np
import os.path

YOLO_MODEL_PATH = f"D:/ANPR/ANPR-ML/best.pt"


def load_model(opt, save_img=False):
    weights, view_img, save_txt, imgsz, trace = opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    return model


def detect(opt, model, img_path, save_img=False):
    # list of number plates
    num_plates = []
    source, weights, view_img, save_txt, imgsz, trace = img_path, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith(
        '.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # # Directories
    # save_dir = Path(
    #     increment_path(Path(opt.project) / opt.name,
    #                    exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(
    #     parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # # Load model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0]
                                     or old_img_h != img.shape[2]
                                     or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad(
        ):  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred,
                                   opt.conf_thres,
                                   opt.iou_thres,
                                   classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + (
            # '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1,
                                          0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh

                    b_box = []
                    for cord in xyxy:
                        b_box.append(int(cord.item()))
                    num_plates.append((conf,b_box))

                    line = (cls, *xywh,
                            conf) if opt.save_conf else (cls,
                                                         *xywh)  # label format
                    # with open(txt_path + '.txt', 'a') as f:
                    # f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy,
                                 im0,
                                 label=label,
                                 color=colors[int(cls)],
                                 line_thickness=1)

            # Print time (inference + NMS)
            print(
                f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS'
            )

    print(f'Done. ({time.time() - t0:.3f}s)')
    f_conf=0
    f_num_plate=None
    for (conf, num_plate) in num_plates:
        if conf>f_conf:
            f_conf=conf
            f_num_plate=num_plate

    return f_num_plate

class Args:
    weights=YOLO_MODEL_PATH
    img_size=640
    conf_thres=0.25
    iou_thres=0.45
    device=""
    view_img=""
    save_txt=""
    save_conf=""
    nosave=""
    classes=0
    agnostic_nms=""
    augment=""
    update=""
    project=""
    name="exp"
    exist_ok=""
    no_trace=""
    
def getNumberPlateRegion(model,img_path):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights',
    #                     nargs='+',
    #                     type=str,
    #                     default=YOLO_MODEL_PATH,
    #                     help='model.pt path(s)')
    # # parser.add_argument('--source', type=str, default=img_path,
    # #                     help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size',
    #                     type=int,
    #                     default=640,
    #                     help='inference size (pixels)')
    # parser.add_argument('--conf-thres',
    #                     type=float,
    #                     default=0.25,
    #                     help='object confidence threshold')
    # parser.add_argument('--iou-thres',
    #                     type=float,
    #                     default=0.45,
    #                     help='IOU threshold for NMS')
    # parser.add_argument('--device',
    #                     default='',
    #                     help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img',
    #                     action='store_true',
    #                     help='display results')
    # parser.add_argument('--save-txt',
    #                     action='store_true',
    #                     help='save results to *.txt')
    # parser.add_argument('--save-conf',
    #                     action='store_true',
    #                     help='save confidences in --save-txt labels')
    # parser.add_argument('--nosave',
    #                     action='store_true',
    #                     help='do not save images/videos')
    # parser.add_argument('--classes',
    #                     nargs='+',
    #                     type=int,
    #                     help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms',
    #                     action='store_true',
    #                     help='class-agnostic NMS')
    # parser.add_argument('--augment',
    #                     action='store_true',
    #                     help='augmented inference')
    # parser.add_argument('--update',
    #                     action='store_true',
    #                     help='update all models')
    # parser.add_argument('--project',
    #                     default='runs/detect',
    #                     help='save results to project/name')
    # parser.add_argument('--name',
    #                     default='exp',
    #                     help='save results to project/name')
    # parser.add_argument('--exist-ok',
    #                     action='store_true',
    #                     help='existing project/name ok, do not increment')
    # parser.add_argument('--no-trace',
    #                     action='store_true',
    #                     help='don`t trace model')

    # opt = parser.parse_args(args=[])
    opt=Args()


    # try:
    #     opt = parser.parse_args() #call from command line
    # except:
    #     opt = parser.parse_args(args=[YOLO_MODEL_PATH, 640]) #call from notebook

    img = cv2.imread(img_path)
    #load the model
    yolo_model = model
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            num_plate = detect(opt, yolo_model, img_path)
            
            region = img[num_plate[1]:num_plate[3],
                            num_plate[0]:num_plate[2]].copy()
            return region
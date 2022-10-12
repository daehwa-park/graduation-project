import RPi.GPIO as GPIO
import time,threading
import numpy as np
from time import sleep

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadWebcam
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

OUTPUT = 1
INPUT = 0

HIGH = 1
LOW = 0

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

trig = 13
echo = 19

GPIO.setup(trig, GPIO.OUT)
GPIO.setup(echo, GPIO.IN)

@torch.no_grad()
def run(
        weights=ROOT / 'best.pt',  # model.pt path(s)
        source= 0,  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    tim = 0
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = Path(project) / name  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
       view_img = check_imshow()
       cudnn.benchmark = True  # set True to speed up constant image size inference
       dataset = LoadWebcam(pipe = '0', img_size=imgsz, stride=stride)
       bs = len(dataset)  # batch_size
    else:
       dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
       bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        if tim == 3:
            break
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            txt_path = str(save_dir / 'labels' / p.stem)
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        line = cls
                        with open(f'{txt_path}.txt', 'w') as f:
                            #f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            f.write(('%g ').rstrip() % line)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                fps, w, h = 30, im0.shape[1], im0.shape[0]
        
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        tim += 1
        

def foo():
    print(time.ctime())
    threading.Timer(5, back_Origin).start()
def servoMotor(pin, degree, t):
    GPIO.setmode(GPIO.BCM) 
    GPIO.setup(pin, GPIO.OUT) 
    pwm=GPIO.PWM(pin, 50) 
    pwm.start(0) 
    time.sleep(1) 
    pwm.ChangeDutyCycle(degree) 
    time.sleep(t)
    pwm.stop()
    GPIO.cleanup(pin)
def servoMotor_Origin(pin, t):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT) 
    pwm=GPIO.PWM(pin, 50)
    pwm.start(7) 
    time.sleep(t) 
    pwm.stop()
    GPIO.cleanup(pin)
       
def main() :
    file_w = open('/home/pi/project/yolov5/runs/detect/exp/labels/w.txt', 'w')
    file_w.write('0')
    file_w.close()

    run()
    cv2.destroyAllWindows()

    #num = input("number")
    file = open('/home/pi/project/yolov5/runs/detect/exp/labels/w.txt', 'r') #number.txt
    file_number = file.read()

    print(int(file_number))

    if int(file_number) == 7:  #plastic
        servoMotor(21, 8.5, 1)
    elif int(file_number) == 6: #can
        servoMotor(21, 5.5, 1)
    elif int(file_number) == 3 or int(file_number) == 4 :  #paper
        servoMotor(12, 8.5, 1)
    elif int(file_number) == 1: #glass
        servoMotor(12, 5.5, 1)
    elif int(file_number) == 0: #else
        sleep(1)

    servoMotor(24, 10.5, 6)   #360 servo

    servoMotor(21, 12, 1)
    servoMotor(12, 2.8, 1)

    file.close()



while True:
    GPIO.output(trig,False)
    time.sleep(0.5)
    
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)
    
    while GPIO.input(echo) == 0 :
        pulse_start = time.time()
        
    while GPIO.input(echo) == 1 :
        pulse_end = time.time()
        
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17000
    distance = round(distance, 2)
    
    print("distance : %.1f cm" % distance)
    
    if distance <= 5.0 :
        main()
        time.sleep(3)
        
GPIO.cleanup()




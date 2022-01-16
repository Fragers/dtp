import copy
import os
import time
import torch.nn as nn
import albumentations
from albumentations import pytorch as AT
import torch
from efficientnet_pytorch import EfficientNet
import timm
import cv2
import numpy as np

import cv2
from PIL import Image
import gc
import torch

import argparse
from my_utlis.annotator import MyAnnotator
from my_utlis.models import CarClassifier, CustomResnext, DtpClassifier

dtp_videos = [
    2,
    4,
    5,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    21,
    22,
    24,
    27,
    33,
    36,
    50
]


def main(args):

    w, h = (args.width, args.height)

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=args.yolo_weights, source='local')
    model.conf = 0.5
    # C:\Users\igors/.cache\torch\hub\ultralytics_yolov5_master
    # C:\\Users\\igors/.cache\\torch\\hub\\ultralytics_yolov5_master
    # model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/yolo_weights.pt')
    # model.conf = args.yolo_conf
    gc.collect()
    car_classfier = CarClassifier('../weights/eff_model_car.pth')

    # dtp_classifier = DtpClassifier('../weights/trans_224.pth')
    acc = 0
    all_dtp_cnt = 0
    for video_number, video_file in enumerate(os.listdir(args.video_dir)):

        cap = cv2.VideoCapture(os.path.join('../data/video', video_file))
        video_id = video_file.split('.')[0]
        cnt_dtp = 0
        cnt_not_dtp = 0
        num_frame = 0
        obr_frame = 0
        while cap.isOpened():
            # читаем кадр из видео
            ret, frame = cap.read()
            t = time.time()
            num_frame += 1
            with_dtp = False
            if num_frame % 30 != 0:
                continue
            if ret is True and frame.size != 0:
                obr_frame += 1
                frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = model(result, size=1280)
                dets = []
                if not results.pandas().xyxy[0].empty:
                    for res in range(len(results.pandas().xyxy[0])):
                        r = results.pandas().xyxy[0].to_numpy()[res]
                        # print(r)

                        dets.append(list(r))

                np_dets = np.array(dets)
                for i in range(len(dets)):
                    x0, y0, x1, y1, conf, cls_id, cls_name = dets[i]
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                    MyAnnotator.rectangle(frame, (x0, y0), (x1, y1))
                    MyAnnotator.put_text(frame,
                                         cls_name,
                                         (x0, y1))
                    if cls_name not in ['car', 'truck', 'bus', 'boat', 'motorcycle']:
                        continue
                    # MyAnnotator.put_text(frame, cls_name + " " + str(round(conf, 2)), (x0, y0))
                    crop = frame_copy[y0:y1, x0:x1]
                    car_name, conf_cls, np_pred = car_classfier.infer(crop)
                    if car_name != 'car':
                        # cnt_dtp += 1
                        with_dtp = True
                    else:
                        cnt_not_dtp += 1
                    # print(num_frame)
                if with_dtp:
                    cnt_dtp += 1
            else:
                break

        if cnt_dtp >= 7 or cnt_dtp >= 1/3 * obr_frame:
            pred = 'dtp'
        else:
            pred = 'not_dtp'
        if pred == 'dtp':
            all_dtp_cnt += 1
        if int(video_id) in dtp_videos:
            true_label = 'dtp'
        else:
            true_label = 'not_dtp'

        acc += int(true_label == pred)
        print(video_number + 1, '/', len(os.listdir('../data/video')), video_id, round(acc / (video_number + 1), 2)
              , '\r\n', 'true: ', true_label
              , '\r\n', '|', 'pred:', pred
              , '\r\n', '|', cnt_dtp
              , '\r\n', '|', obr_frame,
              )

    print(acc)
    print(all_dtp_cnt)
    # print(time.time() - t)

    cv2.destroyAllWindows()


# torch.nn.Module.dump_patches = True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--car_classifier_weights", default='../weights/eff_model_car.pth')
    parser.add_argument("--yolo_weights", default='../weights/yolov5m6.pt')
    parser.add_argument("--save_video", default=True)
    parser.add_argument("--out_video_path", default='../inference/out.avi')
    parser.add_argument("--in_video_path", default='../data/video/50.mp4')
    parser.add_argument("--sort_max_age", default=5)
    parser.add_argument("--sort_min_hits", default=2)
    parser.add_argument("--sort_iou_thresh", default=0.2)
    parser.add_argument("--yolo_conf", default=0.6)
    parser.add_argument("--width", default=640)
    parser.add_argument("--height", default=480)
    parser.add_argument("--video_dir", default='../data/video')

    # track_type
    args = parser.parse_args()
    main(args)

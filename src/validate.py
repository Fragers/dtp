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
    cur_frame = 0
    w, h = (args.width, args.height)
    central_point = (w / 2, h / 2)
    use_stream = args.stream
    streamer = None
    stream_name = args.stream_name

    dtp_classifier = DtpClassifier('../weights/trans_224.pth')
    acc = 0
    all_dtp_cnt = 0
    for video_number, video_file in enumerate(os.listdir('../data/video')):

        cap = cv2.VideoCapture(os.path.join('../data/video', video_file))
        video_id = video_file.split('.')[0]
        cnt_dtp = 0
        cnt_not_dtp = 0
        num_frame = 0
        while cap.isOpened():
            # читаем кадр из видео
            ret, frame = cap.read()
            t = time.time()
            num_frame += 1
            if num_frame % 30 != 0:
                continue
            if ret is True and frame.size != 0:
                frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dtp_name, conf_cls, np_pred = dtp_classifier.infer(frame_copy)

                # MyAnnotator.put_text(frame,
                #                      dtp_name + ' ' + str(round(conf_cls, 2)),
                #                      (20, 20))

                # cv2.imshow("Demo", frame)
                # time.sleep(0.1)
                if dtp_name == 'dtp':
                    cnt_dtp += 1
                else:
                    cnt_not_dtp += 1
                # print(num_frame)
            else:
                break

        if cnt_dtp >= 3:
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
              )

    print(acc)
    print(all_dtp_cnt)
    # print(time.time() - t)

    cv2.destroyAllWindows()


# torch.nn.Module.dump_patches = True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_weights", default='weights/resnext(1).pth')
    parser.add_argument("--yolo_weights", default='../weights/yolov5m6.pt')
    parser.add_argument("--save_video", default=True)
    parser.add_argument("--out_video_path", default='../inference/out.avi')
    parser.add_argument("--in_video_path", default='../data/video/50.mp4')

    parser.add_argument("--stream", default=True)
    parser.add_argument("--stream_name", default='igor_test')
    parser.add_argument("--only_central", default=False)
    parser.add_argument("--sort_max_age", default=5)
    parser.add_argument("--sort_min_hits", default=2)
    parser.add_argument("--sort_iou_thresh", default=0.2)
    parser.add_argument("--classifier_padding", default=15)
    parser.add_argument("--yolo_conf", default=0.6)
    parser.add_argument("--window_size", default=7)
    parser.add_argument("--clear_tracks_every", default=500)
    parser.add_argument("--track_type", default="window", help="Possible tracks window, exp, base")
    parser.add_argument("--track_alpha", default=0.5)
    parser.add_argument("--width", default=640)
    parser.add_argument("--height", default=480)

    # track_type
    args = parser.parse_args()
    main(args)

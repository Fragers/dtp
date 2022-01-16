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
from my_utlis.models import CarClassifier, CustomResnext


def main(args):
    model = torch.hub.load('C:\\Users\\igors/.cache\\torch\\hub\\ultralytics_yolov5_master', 'custom',
                           path=args.yolo_weights, source='local')
    model.conf = 0.5
    # C:\Users\igors/.cache\torch\hub\ultralytics_yolov5_master
    # model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/yolo_weights.pt')
    # model.conf = args.yolo_conf
    gc.collect()

    torch.cuda.empty_cache()

    cur_frame = 0
    w, h = (args.width, args.height)
    central_point = (w / 2, h / 2)
    use_stream = args.stream
    streamer = None
    stream_name = args.stream_name
    for filename in os.listdir('../data/cam_cars'):
        frame = cv2.imread(os.path.join('../data/cam_cars', filename))
        frame_copy = copy.deepcopy(frame)
        result = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(result, size=640)

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

            # MyAnnotator.put_text(frame, cls_name + " " + str(round(conf, 2)), (x0, y0))
            if (x1 - x0) ** 2 + (y1 - y0) ** 2 < 100:
                continue
            crop = frame_copy[x0:x1, y0:y1]
            print(x0, x1, y0, y1, crop.shape, frame.shape, frame_copy.shape)
            cnt_files = str(len(os.listdir('../data/cam_crops')))
            cv2.imwrite(os.path.join('../data/cam_crops', cnt_files + '_crop.png'), crop)
        # print(time.time() - t)


torch.nn.Module.dump_patches = True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_weights", default='weights/resnext(1).pth')
    parser.add_argument("--yolo_weights", default='../weights/yolov5s.pt')
    parser.add_argument("--save_video", default=True)
    parser.add_argument("--out_video_path", default='../inference/out.avi')
    parser.add_argument("--in_video_path", default='../data/video/22.mp4')

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

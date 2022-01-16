import copy
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
    car_classfier = CarClassifier('../weights/best_model_120_2.pth')
    torch.cuda.empty_cache()
    if args.save_video:
        size = (640, 480)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out = cv2.VideoWriter(args.out_video_path, fourcc, 8, size)

    cap = cv2.VideoCapture(args.in_video_path)

    cur_frame = 0
    w, h = (args.width, args.height)
    central_point = (w / 2, h / 2)
    use_stream = args.stream
    streamer = None
    stream_name = args.stream_name
    num_frame = 0
    while cap.isOpened():
        # читаем кадр из видео
        ret, frame = cap.read()
        t = time.time()
        num_frame += 1
        if num_frame % 5 != 0:
            continue
        if ret is True and frame.size != 0:
            frame_full = copy.deepcopy(frame)
            h_full = frame.shape[0]
            w_full = frame.shape[1]
            scale_x = w_full / w
            scale_y = h_full / h
            # frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_full = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
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
                MyAnnotator.put_text(frame,
                                     car_name + ' ' + str(round(conf_cls, 2)) + str(x1 - x0) + ' ' + str(y1 - y0),
                                     (x0, y0))
            cur_frame += 1
            cv2.imshow("Demo", frame)
            # time.sleep(0.1)

            if args.save_video:
                out.write(frame)
        # print(time.time() - t)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    if args.save_video:
        out.release()
    cv2.destroyAllWindows()


torch.nn.Module.dump_patches = True
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

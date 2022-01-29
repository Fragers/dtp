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
from my_utlis.my_tracker import Tracker
import cv2
from PIL import Image
import gc
import torch
import math
import argparse
from my_utlis.annotator import MyAnnotator
from my_utlis.models import CarClassifier, CustomResnext, BrokenCarClassifier, MyEffnet


def add_paddings(x0, y0, x1, y1, w, h, pad):
    """
    Добавить паддинги к координатам
    """
    x0 = int(x0)
    x0 = max(0, x0 - pad)
    y0 = int(y0)
    y0 = max(0, y0 - pad)

    x1 = int(x1)
    x1 = min(w - 1, x1 + pad)
    y1 = int(y1)
    y1 = min(h - 1, y1 + pad)
    return x0, y0, x1, y1


def is_intersected(bbox1, bbox2):
    if bbox1[0] < bbox2[2] and bbox1[1] < bbox2[3] and bbox2[0] < bbox1[2] and bbox2[1] < bbox1[3] or \
            bbox2[0] < bbox1[2] and bbox2[1] < bbox1[3] and bbox1[0] < bbox2[2] and bbox1[1] < bbox2[3]:
        return True
    return False


def sq(x0, y0, x1, y1):
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def find_iou(bbox1, bbox2):
    if not is_intersected(bbox1, bbox2):
        return 0
    x0 = max(bbox1[0], bbox2[0])
    x1 = min(bbox1[2], bbox2[2])
    y0 = max(bbox1[1], bbox2[1])
    y1 = min(bbox1[3], bbox2[3])

    return 2 * sq(x0, y0, x1, y1) / (sq(*bbox1) + sq(*bbox2))


def main(args):
    model = torch.hub.load('C:\\Users\\igors/.cache\\torch\\hub\\ultralytics_yolov5_master', 'custom',
                           path=args.yolo_weights, source='local')
    model.conf = 0.2
    # C:\Users\igors/.cache\torch\hub\ultralytics_yolov5_master
    # model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/yolo_weights.pt')
    # model.conf = args.yolo_conf

    gc.collect()
    tracker = Tracker()

    car_classfier = CarClassifier('../weights/eff_model_car.pth')
    # broken_car_classifier = BrokenCarClassifier('../weights/broken_car_resnext.pth')
    torch.cuda.empty_cache()
    if args.save_video:
        size = (640, 480)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out = cv2.VideoWriter(args.out_video_path, fourcc, 8, size)

    cap = cv2.VideoCapture(args.in_video_path)

    cur_frame = 0
    # w, h = (args.width, args.height)
    # central_point = (w / 2, h / 2)
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
            # scale_x = w_full / w
            # scale_y = h_full / h
            # frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_full = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = model(result, size=1280)

            dets = []
            only_bbox = []

            if not results.pandas().xyxy[0].empty:
                for res in range(len(results.pandas().xyxy[0])):
                    r = results.pandas().xyxy[0].to_numpy()[res]
                    # print(r)
                    if list(r)[6] not in ['car', 'truck', 'train']:
                        continue
                    dets.append(list(r))
                    only_bbox.append(list(r)[:4])

            np_dets = np.array(only_bbox)
            det_with_id = tracker.track(np_dets, num_frame)
            use_track = False
            if use_track:
                for track in det_with_id:
                    x0t, y0t, x1t, y1t, _id = track
                    max_iou = 0
                    max_ind = -1
                    for box_ind in range(len(dets)):
                        x0, y0, x1, y1, conf, cls_id, cls_name = dets[box_ind]
                        bbox1 = (x0, y0, x1, y1)
                        bbox2 = (x0t, y0t, x1t, y1t)
                        iou = find_iou(bbox1, bbox2)
                        if iou > max_iou:
                            max_ind = box_ind
                            max_iou = iou
                    delta = tracker.get_delta_by_id(_id)
                    x0, y0, x1, y1 = int(x0t), int(y0t), int(x1t), int(y1t)
                    MyAnnotator.rectangle(frame, (x0, y0), (x1, y1))
                    to_print = str(_id)
                    if abs(delta) > 15:
                        to_print += ' ' + str(round(delta, 2))
                    MyAnnotator.put_text(frame,
                                         to_print,
                                         (x0, y1))
            else:
                # for track in
                for i in range(len(det_with_id)):
                    x0, y0, x1, y1, conf, cls_id, cls_name = dets[i]
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                    MyAnnotator.rectangle(frame, (x0, y0), (x1, y1))

                    MyAnnotator.put_text(frame,
                                         cls_name,
                                         (x0, y1))


                    # if cls_name not in ['car', 'truck', 'bus', 'boat', 'motorcycle']:
                    #     continue
                    # MyAnnotator.put_text(frame, cls_name + " " + str(round(conf, 2)), (x0, y0))
                    #
                    crop = frame_copy[y0:y1, x0:x1]
                    car_name, conf_cls, np_pred = car_classfier.infer(copy.deepcopy(crop))
                    # x10, y10, x11, y11 = add_paddings(x0, y0, x1, y1, w_full, h_full, 10)
                    #
                    # broken_crop = copy.deepcopy(frame_copy[y10:y11, x10:x11])
                    # # print(y10, y11, x10, x11, frame_copy.shape)
                    # cv2.imshow('123', broken_crop)
                    #
                    # broken_car_name, broken_conf_cls, broken_np_pred = broken_car_classifier.infer(broken_crop)
                    MyAnnotator.put_text(frame,
                                         car_name,
                                         (x0, y0))
            cur_frame += 1
            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("Demo", frame)
            # time.sleep(0.1)

            # if args.save_video:
            #     out.write(frame)
        # print(time.time() - t)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # if args.save_video:
    #     out.release()
    cv2.destroyAllWindows()


torch.nn.Module.dump_patches = True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_weights", default='weights/resnext(1).pth')
    parser.add_argument("--yolo_weights", default='../weights/yolov5m6.pt')
    parser.add_argument("--save_video", default=True)
    parser.add_argument("--out_video_path", default='../inference/out.avi')
    parser.add_argument("--in_video_path",
                        default='../data/video_2/Зафиксировано ДТП и присутствует спецтранспорт/10.mp4')


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

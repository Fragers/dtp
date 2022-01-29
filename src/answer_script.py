import copy
import os
import time

import numpy as np

import cv2
from PIL import Image
import gc
import torch
import math
from my_utlis.my_tracker import Tracker
from my_utlis.annotator import MyAnnotator
from my_utlis.models import CarClassifier, CustomResnext, DtpClassifier
import json


# Id Видео с дтп
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


def main(car_classifier_weights, yolo_weights, yolo_conf, video_dir):
    # Загружаем yolo
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=yolo_weights)
    model.conf = 0.5
    # C:\Users\igors/.cache\torch\hub\ultralytics_yolov5_master
    # C:\\Users\\igors/.cache\\torch\\hub\\ultralytics_yolov5_master
    # model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/yolo_weights.pt')
    # model.conf = args.yolo_conf
    gc.collect()
    # Модель для классификации типа машин
    car_classfier = CarClassifier(car_classifier_weights)

    dtp_classifier = DtpClassifier('../weights/resnext_car4_with_broken_car.pth')

    acc = 0
    all_dtp_cnt = 0
    answer_json = []

    for video_number, video_file in enumerate(os.listdir(video_dir)):
        print(video_number)
        # if video_file == '1.mp4':
        #     continue
        cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
        video_id = video_file.split('.')[0]
        tracker = Tracker()
        cnt_dtp = 0
        cnt_not_dtp = 0
        num_frame = 0
        obr_frame = 0
        frame_number = 0
        cur_json = {'video_name': video_file}

        while cap.isOpened():
            # читаем кадр из видео
            ret, frame = cap.read()
            t = time.time()
            # num_frame += 1
            frame_number += 1
            with_dtp = False
            if frame_number % 30 != 0:
                continue
            if ret is True and frame.size != 0:
                obr_frame += 1
                frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                np_dets = np.array(dets)
                #
                np_dets = np.array(only_bbox)
                det_with_id = tracker.track(np_dets, frame_number)
                use_track = True
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
                        crop = frame_copy[y0:y1, x0:x1]
                        car_name, conf_cls, np_pred = car_classfier.infer(crop)

                        # print(car_name)
                        tracker.update_cls(_id, car_name)
                        # print(_id, frame_number)

                else:
                    # for track in
                    for i in range(len(det_with_id)):
                        x0, y0, x1, y1, conf, cls_id, cls_name = dets[i]
                        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                        # MyAnnotator.rectangle(frame, (x0, y0), (x1, y1))
                        #
                        # MyAnnotator.put_text(frame,
                        #                      cls_name,
                        #                      (x0, y1))

            else:
                break
        CONST = 1000000
        start_dtp = tracker.find_spec()

        print(start_dtp)
        if start_dtp == CONST:
            cur_type = 'not dtp'
        else:
            cur_type = 'dtp'
        if cur_type == 'dtp':
            cur_json['video_type'] = 'dtp'
            cur_json['start_timestamp'] = start_dtp
        else:
            cur_json['video_type'] = 'not dtp'
            cur_json['start_timestamp'] = -1
        if cur_type == 'dtp':
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            if not os.path.exists('inference_videos'):
                os.mkdir('inference_videos')
            size = (1920, 1080)
            tmp_video_number = video_file
            tmp2 = video_file.split('.')[0]
            out = cv2.VideoWriter(os.path.join('inference_videos', tmp_video_number),
                                  fourcc, 25, size)

            cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
            frame_number = 0
            while cap.isOpened():
                # читаем кадр из видео
                ret, frame = cap.read()
                t = time.time()
                # num_frame += 1
                frame_number += 1
                if ret is True and frame.size != 0:
                    if frame_number >= start_dtp:
                        out.write(frame)
                        # cv2.waitKey(10)
                    if frame_number == start_dtp:
                        result = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        results = model(result, size=1280)
                        dets = []

                        if not results.pandas().xyxy[0].empty:
                            for res in range(len(results.pandas().xyxy[0])):
                                r = results.pandas().xyxy[0].to_numpy()[res]
                                # print(r)
                                if list(r)[6] not in ['car', 'truck', 'train']:
                                    continue
                                dets.append(list(r))
                                # only_bbox.append(list(r)[:4])
                            for cur_det_id in range(len(dets)):
                                x0, y0, x1, y1, conf, cls_id, cls_name = dets[cur_det_id]
                                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                                crop = frame[y0:y1, x0:x1]
                                car_name, conf_cls, np_pred = car_classfier.infer(copy.deepcopy(crop))
                                car_name_d, conf_cls, np_pred = dtp_classifier.infer(copy.deepcopy(crop))
                                MyAnnotator.rectangle(frame, (x0, y0), (x1, y1))
                                MyAnnotator.put_text(frame,
                                                     car_name,
                                                     (x0, y1))
                                cv2.imwrite(os.path.join('inference_videos', tmp2 + '.png'), frame)


                else:
                    break
            cap.release()
            out.release()
            # break
        answer_json.append(cur_json)

        print(cur_json)
    with open('answer.json', 'w') as fp:
        json.dump(answer_json, fp)
    # print('Final accuracy:', round(acc / 50, 2))


if __name__ == "__main__":
    # Path to dtp classifier weights
    car_classifier_weights = '../weights/eff_model_car.pth'
    # Path to yolo weights
    yolo_weights = '../weights/yolov5m6.pt'
    # Yolo conf
    yolo_conf = 0.5
    # Path to directory with video
    video_dir = 'D:\\Downloads\\10 роликов тестовых'
    main(car_classifier_weights, yolo_weights, yolo_conf, video_dir)

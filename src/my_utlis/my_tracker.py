import math

from sort.sort import *
import numpy as np


class Tracker:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.2):
        self.sort = Sort(max_age=max_age,
                         min_hits=min_hits,
                         iou_threshold=iou_threshold)
        self.tracked_objects = dict()

    def track(self, detections: np.ndarray, frame_number):
        """
        Обновляет актуальный треки.
        """
        if len(detections) == 0:
            return []

        tracked_dets = self.sort.update(detections)
        for cur_track in tracked_dets:
            self.add_track(cur_track, frame_number)

        return tracked_dets

    def add_track(self, cur_track, frame_number):
        x0, y0, x1, y1, _id = cur_track
        if _id not in self.tracked_objects.keys():
            self.tracked_objects[_id] = MyTrack(_id, x0, y0, x1, y1, frame_number)
        else:
            self.tracked_objects[_id].update(x0, y0, x1, y1)

    def get_delta_by_id(self, _id):
        return self.tracked_objects[_id].get_delta()

    def update_cls(self, _id, cls):
        self.tracked_objects[_id].update_cls(cls)

    def find_spec(self):
        min_frame = 1000000
        CONST = 1000000
        # print('spec')
        for k in self.tracked_objects.keys():
            is_special = self.tracked_objects[k].is_special()

            if is_special:
                print(k)
                min_cur = self.tracked_objects[k].start_frame
                if min_cur < min_frame:
                    min_frame = min_cur
        return min_frame


class MyTrack:
    def __init__(self, _id, x0, y0, x1, y1, frame_number):
        self.avg = []
        self.bboxes = []
        self.bboxes.append((x0, y0, x1, y1))
        self.avg.append((x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2))
        self.deltas = []
        self.speeds = []
        self.cls = []
        self.start_frame = frame_number

    def update(self, x0, y0, x1, y1):
        self.bboxes.append((x0, y0, x1, y1))
        self.avg.append((x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2))
        cur_speed = math.sqrt((self.avg[-1][0] - self.avg[-2][0]) ** 2 + (self.avg[-1][1] - self.avg[-2][1]) ** 2)
        self.speeds.append(cur_speed)

    def update_cls(self, cls):
        self.cls.append(cls)

    def get_delta(self):
        if len(self.speeds) >= 2:
            # return self.speeds[-1]
            return self.speeds[-1] - self.speeds[-2]
        return 0

    def is_special(self):
        special = 0
        for i in range(len(self.cls)):
            cls = self.cls[i]
            if cls == 'special':
                special += 1
        cnt_last = min(len(self.speeds), 4)
        if cnt_last == 0:
            return False

        avg_speed = 0
        for i in range(len(self.speeds) - cnt_last, len(self.speeds)):
            avg_speed += self.speeds[i]
        avg_speed /= cnt_last

        return special >= 1 / 3 * len(self.cls) and avg_speed <= 15
# self.deltas.append()

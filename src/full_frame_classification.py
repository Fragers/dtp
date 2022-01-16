import copy
import time

import cv2

import torch

import argparse
from my_utlis.annotator import MyAnnotator
from my_utlis.models import CarClassifier, CustomResnext, DtpClassifier


def main(args):
    cap = cv2.VideoCapture(args.in_video_path)

    cur_frame = 0
    w, h = (args.width, args.height)
    central_point = (w / 2, h / 2)
    use_stream = args.stream
    streamer = None
    stream_name = args.stream_name
    num_frame = 0
    dtp_classifier = DtpClassifier('../weights/trans_224.pth')
    if args.save_video:
        size = (640, 480)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out = cv2.VideoWriter(args.out_video_path, fourcc, 8, size)
    while cap.isOpened():
        # читаем кадр из видео
        ret, frame = cap.read()
        t = time.time()
        num_frame += 1
        if num_frame % 5 != 0:
            continue
        if ret is True and frame.size != 0:
            frame_full = copy.deepcopy(frame)

            frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dtp_name, conf_cls, np_pred = dtp_classifier.infer(frame_copy)

            MyAnnotator.put_text(frame,
                                 dtp_name + ' ' + str(round(conf_cls, 2)),
                                 (20, 20))
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

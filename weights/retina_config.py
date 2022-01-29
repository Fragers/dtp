"""RetinaNet with ResNet50-FPN, 1x schedule."""

_base_ = [
    "../src/det/configs/_base_/models/retinanet_r50_fpn.py",
    "../src/det/configs/_base_/datasets/bdd100k.py",
    "../src/det/configs/_base_/schedules/schedule_1x.py",
    "../src/det/configs/_base_/default_runtime.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_r50_fpn_1x_det_bdd100k.pth"
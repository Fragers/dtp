from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2
# Specify the path to model config and checkpoint file
config_file = '../weights/retina_config.py'
checkpoint_file = '../weights/retinanet_r50_fpn_1x_det_bdd100k.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# # visualize the results in a new window
# model.show_result(img, result)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
video = mmcv.VideoReader('../data/video/2.mp4')
for frame in video:
    result = inference_detector(model, frame)
    print(result)
    cv2.imshow('123', model.show_result(frame, result, wait_time=1, show=False, score_thr=0.1))

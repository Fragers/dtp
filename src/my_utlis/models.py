import torch.nn as nn
import albumentations
from albumentations import pytorch as AT
import torch
from efficientnet_pytorch import EfficientNet
import timm
import cv2
import numpy as np


# class CustomResnext(nn.Module):
#     def __init__(self, model_name='resnext50d_32x4d', pretrained=True):
#         super().__init__()
#         self.model = timm.create_model(model_name, pretrained=False, num_classes=4)
#
#     def forward(self, x):
#         x = self.model(x)
#         return x

class CustomResnext(nn.Module):
    def __init__(self, model_name='resnext50d_32x4d', pretrained=True):
        super().__init__()
        # self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=2)
        # self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=2)

    def forward(self, x):
        x = self.model(x)
        #         x = torch.squeeze(x)
        return x


class CarClassifier:
    def __init__(self, path, weights_type='dict', device='cuda:0'):
        # if weights_type == 'dict':
        #     self.model = MyEffnet()
        #     self.model.load_state_dict(torch.load(path))
        # else:
        self.model = torch.load(path)
        #     self.model = torch.load(path)
        self.model.to(device)
        self.device = device
        self.model.eval()
        img_size = 224
        self.transforms = albumentations.Compose([

            albumentations.Resize(img_size, img_size),

            # albumentations.ToGray(),
            albumentations.Normalize(),
            # albumentations.ToFloat(),
            AT.ToTensorV2()
        ])
        # self.labels = ['ambulance', 'police', 'evacuator', 'fire', 'car']
        self.labels = ['car', 'police', 'fire']
        self.id_label = {i: label for i, label in enumerate(self.labels)}

        self.sm = nn.Softmax(1)

    def make_square(self, img):
        sz = max(img.shape[0], img.shape[1])
        left = (sz - img.shape[1]) // 2
        right = sz - img.shape[1] - left
        top = (sz - img.shape[0]) // 2
        bottom = sz - img.shape[0] - top
        #         print(top, bottom, left, right)
        return cv2.copyMakeBorder(img, top + 10, bottom + 10, left + 10, right + 10, cv2.BORDER_CONSTANT, None, value=0)

    def infer(self, img):
        image = self.make_square(img)
        # cv2.imshow('123', image)
        # cv2.waitKey(1)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = self.transforms(image=image)['image']
        # print(image.numpy().shape)
        # cv2.imshow('img', image.detach().numpy().transpose(1, 2, 0))
        image = image[np.newaxis, ...]

        #                 print(image.shape)
        image = image.to(self.device)
        pred = self.model(image)
        pred = self.sm(pred.data)
        # print(pred.cpu().numpy().squeeze())
        # exit(0)
        numpy_pred = pred.cpu().numpy().squeeze()
        # print(pred)
        label = torch.max(pred, 1)[1].item()
        conf = torch.max(pred, 1)[0].item()
        return self.id_label[label], conf, numpy_pred


class DtpClassifier:
    def __init__(self, path, weights_type='dict', device='cuda:0'):
        # if weights_type == 'dict':
        #     self.model = MyEffnet()
        #     self.model.load_state_dict(torch.load(path))
        # else:
        self.model = torch.load(path)
        #     self.model = torch.load(path)
        self.model.to(device)
        self.device = device
        self.model.eval()
        img_size = 224
        self.transforms = albumentations.Compose([

            albumentations.Resize(img_size, img_size),

            # albumentations.ToGray(),
            albumentations.Normalize(),
            # albumentations.ToFloat(),
            AT.ToTensorV2()
        ])
        self.labels = ['dtp', 'not_dtp']
        self.id_label = {i: label for i, label in enumerate(self.labels)}
        self.sm = nn.Softmax(1)

    def make_square(self, img):
        sz = max(img.shape[0], img.shape[1])
        left = (sz - img.shape[1]) // 2
        right = sz - img.shape[1] - left
        top = (sz - img.shape[0]) // 2
        bottom = sz - img.shape[0] - top
        #         print(top, bottom, left, right)
        return cv2.copyMakeBorder(img, top + 10, bottom + 10, left + 10, right + 10, cv2.BORDER_CONSTANT, None, value=0)

    def infer(self, img):
        image = self.make_square(img)
        # cv2.imshow('123', image)
        # cv2.waitKey(1)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = self.transforms(image=image)['image']
        # print(image.numpy().shape)
        # cv2.imshow('img', image.detach().numpy().transpose(1, 2, 0))
        image = image[np.newaxis, ...]

        #                 print(image.shape)
        image = image.to(self.device)
        pred = self.model(image)
        pred = self.sm(pred.data)

        numpy_pred = pred.cpu().numpy().squeeze()

        label = torch.max(pred, 1)[1].item()
        conf = torch.max(pred, 1)[0].item()
        return self.id_label[label], conf, numpy_pred

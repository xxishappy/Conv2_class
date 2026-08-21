'''
img图像增强
train:
--RandomResizedCop
--hflip
--brightness_contrast
--gauss_noise
--normalize

valid/test:
--resize:短边等比例缩放到256,中心裁剪为224*224
--normalize
'''

import  albumentations as A
from albumentations.pytorch import ToTensorV2

#定义常量
IMG_SIZE=224
RESIZE_SIZE=256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#Train
def get_train_transform():
    train_t=A.Compose([
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=1), #测试结果0.8980
        # A.SmallestMaxSize(max_size=RESIZE_SIZE, interpolation=1),#测试结果 0.8748
        # A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE), 
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15,contrast_limit=0.15,p=0.5),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_t

#valid/test
def get_val_transform():
    val_t = A.Compose([
    A.SmallestMaxSize(max_size=RESIZE_SIZE, interpolation=1),
    A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
    return val_t

#接口
def get_transform(mode="train"):
    if mode=="train":
        return get_train_transform()
    else:
        return get_val_transform()
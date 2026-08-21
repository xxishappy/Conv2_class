from .transform import get_transform
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

#数据集表格,(img_path,uid,cls_name,split,id,horizontal_box)
EXCEL_PATH="/data/users/dyx/Myproject/FetalPlaneClass/data/dataset/dataset.xlsx"

def read_table(path):
    path=Path(path)
    df=pd.read_excel(path)
    img_path=df['img_path']
    split=df['split']
    label=df['label']
    return img_path,split,label


class FetalPlaneDataset(Dataset):
    def __init__(self,excel_path,split,return_path=False):
        super().__init__()
        self.split=split
        self.return_path = return_path
        self.transform=get_transform(split)

        img_paths,splits,labels=read_table(excel_path)
        mask=(splits==split)
        self.img_paths = img_paths[mask].tolist()
        self.labels=labels[mask].astype(int).tolist()

        print(f"[Dataset]{split}:{len(self)} samples")

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path=self.img_paths[idx]
        label=self.labels[idx]

        img=cv2.imread(img_path,cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像:{img_path}")
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #transform
        out=self.transform(image=img)
        img_tensor=out["image"]
        if self.return_path:
            return img_tensor, label, str(img_path)

        return img_tensor,label
        
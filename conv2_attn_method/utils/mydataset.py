from .transform import get_transform
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

import ast
import numpy as np

#数据集表格,(img_path,uid,cls_name,split,id,box)
EXCEL_PATH="/data/users/dyx/Myproject/FetalPlaneClass/data/dataset/dataset/dataset.xlsx"

##解析坐标
def parse_box(raw):
    if not isinstance(raw,str):
        return None #test
    else:
        site=np.asarray(ast.literal_eval(raw),dtype=np.float32)
        return site

def box_to_mask(box,img_height,img_width,dtype=np.uint8):
    mask=np.zeros((img_height,img_width),dtype=dtype)
    if box is None:
        return mask
    else:
        x1=int(np.clip(np.floor(box[0]),0,img_width))
        y1=int(np.clip(np.floor(box[1]),0,img_height))
        x2=int(np.clip(np.ceil(box[2]),0,img_width))
        y2=int(np.clip(np.ceil(box[3]),0,img_height))

        if x2>x1 and y2>y1:
            mask[y1:y2,x1:x2]=1
        return mask
    
##读取表格
def read_table(path=EXCEL_PATH):
    path=Path(path)
    df=pd.read_excel(path)
    img_path=df['img_path']
    split=df['split']
    label=df['label']
    box=df['outline'] #[135.34, 176.48, 1040.22, 565.96],左上xy，右下xy
    print(type(box))
    return img_path,split,label,box


class FetalPlaneDataset(Dataset):
    def __init__(self,excel_path,split,return_path=False,
                 return_mask=True, #不用mask时改为False
                 mask_fill="zeros",#当没有框时使用什么填充
                 return_has_box=False,#记录当前图像是否有框
                 ):
        super().__init__()
        self.split=split
        self.return_path = return_path
        self.return_mask=return_mask
        self.mask_fill=mask_fill
        self.return_has_box=return_has_box
        self.transform=get_transform(split)

        img_paths,splits,labels,boxes=read_table(excel_path)
        mask=(splits==split)
        self.img_paths = img_paths[mask].tolist()
        self.labels=labels[mask].astype(int).tolist()
        self.boxes=[parse_box(b) for b in boxes[mask].tolist()]

        n_missing=sum(b is None for b in self.boxes)
        # print(f"[Dataset]{split}:{len(self)} samples")
        if self.return_mask and n_missing:
            print(f"[Dataset]{split}:{n_missing}个样本缺少box"
                  f"mask格式为'{self.mask_fill}'")

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path=self.img_paths[idx]
        label=self.labels[idx]
        box=self.boxes[idx]

        img=cv2.imread(img_path,cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像:{img_path}")
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #不使用mask时
        if not self.return_mask:
            #transform
            out=self.transform(image=img)
            img_tensor=out["image"]
            if self.return_path:
                return img_tensor, label, str(img_path)

            return img_tensor,label
        
        #使用mask时
        h,w=img.shape[:2]
        if box is None and self.mask_fill=="ones":
            roi=np.ones((h,w),dtype=np.uint8)
        else:
            roi=box_to_mask(box,h,w)
        
        #img和mask做同步变换
        out=self.transform(image=img,mask=roi)
        img_tensor=out['image']
        mask_tensor=out['mask'].unsqueeze(0).float() #加通道转浮点

        has_box=bool(mask_tensor.any())

        if self.return_has_box and self.return_path:
            return img_tensor,mask_tensor,label,has_box,str(img_path)
        if self.return_has_box:
            return img_tensor,mask_tensor,label,has_box
        if self.return_path:
            return img_tensor,mask_tensor,label,str(img_path)

        return img_tensor,mask_tensor,label
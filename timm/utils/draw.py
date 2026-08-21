from pathlib import Path
import numpy as np
import seaborn as sns
import json

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import to_hex

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch

FONT_PATH="/data/users/dyx/Myproject/FetalPlaneClass/models/conv2/utils/MSYH.TTC"
CLASS_NAME="/data/users/dyx/Myproject/FetalPlaneClass/models/conv2/utils/class_names.json"
GROUP_CONFIGS = [
    {"name": "骨骼", "labels": ["0", "1", "2", "3", "4"],
     "colors": ["#3C5488", "#F39B7F", "#00A087", "#DC0000", "#6A6599"]},
    {"name": "心脏", "labels": ["5", "6", "7", "8"],
     "colors": ["#7E6148", "#4DBBD5", "#B2DF8A", "#FF7F00"]},
    {"name": "神经", "labels": ["9", "10", "11"],
     "colors": ["#E64B35", "#A6CEE3", "#33A02C"]},
    {"name": "面部", "labels": ["12", "13", "14"],
     "colors": ["#8491B4", "#FDBF6F", "#CAB2D6"]},
    {"name": "腹部", "labels": ["15", "16", "17", "18"],
     "colors": ["#1F78B4", "#91D1C2", "#E31A1C", "#B09C85"]},
    {"name": "脐带附属", "labels": ["19", "20", "21"],
     "colors": ["#80796B", "#FB9A99", "#00A1D5"]},
]

#设置字体
fm.fontManager.addfont(str(FONT_PATH))
FONT = fm.FontProperties(fname=FONT_PATH)
plt.rcParams["font.family"] = FONT.get_name()
plt.rcParams["axes.unicode_minus"] = False

#保存图片
def save_figure(fig, output_path, dpi=300):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    

#归一化混淆矩阵，突出>0.05的错分类别
def plot_cm(y_true,y_pred,class_names,output_path,font=FONT):
    n_classes=len(class_names)
    matrix=confusion_matrix(y_true,y_pred,labels=np.arange(n_classes),normalize="true")
    matrix=np.nan_to_num(matrix) #将真实样本数为0导致的NaN转换为0，使得绘图不报错
    fig,ax=plt.subplots(figsize=(20,16))
    sns.heatmap(matrix,cmap="Blues",vmin=0,vmax=1, #颜色
                annot=True,annot_kws={"size":7},fmt=".2f", #数字
                xticklabels=class_names,yticklabels=class_names,ax=ax)
    
    for idx,text in enumerate(ax.texts):
        row=idx//n_classes
        col=idx%n_classes
        if row!=col and matrix[row,col]>=0.05:
            text.set_color("red")
            # text.set_fontweight("bold")
    
    ax.set_xlabel("Predicted Label",fontproperties=font)
    ax.set_ylabel("True Label",fontproperties=font)
    ax.set_title("Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    save_figure(fig,output_path)

#绘制ROC曲线
def plot_roc(y_true,y_prob,class_names,output_path,font=FONT):
    y_true=np.asarray(y_true)
    y_prob=np.asarray(y_prob)
    n_classes=len(class_names)
    binary_labels=label_binarize(y_true,classes=np.arange(n_classes))
    roc_colors=list(plt.get_cmap("tab20").colors)+list(plt.get_cmap("Set2").colors)
    fig,ax=plt.subplots(figsize=(10,8))
    for class_index,class_name in enumerate(class_names):
        positives=binary_labels[:,class_index].sum()
        if positives==0 or positives==len(binary_labels):
            continue
        fpr,tpr,_=roc_curve(binary_labels[:,class_index],y_prob[:,class_index])
        ax.plot(fpr,tpr,lw=1.4,color=roc_colors[class_index % len(roc_colors)],
                label=f"{class_name}(AUC={auc(fpr,tpr):.3f})",
                )
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_title("Receiver Operating Characteristic(ROC)", fontproperties=font)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", prop=font, fontsize=8,frameon=True)
    fig.tight_layout()
    save_figure(fig, output_path)

#绘制tsne，组内颜色差异大
def plot_tsne(features,labels,class_names,output_path,font=FONT,max_samples=5000,group_configs=GROUP_CONFIGS,random_state=42):
    features=np.asarray(features)
    labels=np.asarray(labels)
    if max_samples and len(features) > max_samples:
        rng=np.random.default_rng(random_state)
        selected=rng.choice(len(features),size=max_samples,replace=False)
        features,labels=features[selected],labels[selected]
    
    embedding=TSNE(n_components=2,init="pca",learning_rate="auto",
                   perplexity=30,random_state=random_state).fit_transform(features)
    
    color_map,group_map={},{}
    for group in group_configs:
        class_labels = [int(label) for label in group["labels"]]
        colors = group["colors"]
        for class_index,color in zip(class_labels,colors):
            color_map[class_index]=color
            group_map[class_index]=group["name"]
    
    fig,ax=plt.subplots(figsize=(20,16))
    for label in sorted(np.unique(labels)):
        label=int(label)
        mask=labels==label
        color=color_map[label]
        group=group_map[label]
        display_name=str(class_names[label])
        ax.scatter(
            embedding[mask,0],embedding[mask,1],
            s=22,alpha=0.78,color=color,edgecolors="white",
            linewidths=0.25,label=f"{display_name}"
        )
        center_x=np.median(embedding[mask,0])
        center_y=np.median(embedding[mask,1])
        ax.text(
            center_x, center_y, display_name,
            ha="center", va="center", fontsize=9, fontproperties=font,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.88,
                "linewidth": 1.2,
            },
        )
    ax.set_title("t-SNE", fontproperties=font, fontsize=16)
    ax.set_xlabel("Dimension 1", fontproperties=font)
    ax.set_ylabel("Dimension 2", fontproperties=font)
    ax.grid(alpha=0.15)
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0,
        prop=font, fontsize=8, frameon=True,
    )
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    save_figure(fig, output_path)

#绘制grad_cam
#如果指定了图像路径，对当前这张图绘制热力图，如果没有，对每一类随机采样2张进行绘图
def plot_gradcam(model,dataset,device,class_names,output_dir,imgpath=None,alpha=0.45,cmap="turbo",sample_per_class=2,random_state=42):
    target_layer = model.stages[-1][-1]
    output_dir=Path(output_dir)/"gradcam"
    output_dir.mkdir(parents=True,exist_ok=True)

    model.eval()

    #如果指定了图像路径
    if imgpath is not None:
        requested_path=Path(imgpath).resolve()
        selected_indices=[index for index,path in enumerate(dataset.img_paths) 
                          if Path(path).resolve()==requested_path]
        selected_indices=[selected_indices[0]]
    
    #默认模式：未指定图像路径，随机采样
    else:
        rng=np.random.default_rng(random_state)
        all_labels=np.asarray(dataset.labels)
        selected_indices=[]
        for class_index in range(len(class_names)):
            candidates=np.where(all_labels==class_index)[0]
            chosen=rng.choice(candidates,size=min(sample_per_class,len(candidates)),replace=False)
            selected_indices.extend(chosen.tolist())

    #绘图
    for dataset_index in selected_indices:
        image,true_label,_=dataset[dataset_index]
        true_label=int(true_label)
        class_name=str(class_names[true_label])
        
        input_tensor=image.unsqueeze(0).to(device)
        with GradCAM(model=model,target_layers=[target_layer]) as cam:
            heatmap=cam(input_tensor=input_tensor,targets=[ClassifierOutputTarget(true_label)],)[0]

            image_rgb=image.detach().float().cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_rgb = (image_rgb * std + mean).clamp(0, 1)
            image_rgb = image_rgb.permute(1, 2, 0).numpy()

            heatmap = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
            heatmap_rgb = plt.get_cmap(cmap)(heatmap)[..., :3]
            alpha_map = (heatmap * alpha)[..., np.newaxis]
            overlay = image_rgb * (1.0 - alpha_map) + heatmap_rgb * alpha_map

            class_dir = output_dir / f"class_{true_label:02d}_{class_name}"
            class_dir.mkdir(parents=True, exist_ok=True)
            source_path = Path(dataset.img_paths[dataset_index])
            output_path = class_dir / (
                f"{source_path.stem}.png"
            )
            plt.imsave(output_path, np.clip(overlay, 0.0, 1.0))
            print(f"Saved Grad-CAM: {output_path}")
        


'''
将训练参数变化绘制成可视化图
'''
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


#获取值
def values(history,key):
    out=[]
    for row in history:
        value=row.get(key,np.nan)
        out.append(float(value))
    return np.asarray(out,dtype=np.float64)

#(train_loss,val_loss),lr,val_acc,val_f1,val_auc,val_specificity
def plot_history(history_path,output_path,font_path):
    #设置字体
    fm.fontManager.addfont(font_path)
    font=fm.FontProperties(fname=font_path)
    plt.rcParams['font.family']=font.get_name()

    #加载json
    with open(history_path,"r",encoding="utf-8") as f:
        history=json.load(f)
    
    epochs=values(history,"epoch")
    epochs=epochs.astype(int)

    fig,axes=plt.subplots(2,3,figsize=(18,10),squeeze=False)
    axes=axes.ravel() #把两行三列的表格索引拍成一排，简化索引

    #图1：loss
    ax=axes[0]
    train_loss=values(history,"train_loss")
    val_loss=values(history,"val_loss")
    ax.plot(epochs,train_loss,label="Train",linewidth=2)
    ax.plot(epochs,val_loss,label="Valid",linewidth=2)
    ax.set_title("Loss",fontproperties=font)
    ax.set_xlabel("Epoch",fontproperties=font)
    ax.set_ylabel("Loss",fontproperties=font)
    ax.grid(alpha=0.25) #开启透明度为0.25的背景网格线
    ax.legend(prop=font) #显示图例
    
    #图2：lr
    ax = axes[1]
    lr = values(history, "lr")
    if not np.isnan(lr).all():
        ax.plot(epochs, lr, color="tab:blue", linewidth=2, label="LR")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title("Learning Rate", fontproperties=font)
    ax.set_xlabel("Epoch", fontproperties=font)
    ax.set_ylabel("LR", fontproperties=font)
    ax.grid(alpha=0.25)
    ax.legend(prop=font)

    #图3：accuracy
    ax = axes[2]
    val_acc = values(history, "val_acc")
    ax.plot(epochs, val_acc, color="tab:blue", linewidth=2, label="Val Acc")
    ax.set_title("Accuracy (%)", fontproperties=font)
    ax.set_xlabel("Epoch", fontproperties=font)
    ax.set_ylabel("Accuracy (%)", fontproperties=font)
    ax.grid(alpha=0.25)
    ax.legend(prop=font)

    
    #图4:macro_f1
    ax = axes[3]
    val_f1 = values(history, "val_f1")
    if not np.isnan(val_f1).all():
        ax.plot(epochs, val_f1, color="tab:blue", linewidth=2, label="Val F1")
    ax.set_title("Macro-F1 (%)", fontproperties=font)
    ax.set_xlabel("Epoch", fontproperties=font)
    ax.set_ylabel("Macro-F1 (%)", fontproperties=font)
    ax.grid(alpha=0.25)
    ax.legend(prop=font)

    #图5：AUC
    ax = axes[4]
    val_auc = values(history, "val_auc")
    if not np.isnan(val_auc).all():
        ax.plot(epochs, val_auc, color="tab:blue", linewidth=2, label="Val AUC")
    ax.set_title("AUC (%)", fontproperties=font)
    ax.set_xlabel("Epoch", fontproperties=font)
    ax.set_ylabel("AUC (%)", fontproperties=font)
    ax.grid(alpha=0.25)
    ax.legend(prop=font)

    #图6：specificity
    ax = axes[5]
    val_precision = values(history, "val_precision")
    if not np.isnan(val_precision,).all():
        ax.plot(epochs, val_precision,color="tab:blue", linewidth=2, label="Val Precision")
        ax.legend(prop=font)

    ax.set_title("Macro-Precision (%)", fontproperties=font)
    ax.set_xlabel("Epoch", fontproperties=font)
    ax.set_ylabel("Macro-Precision (%)", fontproperties=font)
    ax.grid(alpha=0.25)

    #设置总标题
    best_idx=int(np.nanargmax(val_f1))
    best_epoch=epochs[best_idx]
    best_value=val_f1[best_idx]

    fig.suptitle(
            f"Training History | Best Val Macro-F1: {best_value:.4f}% (epoch {best_epoch})",
            fontproperties=font, fontsize=16,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved history plot: {output_path}")
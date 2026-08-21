import os
from pathlib import Path
import json
import shutil
import time
import numpy as np

from sklearn.metrics import classification_report
import torch
import torch.nn as nn

from utils.mydataset import FetalPlaneDataset
from utils.metric import cal_metrics
from utils.draw import plot_cm,plot_roc,plot_tsne,plot_gradcam
from torch.utils.data import DataLoader

import timm

# ==================== Config ====================
DATA_DIR = "/data/users/dyx/Myproject/FetalPlaneClass/data/dataset/dataset"
OUTPUT_DIR = "/data/users/dyx/Myproject/FetalPlaneClass/output/timm/vit_base_patch16_224/test"
EXCEL_PATH = "/data/users/dyx/Myproject/FetalPlaneClass/data/dataset/RSFetalPlanes/dataset_rm_badtest.xlsx"
ID2NAME_PATH="/data/users/dyx/Myproject/FetalPlaneClass/models/conv2/utils/class_names.json"

MODEL = "vit_base_patch16_224"  # tiny / base / large tiny_vit_11m_224
CHECKPOINT_PATH="/data/users/dyx/Myproject/FetalPlaneClass/output/timm/vit_base_patch16_224/f1_90.82518-epoch_150.pth"
NUM_CLASSES = 22

BATCH_SIZE=128
NUM_WORKERS=8
AMP = "bf16"       
LABEL_SMOOTHING = 0.1

#加载名称映射
with open(ID2NAME_PATH,"r",encoding="utf-8") as f:
    data=json.load(f)
class_names=[str(data.get(str(index),data.get(index,index))) 
             for index in range(NUM_CLASSES)]

def autocast_context(device):
    if AMP == "none" or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if AMP == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
  
#测试
@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval()
    criterion=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    loss_sum=total_samples=0
    total_time=0.0
    labels_all,preds_all,probs_all,features_all=[],[],[],[]
    misclassified=[]

    for batch_index , (images,labels,image_paths) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with autocast_context(device):
            logits = model(images)
            loss = criterion(logits, labels)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start

        features = model.forward_features(images)
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
      
        probabilities = torch.softmax(logits.float(), dim=1)
        predictions = probabilities.argmax(dim=1)
        batch_size = images.size(0)

        loss_sum += loss.item() * batch_size
        total_samples += batch_size
        
        labels_np = labels.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        labels_all.append(labels_np)
        preds_all.append(preds_np)
        probs_all.append(probabilities.cpu().numpy())
        features_all.append(features.float().cpu().numpy())

        for item_index, (true_label, pred_label) in enumerate(zip(labels_np, preds_np)):
            if int(true_label) != int(pred_label):
                misclassified.append({
                    "dataset_index": batch_index * loader.batch_size + item_index,
                    "image_path":str(image_paths[item_index]),
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "confidence": float(probabilities[item_index, pred_label].item()),
                })

    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(preds_all)
    y_prob = np.concatenate(probs_all)
    features = np.concatenate(features_all)

    metrics = cal_metrics(y_true,y_pred,num_classes=NUM_CLASSES,
                          avg_inference_ms=total_time / max(total_samples, 1) * 1000.0,y_prob=y_prob,
    )
    metrics["loss"] = loss_sum / max(total_samples, 1)
    return metrics, y_true, y_pred, y_prob, features, misclassified

# ==================== Main ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")

    print("=" * 60)
    print(f"Fetal Plane Classification Test - {MODEL}")
    print(f"Split:test | Batch: {BATCH_SIZE} | AMP: {AMP}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    test_set = FetalPlaneDataset(excel_path=EXCEL_PATH, split="test",return_path=True)
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    print(f"Test samples: {len(test_set)}")

    
    model = timm.create_model(MODEL,num_classes=NUM_CLASSES)
    print(f"Loaded model '{MODEL}' via timm.")
   
    model = model.cuda()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing or unexpected:
        print("Warning: verify MODEL, NUM_CLASSES, and DROP_PATH_RATE match train.py.")

    metrics, y_true, y_pred, y_prob, features, misclassified = evaluate(model, test_loader, device)

    # ===== Metrics and reports =====
    report = classification_report(y_true,y_pred,labels=np.arange(NUM_CLASSES),target_names=class_names,
        zero_division=0,digits=4,
    )
    print("\n" + report)
    Path(OUTPUT_DIR, "classification_report.txt").write_text(report, encoding="utf-8")
    with open(Path(OUTPUT_DIR, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"checkpoint": CHECKPOINT_PATH,"model": MODEL,"num_classes": NUM_CLASSES,"test_split": "test","metrics": metrics,},
            f,ensure_ascii=False,indent=2,)
    
    misclassified_root = Path(OUTPUT_DIR) / "misclassified"
    # for item in misclassified:
    #     true_label=item["true_label"]
    #     pred_label=item["pred_label"]
    #     source=Path(item["image_path"])
    #     target_dir=(misclassified_root/f"true_{true_label}"/f"pred_{pred_label}")
    #     target_dir.mkdir(parents=True,exist_ok=True)
    #     target_path=target_dir/source.name
    #     shutil.copy2(source,target_path)

    with open(Path(OUTPUT_DIR, "misclassified.json"), "w", encoding="utf-8") as f:
        json.dump(misclassified, f, ensure_ascii=False, indent=2)

    # ===== Figures from utils.draw =====
    plot_cm(y_true, y_pred, class_names, Path(OUTPUT_DIR, "confusion_matrix.png"),)
    plot_tsne(features,y_true,class_names,Path(OUTPUT_DIR, "tsne.png"),)
    plot_roc(y_true, y_prob, class_names, Path(OUTPUT_DIR, "roc.png"))
    # plot_gradcam(model=model,dataset=test_set,device=device,alpha=0.5,imgpath=None,class_names=class_names,output_dir=OUTPUT_DIR,)

    print("\nMetrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

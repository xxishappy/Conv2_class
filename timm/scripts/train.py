import os
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, accuracy_score, roc_auc_score

import timm
from utils.mydataset import FetalPlaneDataset
from utils.plot_history import plot_history

#HF_ENDPOINT=https://hf-mirror.com
# ==================== Config ====================
DATA_DIR = "/data/users/dyx/Myproject/FetalPlaneClass/data/dataset/RSFetalPlanes"
OUTPUT_DIR_FIRST = "/data/users/dyx/Myproject/FetalPlaneClass/output200/timm"
EXCEL_PATH = "/data/users/dyx/Myproject/FetalPlaneClass/data/dataset/RSFetalPlanes/dataset.xlsx"

MODEL = "densenet121"  # tiny / base / large
OUTPUT_DIR=os.path.join(OUTPUT_DIR_FIRST,MODEL)
NUM_CLASSES = 22
PRETRAINED = True


EPOCHS = 200
BATCH_SIZE = 256
NUM_WORKERS = 16
AMP = "bf16"                # "bf16", "fp16", or "none"
SEED = 42
LOG_INTERVAL = 50

LR = 4e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 10
MIN_LR = 1e-6

LABEL_SMOOTHING = 0
DROP_PATH_RATE = 0.1

# ==================== Utilities ====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_time(seconds:float):
    m,s=divmod(int(seconds),60)
    h,m=divmod(m,60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def save_review_json(review_path,total_sec,best_f1):
    config={
        "DATA_DIR": DATA_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "EXCEL_PATH": EXCEL_PATH,
        "MODEL": MODEL,
        "NUM_CLASSES": NUM_CLASSES,
        "PRETRAINED": PRETRAINED,
        "DROP_PATH_RATE": DROP_PATH_RATE,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "AMP": AMP,
        "SEED": SEED,
        "LOG_INTERVAL": LOG_INTERVAL,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "MIN_LR": MIN_LR,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
    }
    review_data = {
        "config": config,
        "summary": {
            "total_train_time_sec": round(total_sec, 2), 
            "total_train_time_formatted": format_time(total_sec),
            "best_f1": round(best_f1, 4) if best_f1 != -float("inf") else 0.0,
        }
    }

    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, ensure_ascii=False, indent=2)

def save_checkpoint(path, epoch, model, optimizer, best_f1):
    """Save the current raw model state."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
        },
        path,
    )


# ==================== Train / Validate ====================
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0

    amp_enabled = AMP != "none"
    amp_dtype = torch.bfloat16 if AMP == "bf16" else torch.float16

    for step, (images, labels) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (step + 1) % LOG_INTERVAL == 0:
            print(
                f"  [Epoch {epoch + 1}] Step {step + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    return {"train_loss": total_loss / max(total_samples, 1)}


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
 
    for images, labels in loader:

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
       
        logits = model(images)
        loss = criterion(logits, labels)
            
        logits=logits.float() 
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    return {
        "val_loss": total_loss / len(all_labels),
        "val_acc": accuracy_score(all_labels, all_preds) * 100.0,
        "val_f1": f1_score(all_labels,all_preds,labels=list(range(NUM_CLASSES)),average="macro",zero_division=0,) * 100.0,
        "val_auc": roc_auc_score(all_labels,all_probs,multi_class="ovr",average="macro",labels=list(range(NUM_CLASSES)),) * 100.0,
        "val_precision": precision_score(all_labels,all_preds,labels=list(range(NUM_CLASSES)),average="macro",zero_division=0,) * 100.0,
    }

def get_param_groups(model, weight_decay):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "bn" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

# ==================== Main ====================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history_path = os.path.join(OUTPUT_DIR, "history.json")
    review_path = os.path.join(OUTPUT_DIR, "review.json")

    print("=" * 60)
    print(f"Fetal Plane Classification - {MODEL}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR} | AMP: {AMP}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    train_set = FetalPlaneDataset(excel_path=EXCEL_PATH, split="train")
    val_set = FetalPlaneDataset(excel_path=EXCEL_PATH, split="valid")

    print(f"Train samples: {len(train_set)} | Validation samples: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    create_kwargs = {"pretrained": PRETRAINED, "num_classes": NUM_CLASSES}
    # if DROP_PATH_RATE is not None:
    #     create_kwargs["drop_path_rate"] = DROP_PATH_RATE

    model = timm.create_model(MODEL,**create_kwargs).cuda()
    print(f"Loaded model '{MODEL}' via timm.")

    optimizer = AdamW(
            get_param_groups(model, WEIGHT_DECAY),
            lr=LR,
            betas=(0.9, 0.999),
        )

    warmup = LinearLR(
        optimizer,
        start_factor=MIN_LR / LR,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - WARMUP_EPOCHS,
        eta_min=MIN_LR,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[WARMUP_EPOCHS],
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    # GradScaler is required only for fp16. bf16 does not need loss scaling.
    scaler = GradScaler(enabled=(AMP == "fp16"))

    history = []
    best_f1 = -float("inf")
    top3_checkpoints = []

    train_start_time = time.time()

    for epoch in range(EPOCHS):
        t0 = time.time()
        lr_used_this_epoch = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
        )

        # Always validate the raw training model; EMA is intentionally disabled.
        val_metrics = validate(model, val_loader, criterion)
        epoch_time = time.time() - t0
        record = {
            "epoch": epoch + 1,
            **train_metrics,
            **val_metrics,
            "lr": lr_used_this_epoch,
            "epoch_time_sec": round(epoch_time, 2),
            "eval_model": "raw model",
        }
        history.append(record)

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        cur_f1 = val_metrics["val_f1"]
        if cur_f1 > best_f1:
            best_f1 = cur_f1

        total_elapsed_sec = time.time() - train_start_time
        save_review_json(review_path, total_elapsed_sec, best_f1)

        print(
            f"[Epoch {epoch + 1}/{EPOCHS}] ({epoch_time:.1f}s) | " 
            f"Eval: raw model | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Acc: {val_metrics['val_acc']:.2f}% | "
            f"F1: {val_metrics['val_f1']:.2f}% | "
            f"AUC: {val_metrics['val_auc']:.2f}% | "
            f"Precision: {val_metrics['val_precision']:.2f}%"
        )

        output_png = os.path.join(OUTPUT_DIR, "history.png")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        font_path = os.path.join(project_root, "utils", "MSYH.TTC")
        plot_history(history_path, output_png, font_path)
        

        # Save the raw model used for this epoch's validation metric.
        save_checkpoint(
            os.path.join(OUTPUT_DIR, "last.pth"),
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            best_f1=best_f1,
        )

        ckpt_name = f"f1_{cur_f1:.5f}-epoch_{epoch + 1}.pth"
        ckpt_path = os.path.join(OUTPUT_DIR, ckpt_name)

        if len(top3_checkpoints) < 3 or cur_f1 > top3_checkpoints[0][0]:
            if len(top3_checkpoints) >= 3:
                _, worst_path = top3_checkpoints.pop(0)
                if os.path.exists(worst_path):
                    os.remove(worst_path)

            save_checkpoint(
                ckpt_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                best_f1=best_f1,
            )
            top3_checkpoints.append((cur_f1, ckpt_path))
            top3_checkpoints.sort(key=lambda item: item[0])
            print(f"  Saved top-3 checkpoint: {ckpt_name}")

        # Step after the epoch, so lr in history is the LR that was actually used.
        scheduler.step()
    total_train_time = time.time() - train_start_time
    save_review_json(review_path, total_train_time, best_f1)
    print(f"\nTraining finished in {format_time(total_train_time)}. Review outputted to {review_path}")


if __name__ == "__main__":
    main()

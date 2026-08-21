"""可视化：原始图像/原始框 -> DataLoader 同步增强 -> 四个 Stage 的 prior 融合前后特征。"""

import sys
from pathlib import Path

import cv2
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 当前工作区使用 models；若服务器目录为 model，也兼容该导入。
try:
    import models.convnextv2 as convnextv2
except ModuleNotFoundError:
    import model.convnextv2 as convnextv2

from utils.mydataset import FetalPlaneDataset

# ==================== Config ====================
EXCEL_PATH = "/data/users/dyx/Myproject/FetalPlaneClass/data/dataset/RSFetalPlanes/dataset.xlsx"
CHECKPOINT_PATH = "/data/users/dyx/Myproject/FetalPlaneClass/output2/conv2_attn_method/gau/f1_8.34567-epoch_1.pth"
OUTPUT_DIR = "/data/users/dyx/Myproject/FetalPlaneClass/output2/conv2_attn_method/gau/pipeline_visualization"
FONT_PATH = "/data/users/dyx/Myproject/FetalPlaneClass/models/conv2_attn_method/utils/MSYH.TTC"

MODEL = "convnextv2_tiny"
NUM_CLASSES = 22
DROP_PATH_RATE = 0.2
PRIOR_TYPE = "polar"
PRIOR_ALPHA = 0.5
POLAR_POLE = "up"

SPLIT = "train"       # train / valid / test
DATASET_INDEX = 11116      # 当前 split 中的样本索引
# True：将 DataLoader 输出 mask 强制替换为全零，观察测试时 fallback prior。
USE_EMPTY_MASK = False

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

FONT = fm.FontProperties(fname=FONT_PATH) if Path(FONT_PATH).is_file() else None
if FONT:
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = FONT.get_name()
plt.rcParams["axes.unicode_minus"] = False


def load_model(device):
    kwargs = dict(
        pretrained=False,
        num_classes=NUM_CLASSES,
        drop_path_rate=DROP_PATH_RATE,
        prior_type=PRIOR_TYPE,
        prior_alpha=PRIOR_ALPHA,
    )
    if PRIOR_TYPE == "polar":
        kwargs["pole"] = POLAR_POLE
    model = getattr(convnextv2, MODEL)(**kwargs).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"missing: {len(missing)}, unexpected: {len(unexpected)}")
    return model.eval()


def rgb_from_tensor(image):
    return (image.detach().cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def box_from_mask(mask):
    """使用与 PriorGenerator 相同的规则，从增强后的 mask 得到显示用 xyxy box。"""
    foreground = mask[0].detach().cpu().numpy() > 0.5
    height, width = foreground.shape
    ys, xs = np.where(foreground)
    if len(xs) == 0:
        return (0, 0, width, height), True
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), False


def add_box(ax, box, color, label, linewidth=2):
    x1, y1, x2, y2 = box
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth))
    ax.text(x1, max(0, y1 - 4), label, color=color, fontsize=9, fontproperties=FONT, bbox={"facecolor": "black", "alpha": 0.55, "pad": 1})


@torch.no_grad()
def collect_stage_records(model, image, mask):
    """复现当前 ConvNeXtV2：Downsample -> Stage -> Prior -> 下一个 Stage。"""
    x, records = image, []
    for i in range(4):
        x = model.downsample_layers[i](x)
        x = model.stages[i](x)
        before_prior = x
        attention = model.prior_generator(mask, x.shape[-2:])
        after_prior = before_prior if attention is None else before_prior * (1 + model.prior_alpha[i] * attention)

        response = lambda tensor: tensor[0].abs().mean(0).float().cpu().numpy()
        records.append({
            "stage": i + 1,
            "attention": np.zeros(x.shape[-2:]) if attention is None else attention[0, 0].float().cpu().numpy(),
            "before": response(before_prior),
            "after": response(after_prior),
            "change": response(after_prior - before_prior),
        })
        x = after_prior
    return records


def plot_pipeline(raw_rgb, raw_box, image, mask, transformed_box, is_empty, records, image_path, label, output_path):
    fig = plt.figure(figsize=(24, 18), constrained_layout=True)
    grid = fig.add_gridspec(5, 5)

    # 第一行：DataLoader 前与后。
    ax_raw = fig.add_subplot(grid[0, 0])
    ax_raw.imshow(raw_rgb)
    if raw_box is not None:
        add_box(ax_raw, raw_box, "lime", "Excel 原始 box")
    ax_raw.set_title("DataLoader 前：原始图像 + 原始坐标框", fontproperties=FONT)
    ax_raw.axis("off")

    ax_img = fig.add_subplot(grid[0, 1])
    ax_img.imshow(rgb_from_tensor(image))
    add_box(ax_img, transformed_box, "cyan", "增强后 box")
    ax_img.set_title("DataLoader 后：同步增强图像", fontproperties=FONT)
    ax_img.axis("off")

    ax_mask = fig.add_subplot(grid[0, 2])
    ax_mask.imshow(mask[0].cpu(), cmap="gray", vmin=0, vmax=1)
    ax_mask.set_title("DataLoader 后：同步增强 mask", fontproperties=FONT)
    ax_mask.axis("off")

    ax_note = fig.add_subplot(grid[0, 3:])
    ax_note.axis("off")
    status = "空 mask：PriorGenerator 回退为整图 box" if is_empty else "mask 反解得到增强后的 ROI box"
    ax_note.text(
        0.02, 0.8,
        "流程说明\n\n"
        "原始图像 + Excel box\n"
        "↓ 同步几何增强（image 与 mask 相同）\n"
        "增强图像 + 增强 mask\n"
        "↓ mask → box → Attention A\n"
        "↓ 每个 Stage：Post = Pre × (1 + α × A)\n\n"
        f"当前状态：{status}\n"
        f"prior={PRIOR_TYPE}, alpha={PRIOR_ALPHA}, label={label}",
        fontsize=12, fontproperties=FONT, va="top",
    )

    # 后四行：Stage 1~4 的先验融合前后特征。
    titles = ["融合前特征响应", "注意力图 A", "融合后特征响应", "特征变化 |Post - Pre|", "融合公式"]
    for row, record in enumerate(records, start=1):
        before, after, change = record["before"], record["after"], record["change"]
        feature_max = max(np.percentile(np.r_[before.ravel(), after.ravel()], 99), 1e-8)
        change_max = max(np.percentile(change, 99), 1e-8)
        axes = [fig.add_subplot(grid[row, col]) for col in range(5)]
        maps = [
            axes[0].imshow(before, cmap="magma", vmin=0, vmax=feature_max),
            axes[1].imshow(record["attention"], cmap="viridis", vmin=0, vmax=1),
            axes[2].imshow(after, cmap="magma", vmin=0, vmax=feature_max),
            axes[3].imshow(change, cmap="magma", vmin=0, vmax=change_max),
        ]
        for col, ax in enumerate(axes[:4]):
            ax.axis("off")
            if row == 1:
                ax.set_title(titles[col], fontproperties=FONT)
        axes[0].set_ylabel(f"Stage {record['stage']}", rotation=0, labelpad=35, va="center", fontproperties=FONT)

        axes[4].axis("off")
        axes[4].text(
            0.02, 0.7,
            f"A 尺寸：{record['attention'].shape[0]} × {record['attention'].shape[1]}\n"
            f"alpha = {PRIOR_ALPHA}\n"
            "Post = Pre × (1 + alpha × A)",
            fontsize=10, fontproperties=FONT, va="top",
        )
        if row == 1:
            axes[4].set_title(titles[4], fontproperties=FONT)

        for image_map, axis, name in ((maps[1], axes[1], "A"), (maps[2], axes[2], "|F|"), (maps[3], axes[3], "变化")):
            colorbar = fig.colorbar(image_map, ax=axis, fraction=0.04, pad=0.02)
            colorbar.set_label(name, fontproperties=FONT, fontsize=8)
            colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"Prior Attention 全流程可视化\n{image_path}", fontsize=16, fontproperties=FONT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    

    dataset = FetalPlaneDataset(EXCEL_PATH, SPLIT, return_mask=True, return_path=True)
    if not 0 <= DATASET_INDEX < len(dataset):
        raise IndexError(f"DATASET_INDEX 必须在 [0, {len(dataset) - 1}] 内")

    # 原始图像和原始 Excel box：发生 DataLoader/transform 前。
    raw_path = dataset.img_paths[DATASET_INDEX]
    raw_bgr = cv2.imread(raw_path, cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise FileNotFoundError(raw_path)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    raw_box = dataset.boxes[DATASET_INDEX]

    # 通过真正的 DataLoader 取样，得到 transform 后的 image/mask。
    loader = DataLoader(Subset(dataset, [DATASET_INDEX]), batch_size=1, shuffle=False, num_workers=0)
    images, masks, labels, paths = next(iter(loader))
    image, mask, label, image_path = images[0], masks[0], int(labels[0]), paths[0]

    if USE_EMPTY_MASK:
        mask = torch.zeros_like(mask)

    transformed_box, is_empty = box_from_mask(mask)
    model = load_model(torch.device("cuda"))
    records = collect_stage_records(model, image.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda().float())

    output = Path(OUTPUT_DIR) / f"{Path(image_path).stem}_index{DATASET_INDEX}_pipeline.png"
    plot_pipeline(raw_rgb, raw_box, image, mask, transformed_box, is_empty, records, image_path, label, output)


if __name__ == "__main__":
    main()
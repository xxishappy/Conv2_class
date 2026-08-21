"""将训练参数变化绘制成可视化图。

适配 AAG 训练脚本 ``scripts/train.py``：
- Val-ROI：使用真实 ROI 的验证结果；
- Val-Full：所有样本按无 ROI 处理的验证结果，模拟实际测试条件；
- ROI Dependency Gap：Val-ROI F1 - Val-Full F1。
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# 获取值
def values(history, key):
    """从 history 中提取数值字段；字段缺失或无效时使用 NaN。"""
    output = []
    for row in history:
        value = row.get(key, np.nan)
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            output.append(np.nan)
    return np.asarray(output, dtype=np.float64)


def has_valid(array):
    """判断数组中是否包含至少一个有效数值。"""
    return array.size > 0 and not np.isnan(array).all()


def get_new_or_legacy(history, new_key, legacy_key):
    """优先读取 AAG 新字段；兼容旧 history 的字段名。"""
    new_value = values(history, new_key)
    if has_valid(new_value):
        return new_value
    return values(history, legacy_key)


def plot_metric(ax, epochs, roi_value, full_value, title, ylabel, font, color):
    """绘制 Val-ROI 与 Val-Full 两条指标曲线。"""
    drawn = False

    if has_valid(roi_value):
        ax.plot(
            epochs,
            roi_value,
            color=color,
            linewidth=2,
            label="Val",
        )
        drawn = True

    if has_valid(full_value):
        ax.plot(
            epochs,
            full_value,
            color=color,
            linewidth=2,
            linestyle="--",
            label="Val",
        )
        drawn = True

    ax.set_title(title, fontproperties=font)
    ax.set_xlabel("Epoch", fontproperties=font)
    ax.set_ylabel(ylabel, fontproperties=font)
    ax.grid(alpha=0.25)

    if drawn:
        ax.legend(prop=font)


def plot_history(history_path, output_path, font_path=None):
    # 设置字体；字体不存在时使用 Matplotlib 默认字体。
    font = fm.FontProperties()
    if font_path and Path(font_path).exists():
        try:
            fm.fontManager.addfont(str(font_path))
            font = fm.FontProperties(fname=str(font_path))
            plt.rcParams["font.family"] = font.get_name()
        except Exception:
            pass

    # 加载 JSON
    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)

    if not history:
        return

    epochs = values(history, "epoch").astype(int)

    # 新训练脚本字段；如读取旧 history，则回退至 val_* 字段。
    train_loss = values(history, "train_loss")
    val_roi_loss = get_new_or_legacy(history, "val_roi_loss", "val_loss")
    val_full_loss = values(history, "val_full_loss")

    lr = values(history, "lr")

    val_roi_acc = get_new_or_legacy(history, "val_roi_acc", "val_acc")
    val_full_acc = values(history, "val_full_acc")

    val_roi_f1 = get_new_or_legacy(history, "val_roi_f1", "val_f1")
    val_full_f1 = values(history, "val_full_f1")

    val_roi_auc = get_new_or_legacy(history, "val_roi_auc", "val_auc")
    val_full_auc = values(history, "val_full_auc")

    val_roi_precision = get_new_or_legacy(
        history,
        "val_roi_precision",
        "val_precision",
    )
    val_full_precision = values(history, "val_full_precision")

    roi_gap = values(history, "roi_dependency_gap")
    scheduled_p_full = values(history, "scheduled_p_full")
    train_prior_full_ratio = values(history, "train_prior_full_ratio")
    train_prior_expand_ratio = values(history, "train_prior_expand_ratio")

    # 基础 6 图；存在 AAG 相关指标时再增加 Gap / Prior Dropout 图。
    panels = 6
    show_gap = has_valid(roi_gap)
    show_prior_dropout = any(
        has_valid(item)
        for item in (
            scheduled_p_full,
            train_prior_full_ratio,
            train_prior_expand_ratio,
        )
    )
    panels += int(show_gap) + int(show_prior_dropout)

    ncols = 3
    nrows = int(np.ceil(panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()
    index = 0

    # 图 1：Loss
    ax = axes[index]
    index += 1
    drawn = False

    if has_valid(train_loss):
        ax.plot(epochs, train_loss, label="Train", linewidth=2, color="tab:green")
        drawn = True
    if has_valid(val_roi_loss):
        ax.plot(epochs, val_roi_loss, label="Val-ROI", linewidth=2, color="tab:blue")
        drawn = True
    if has_valid(val_full_loss):
        ax.plot(
            epochs,
            val_full_loss,
            label="Val-Full (deployment)",
            linewidth=2,
            color="tab:blue",
            linestyle="--",
        )
        drawn = True

    ax.set_title("Loss", fontproperties=font)
    ax.set_xlabel("Epoch", fontproperties=font)
    ax.set_ylabel("Loss", fontproperties=font)
    ax.grid(alpha=0.25)
    if drawn:
        ax.legend(prop=font)

    # 图 2：Learning Rate
    ax = axes[index]
    index += 1
    if has_valid(lr):
        ax.plot(epochs, lr, color="tab:purple", linewidth=2, label="LR")
        ax.legend(prop=font)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title("Learning Rate", fontproperties=font)
    ax.set_xlabel("Epoch", fontproperties=font)
    ax.set_ylabel("LR", fontproperties=font)
    ax.grid(alpha=0.25)

    # 图 3：Accuracy
    plot_metric(
        axes[index],
        epochs,
        val_roi_acc,
        val_full_acc,
        "Accuracy (%)",
        "Accuracy (%)",
        font,
        "tab:orange",
    )
    index += 1

    # 图 4：Macro-F1
    plot_metric(
        axes[index],
        epochs,
        val_roi_f1,
        val_full_f1,
        "Macro-F1 (%)",
        "Macro-F1 (%)",
        font,
        "tab:blue",
    )
    index += 1

    # 图 5：AUC
    plot_metric(
        axes[index],
        epochs,
        val_roi_auc,
        val_full_auc,
        "Macro-AUC (%)",
        "AUC (%)",
        font,
        "tab:cyan",
    )
    index += 1

    # 图 6：Macro-Precision
    plot_metric(
        axes[index],
        epochs,
        val_roi_precision,
        val_full_precision,
        "Macro-Precision (%)",
        "Precision (%)",
        font,
        "tab:green",
    )
    index += 1

    # 图 7：ROI Dependency Gap
    if show_gap:
        ax = axes[index]
        index += 1
        ax.plot(
            epochs,
            roi_gap,
            color="tab:red",
            linewidth=2,
            label="Val-ROI F1 - Val-Full F1",
        )
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_title("ROI Dependency Gap", fontproperties=font)
        ax.set_xlabel("Epoch", fontproperties=font)
        ax.set_ylabel("Macro-F1 Gap (percentage points)", fontproperties=font)
        ax.grid(alpha=0.25)
        ax.legend(prop=font)

    # 图 8：Anatomical Prior Dropout 训练比例
    if show_prior_dropout:
        ax = axes[index]
        index += 1
        drawn = False

        if has_valid(scheduled_p_full):
            ax.plot(
                epochs,
                scheduled_p_full,
                color="tab:red",
                linewidth=2,
                linestyle=":",
                label="Scheduled missing-prior probability",
            )
            drawn = True

        if has_valid(train_prior_full_ratio):
            ax.plot(
                epochs,
                train_prior_full_ratio,
                color="tab:red",
                linewidth=2,
                label="Actual missing-prior ratio",
            )
            drawn = True

        if has_valid(train_prior_expand_ratio):
            ax.plot(
                epochs,
                train_prior_expand_ratio,
                color="tab:purple",
                linewidth=2,
                label="Expanded-ROI ratio",
            )
            drawn = True

        ax.set_title("Anatomical Prior Dropout", fontproperties=font)
        ax.set_xlabel("Epoch", fontproperties=font)
        ax.set_ylabel("Training sample ratio", fontproperties=font)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        if drawn:
            ax.legend(prop=font, fontsize=9)

    # 隐藏未使用子图。
    for axis in axes[index:]:
        axis.axis("off")

    # 标题：优先显示模拟测试条件下的最佳 Val-Full Macro-F1。
    selection_f1 = val_full_f1 if has_valid(val_full_f1) else val_roi_f1
    selection_name = "Val-Full" if has_valid(val_full_f1) else "Validation"

    title = "Training History"
    if has_valid(selection_f1):
        best_idx = int(np.nanargmax(selection_f1))
        title += (
            f" | Best {selection_name} Macro-F1: "
            f"{selection_f1[best_idx]:.4f}% "
            f"(epoch {epochs[best_idx]})"
        )
        if show_gap:
            title += f" | ROI Gap: {roi_gap[best_idx]:.4f}%"

    fig.suptitle(title, fontproperties=font, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved history plot: {output_path}")

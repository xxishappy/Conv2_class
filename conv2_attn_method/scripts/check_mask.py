"""
先验注入有效性诊断 + 可视化
============================

回答三个问题：
  Q1 先验图长什么样？是否真的"聚焦"？（还是一片缓变背景）
  Q2 注入后特征真的被改变了吗？改变量在网络里如何衰减？
  Q3 LayerNorm 是否抵消了通道均一的调制？

产出：
  prior_maps.png        各 prior_type × 各 stage 分辨率的先验图
  injection_decay.png   注入影响沿网络深度的衰减曲线
  feature_change.png    注入前后特征图对比（空间上哪里被改变）
  报告打印到 stdout

用法:
  python scripts/viz_prior_injection.py                       # 合成 box
  python scripts/viz_prior_injection.py --real --n 8          # 用真实数据
  python scripts/viz_prior_injection.py --ckpt xxx.pth        # 用训练好的权重
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = PROJECT_ROOT / "utils" / "MSYH.TTC"
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

STAGE_SIZES = [(56, 56), (28, 28), (14, 14), (7, 7)]


# ==========================================================================
# Q1  先验图形态
# ==========================================================================
def viz_prior_maps(out_png, box=None):
    from model.priors import PriorGenerator

    if box is None:
        box = torch.tensor([[0.20, 0.15, 0.80, 0.75]])

    configs = [
        ("rect", {}),
        ("gaussian", {}),
        ("polar(up)", {"pole": "up"}),
        ("polar(down)", {"pole": "down"}),
        ("polar(left)", {"pole": "left"}),
        ("polar(random)", {"pole": "random"}),
    ]

    fig, axes = plt.subplots(len(configs), 5, figsize=(17, 2.9 * len(configs)))
    print("=" * 78)
    print("Q1  先验图形态统计  —— 关注 '有效聚焦率'：值>0.5 的像素占比")
    print("=" * 78)
    print(f"{'先验':16s}{'分辨率':>8s}{'mean':>8s}{'std':>8s}"
          f"{'>0.5占比':>10s}{'>0.1占比':>10s}   判定")
    print("-" * 78)

    for r, (name, kw) in enumerate(configs):
        ptype = name.split("(")[0]
        gen = PriorGenerator(ptype, **kw)
        gen.eval()
        for c, size in enumerate(STAGE_SIZES):
            a = gen.from_box(box, size)
            a = a[0, 0].detach().numpy()
            ax = axes[r, c]
            im = ax.imshow(a, cmap="jet", vmin=0, vmax=1)
            ax.set_title(f"{name}\nstage{c+1} {size[0]}×{size[0]}", fontsize=8)
            ax.axis("off")
            if c == 0:
                hi = float((a > 0.5).mean())
                lo = float((a > 0.1).mean())
                verdict = ("聚焦良好" if hi < 0.35 else
                           "偏弥散" if hi < 0.55 else "几乎无聚焦")
                print(f"{name:16s}{f'{size[0]}×{size[0]}':>8s}{a.mean():8.4f}"
                      f"{a.std():8.4f}{hi:10.3f}{lo:10.3f}   {verdict}")

        # 最后一列：沿中轴的剖面
        ax = axes[r, 4]
        a56 = gen.from_box(box, (56, 56))[0, 0].detach().numpy()
        ax.plot(a56[:, 28], label="垂直中轴")
        ax.plot(a56[28, :], label="水平中轴", ls="--")
        ax.axhline(0.5, color="r", ls=":", lw=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6)
        ax.set_title("剖面 (红线=0.5)", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out_png}")
    print("\n判读：>0.5 占比若接近 50%，说明先验退化为缓变背景，起不到空间选择作用。")


# ==========================================================================
# Q2/Q3  注入影响与衰减
# ==========================================================================
@torch.no_grad()
def probe_injection(model, x, mask, alpha=None, out_png=None):
    """逐 stage 单独注入，测量影响如何沿深度衰减。"""
    gen = model.prior_generator
    if alpha is None:
        alpha = model.prior_alpha

    def forward_capture(inject_set):
        h = x
        rec = {}
        for i in range(4):
            h = model.downsample_layers[i](h)
            if i in inject_set:
                a = gen(mask, h.shape[-2:])
                if a is not None:
                    pre = h.clone()
                    h = h * (1.0 + alpha[i] * a)
                    rec[f"inj{i}"] = ((h - pre).norm() / pre.norm()).item()
            h = model.stages[i](h)
            rec[f"out{i}"] = h.clone()
        f = model.norm(h.mean([-2, -1]))
        rec["feat"] = f
        rec["logit"] = model.head(f)
        return rec

    base = forward_capture(set())

    print("\n" + "=" * 78)
    print("Q2  逐 stage 单独注入的影响衰减  (相对 L2 变化率)")
    print("=" * 78)
    print(f"{'注入stage':>10s}{'注入瞬间':>11s}{'stage输出':>11s}"
          f"{'过LN后':>10s}{'最终feat':>11s}{'logit':>10s}   衰减倍数")
    print("-" * 78)

    rows = []
    for i in range(4):
        o = forward_capture({i})
        r_inj = o.get(f"inj{i}", 0.0)
        r_out = ((o[f"out{i}"] - base[f"out{i}"]).norm()
                 / base[f"out{i}"].norm()).item()
        if i < 3:
            ln = model.downsample_layers[i + 1][0]
            r_ln = ((ln(o[f"out{i}"]) - ln(base[f"out{i}"])).norm()
                    / ln(base[f"out{i}"]).norm()).item()
        else:
            r_ln = float("nan")
        r_f = ((o["feat"] - base["feat"]).norm() / base["feat"].norm()).item()
        r_l = ((o["logit"] - base["logit"]).norm() / base["logit"].norm()).item()
        decay = r_inj / max(r_f, 1e-9)
        rows.append((i + 1, r_inj, r_out, r_ln, r_f, r_l))
        warn = "  <<< 严重衰减" if decay > 10 else ""
        print(f"{i+1:>10d}{r_inj:11.4f}{r_out:11.4f}{r_ln:10.4f}"
              f"{r_f:11.4f}{r_l:10.4f}{decay:10.1f}×{warn}")

    o_all = forward_capture({0, 1, 2, 3})
    r_all_f = ((o_all["feat"] - base["feat"]).norm() / base["feat"].norm()).item()
    r_all_l = ((o_all["logit"] - base["logit"]).norm()
               / base["logit"].norm()).item()
    print(f"\n{'全部注入':>10s}{'':11s}{'':11s}{'':10s}{r_all_f:11.4f}{r_all_l:10.4f}")
    print(f"{'仅stage4':>10s}{'':11s}{'':11s}{'':10s}{rows[3][4]:11.4f}{rows[3][5]:10.4f}")
    if rows[3][4] > 0 and r_all_f / rows[3][4] < 1.3:
        print("\n  ！全部注入 ≈ 仅 stage4 注入 -> 前三个 stage 的注入基本无效")

    # ---- Q3 LayerNorm 抵消验证 ----
    print("\n" + "=" * 78)
    print("Q3  LayerNorm(channels_first) 对通道均一调制的抵消")
    print("=" * 78)
    from model.utils import LayerNorm
    t = torch.randn(2, 96, 8, 8)
    c = 1.0 + 0.5 * torch.rand(2, 1, 8, 8)     # 通道均一的逐像素缩放
    ln = LayerNorm(96, data_format="channels_first")
    d = (ln(t * c) - ln(t)).abs().max().item()
    print(f"  LN(c·x) 与 LN(x) 的最大绝对差 = {d:.3e}")
    if d < 1e-4:
        print("  => 通道均一缩放被 LayerNorm 完全抵消。")
        print("     x = x*(1+α·attn) 中 attn 为 (B,1,h,w)，对所有通道相同，")
        print("     其后若紧跟 LayerNorm(channels_first)，调制信息将被消除。")
        print("     stage1-3 之后正是 downsample_layers[i+1][0] = LayerNorm。")

    if out_png:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        idx = [r[0] for r in rows]
        ax[0].plot(idx, [r[1] for r in rows], "o-", label="注入瞬间")
        ax[0].plot(idx, [r[2] for r in rows], "s-", label="stage 输出")
        ax[0].plot(idx, [r[4] for r in rows], "^-", label="最终特征")
        ax[0].plot(idx, [r[5] for r in rows], "v--", label="logit")
        ax[0].set_xlabel("注入的 stage"); ax[0].set_ylabel("相对 L2 变化")
        ax[0].set_xticks(idx); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[0].set_title("注入影响随位置的变化")
        ax[1].bar([str(i) for i in idx], [r[1] / max(r[4], 1e-9) for r in rows],
                  color=["#d62728" if r[1] / max(r[4], 1e-9) > 10 else "#2ca02c"
                         for r in rows])
        ax[1].axhline(10, color="r", ls=":", label="10× 衰减线")
        ax[1].set_xlabel("注入的 stage"); ax[1].set_ylabel("衰减倍数")
        ax[1].set_yscale("log"); ax[1].legend(); ax[1].grid(alpha=.3)
        ax[1].set_title("注入瞬间 / 最终特征 的衰减比")
        fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
        print(f"\n-> {out_png}")
    return rows


# ==========================================================================
@torch.no_grad()
def viz_feature_change(model, x, mask, out_png, stage=3):
    """可视化注入前后，特征图在空间上哪里被改变了。"""
    gen = model.prior_generator
    h_no = h_yes = x
    for i in range(4):
        h_no = model.stages[i](model.downsample_layers[i](h_no))
        h_yes = model.downsample_layers[i](h_yes)
        a = gen(mask, h_yes.shape[-2:])
        if a is not None:
            h_yes = h_yes * (1.0 + model.prior_alpha[i] * a)
        h_yes = model.stages[i](h_yes)
        if i == stage:
            break
    d = (h_yes - h_no).abs().mean(1)[0].cpu().numpy()
    n = h_no.abs().mean(1)[0].cpu().numpy()
    a = gen(mask, h_no.shape[-2:])[0, 0].cpu().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for k, (img, t) in enumerate([
            (x[0].permute(1, 2, 0).cpu().numpy() * 0.22 + 0.45, "输入"),
            (a, f"先验图 stage{stage+1}"),
            (n, "无注入特征 |·|"),
            (d / (n + 1e-8), "相对改变量")]):
        im = ax[k].imshow(np.clip(img, 0, 1) if k == 0 else img,
                          cmap=None if k == 0 else "jet")
        ax[k].set_title(t, fontsize=10); ax[k].axis("off")
        if k: fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print(f"-> {out_png}")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prior_diag")
    ap.add_argument("--prior", default="polar")
    ap.add_argument("--pole", default="up")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--real", action="store_true", help="用真实数据而非合成 box")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    viz_prior_maps(out / "prior_maps.png")

    import model.convnextv2 as convnextv2
    kw = dict(pretrained=False, num_classes=22, drop_path_rate=0.0,
              prior_type=args.prior, prior_alpha=args.alpha)
    if args.prior == "polar":
        kw["pole"] = args.pole
    net = getattr(convnextv2, "convnextv2_tiny")(**kw).eval()

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        sd = ck.get("model_state_dict", ck)
        print(f"\n加载权重: {net.load_state_dict(sd, strict=False)}")

    if args.real:
        from utils.mydataset import FetalPlaneDataset
        from torch.utils.data import DataLoader
        EX = ("/data/users/dyx/Myproject/FetalPlaneClass/data/dataset/"
              "RSFetalPlanes/dataset_rm_badtest.xlsx")
        ds = FetalPlaneDataset(excel_path=EX, split="valid", return_mask=True)
        x, mask, _ = next(iter(DataLoader(ds, batch_size=args.n, shuffle=True)))
        mask = mask.float()
        print(f"\n真实数据: x={tuple(x.shape)} mask前景占比={mask.mean():.4f}")
    else:
        torch.manual_seed(0)
        x = torch.randn(args.n, 3, 224, 224)
        mask = torch.zeros(args.n, 1, 224, 224)
        mask[:, :, 34:168, 45:179] = 1.0
        print(f"\n合成数据: mask前景占比={mask.mean():.4f}")

    probe_injection(net, x, mask, out_png=out / "injection_decay.png")
    viz_feature_change(net, x, mask, out / "feature_change.png")

    print("\n" + "=" * 78)
    print("结论速查")
    print("=" * 78)
    print("1. 若 stage1-3 衰减 >10×  -> 通道均一调制被 LayerNorm 抵消，")
    print("   建议改为通道相关调制（如 FiLM）或把注入点移到 LayerNorm 之后。")
    print("2. 若先验图 >0.5 占比 ≈50% -> 调小 sigma_r / sigma_theta 以增强聚焦。")
    print("3. 若 '全部注入' ≈ '仅stage4' -> 浅层注入无实效，可只在深层注入。")


if __name__ == "__main__":
    main()

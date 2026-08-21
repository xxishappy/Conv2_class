
import argparse
import csv
import glob as globmod
import json
import math
import sys
from pathlib import Path


def flatten(obj, prefix=""):
    """把嵌套 dict/list 拍平成 {'a.b.0': 数值}，只保留数值型。"""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        # 数值列表(如 per-class f1)按下标展开; 非数值列表直接跳过
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            for i, v in enumerate(obj):
                out[f"{prefix}[{i}]"] = float(v)
    elif isinstance(obj, bool):
        pass                              # 布尔不参与平均
    elif isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out[prefix] = float(obj)
    return out


def pick_metrics(data):
    """test_results.json 里指标可能在 'metrics' 下, 也可能就在顶层。"""
    if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
        return data["metrics"]
    return data


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)   # 样本标准差
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="目标目录")
    ap.add_argument("--glob", default=None, help="用通配符批量指定")
    ap.add_argument("--out", default=None, help="输出前缀, 生成 xxx.json 和 xxx.csv")
    ap.add_argument("--digits", type=int, default=4)
    ap.add_argument("--percent", action="store_true", help="数值×100 后显示")
    ap.add_argument("--sort", action="store_true", help="按指标名排序(默认保持原顺序)")
    args = ap.parse_args()

    raw_paths = list(args.dirs)
    if args.glob:
        raw_paths += sorted(globmod.glob(args.glob))

    paths = []
    for raw in raw_paths:
        p = Path(raw)
        if p.is_dir():
            found = False
            for child in sorted(p.iterdir()):
                if child.is_dir() and child.name.startswith("test"):
                    candidate = child / "test_results.json"
                    if candidate.exists():
                        paths.append(candidate)
                        found = True
                    else:
                        print(f"[!] 目录中未找到 test_results.json, 跳过: {candidate}")
            if not found:
                candidate = p / "test_results.json"
                if candidate.exists():
                    paths.append(candidate)
                else:
                    print(f"[!] 目录中没有可用 test_*/test_results.json: {p}")
        else:
            paths.append(p)

    paths = [Path(p) for p in paths]
    if not paths:
        ap.error("没有指定任何文件。用位置参数、目录或 --glob")

    runs, names = [], []
    for p in paths:
        if not p.exists():
            print(f"[!] 文件不存在, 跳过: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        flat = flatten(pick_metrics(data))
        if not flat:
            print(f"[!] 没解析到任何数值指标, 跳过: {p}")
            continue
        runs.append(flat)
        # 用上一级目录名当这次运行的标识, 比完整路径短
        names.append(p.parent.name or p.stem)
        print(f"[load] {p}   ({len(flat)} 项指标)")

    if not runs:
        print("[!] 没有任何可用结果")
        sys.exit(1)

    # 只统计所有 run 都有的指标, 缺失的单独提示
    common = set(runs[0])
    for r in runs[1:]:
        common &= set(r)
    missing = (set().union(*[set(r) for r in runs])) - common
    keys = [k for k in runs[0] if k in common]
    if args.sort:
        keys = sorted(keys)

    scale = 100.0 if args.percent else 1.0
    d = args.digits

    # ---------------- 终端输出 ----------------
    print("\n" + "=" * (34 + 13 * len(runs) + 24))
    print(f"{len(runs)} 次测试结果汇总" + ("  (数值已×100)" if args.percent else ""))
    print("=" * (34 + 13 * len(runs) + 24))
    head = f"{'指标':<32s}"
    for nm in names:
        head += f"{nm[:11]:>13s}"
    head += f"{'mean':>11s}{'std':>10s}{'mean±std':>20s}"
    print(head)
    print("-" * len(head))

    rows = []
    for k in keys:
        vals = [r[k] * scale for r in runs]
        m, s = mean_std(vals)
        line = f"{k[:32]:<32s}"
        for v in vals:
            line += f"{v:>13.{d}f}"
        line += f"{m:>11.{d}f}{s:>10.{d}f}"
        line += f"{f'{m:.{d}f} ± {s:.{d}f}':>20s}"
        print(line)
        rows.append({"metric": k,
                     **{f"run_{nm}": v for nm, v in zip(names, vals)},
                     "mean": m, "std": s,
                     "mean_std": f"{m:.{d}f} ± {s:.{d}f}"})

    if missing:
        print(f"\n[warn] 以下指标并非每次都有, 未参与统计: {sorted(missing)[:8]}"
              + (" ..." if len(missing) > 8 else ""))

    # ---------------- 落盘 ----------------
    out = Path(args.out) if args.out else None
    if out is None:
        first_raw = Path(raw_paths[0])
        out = first_raw if first_raw.is_dir() else first_raw.parent
        out = out / "summary"

    if out.exists() and out.is_dir():
        out = out / "summary"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_runs": len(runs),
            "sources": [str(p) for p in paths],
            "percent_scaled": args.percent,
            "summary": {k: {"values": [r[k] * scale for r in runs],
                            "mean": mean_std([r[k] * scale for r in runs])[0],
                            "std":  mean_std([r[k] * scale for r in runs])[1]}
                        for k in keys},
        }, f, ensure_ascii=False, indent=2)

    with open(out.with_suffix(".csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n已保存: {out.with_suffix('.json')}")
    print(f"        {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

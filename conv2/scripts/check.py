"""
解决 train/valid 解剖结构 label ID 不一致（名称相同）的问题。

现象：同一解剖结构（如"颈椎"）在 train 中 label=75，在 valid 中可能是别的数字。
后果：任何按 label ID 索引的逻辑（POLE_PAIRS / 语义权重 / 结构类型嵌入）
      在 valid 上全部错位，且不报错 —— 与本项目此前踩的坑同类。

对策：**以解剖名称为唯一键**，构建 canonical_id，两个 split 都映射到同一空间。

产出 anatomy_vocab.json：
    {"name2cid": {"颈椎":0, "胸椎":1, ...},
     "cid2name": [...],
     "train_label2cid": {75:0, ...},
     "valid_label2cid": {...}}
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.structure import parse_structures, STRUCTURE_COLS

EXCEL = ("/data/users/dyx/Myproject/FetalPlaneClass/data/dataset/"
         "RSFetalPlanes/dataset_rm_badtest.xlsx")
OUT = "/data/users/dyx/Myproject/FetalPlaneClass/data/dataset/anatomy_vocab.json"


def collect(df, split):
    """返回 {anatomy_name: Counter({label_id: 次数})} 与 {(cls_name, anatomy)} 共现"""
    name2labels = defaultdict(Counter)
    cls2anat = defaultdict(Counter)
    for _, row in df[df.split == split].iterrows():
        cls = str(row.get("cls_name"))
        for s in parse_structures(row):
            name2labels[s.anatomy][s.label] += 1
            cls2anat[cls][s.anatomy] += 1
    return name2labels, cls2anat


def main():
    df = pd.read_excel(EXCEL)
    tr_n2l, tr_c2a = collect(df, "train")
    va_n2l, va_c2a = collect(df, "valid")

    print("=" * 74)
    print("STEP 1  同名结构在 train / valid 的 label ID 对照")
    print("=" * 74)
    all_names = sorted(set(tr_n2l) | set(va_n2l))
    print(f"{'解剖名':16s}{'train label':>16s}{'valid label':>16s}   一致?")
    print("-" * 74)
    n_conflict = 0
    for nm in all_names:
        t = tr_n2l.get(nm, Counter())
        v = va_n2l.get(nm, Counter())
        ts = ",".join(str(k) for k, _ in t.most_common(3)) or "-"
        vs = ",".join(str(k) for k, _ in v.most_common(3)) or "-"
        same = set(t) == set(v)
        if not same:
            n_conflict += 1
        print(f"{nm[:15]:16s}{ts:>16s}{vs:>16s}   {'OK' if same else '<<< 不一致'}")
    print(f"\n不一致的结构: {n_conflict}/{len(all_names)}")

    # 反向：同一 label ID 在两 split 指向不同名称（更危险）
    print("\n" + "=" * 74)
    print("STEP 2  同一 label ID 指向不同解剖（最危险，会静默串类）")
    print("=" * 74)
    tr_l2n, va_l2n = defaultdict(set), defaultdict(set)
    for nm, c in tr_n2l.items():
        for l in c:
            tr_l2n[l].add(nm)
    for nm, c in va_n2l.items():
        for l in c:
            va_l2n[l].add(nm)
    bad = 0
    for l in sorted(set(tr_l2n) | set(va_l2n)):
        a, b = tr_l2n.get(l, set()), va_l2n.get(l, set())
        if a and b and a != b:
            bad += 1
            print(f"  label {l:5d}: train={sorted(a)}  valid={sorted(b)}  <<< 冲突")
    if bad == 0:
        print("  无此类冲突")
    else:
        print(f"\n  {bad} 个 label ID 存在跨 split 语义漂移 -> 绝不可按 label ID 索引")

    # STEP 3 建立以名称为准的 canonical id
    print("\n" + "=" * 74)
    print("STEP 3  构建 canonical anatomy id (以名称为唯一键)")
    print("=" * 74)
    name2cid = {nm: i for i, nm in enumerate(all_names)}
    vocab = {
        "name2cid": name2cid,
        "cid2name": all_names,
        "train_label2cid": {str(l): name2cid[nm]
                            for nm, c in tr_n2l.items() for l in c},
        "valid_label2cid": {str(l): name2cid[nm]
                            for nm, c in va_n2l.items() for l in c},
    }
    print(f"共 {len(all_names)} 个 canonical 解剖类别")

    # STEP 4 各切面的结构构型（用 cid，跨 split 可比）
    print("\n" + "=" * 74)
    print("STEP 4  切面 -> 结构构型 一致性检查 (train vs valid)")
    print("=" * 74)
    mismatch = []
    for cls in sorted(set(tr_c2a) | set(va_c2a)):
        a = set(tr_c2a.get(cls, {}))
        b = set(va_c2a.get(cls, {}))
        if a != b:
            mismatch.append((cls, a - b, b - a))
    if not mismatch:
        print("  全部切面的结构构型一致")
    for cls, only_tr, only_va in mismatch:
        print(f"  {cls[:22]:24s} 仅train={sorted(only_tr)}  仅valid={sorted(only_va)}")

    vocab["cls2anat"] = {c: sorted(v) for c, v in tr_c2a.items()}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {OUT}")
    print("后续所有代码一律使用 canonical id，禁止直接使用原始 label ID。")


if __name__ == "__main__":
    main()
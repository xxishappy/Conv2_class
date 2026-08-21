#查看已有权重的模型
import torch
import timm
# 1. 加载你的权重文件
ckpt_path = torch.load("/data/users/dyx/Myproject/FetalPlaneClass/output/timm/tinyvit/f1_95.24197-epoch_178.pth", map_location="cpu")

print(f"正在加载 checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location="cpu")

# 获取具体的模型权重 state_dict
state_dict = checkpoint["model_state_dict"]

# 如果 state_dict 里的 key 带有 "module." 前缀（比如用过 DDP 训练），先清理掉
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v
state_dict = new_state_dict

print("\n" + "=" * 40)
print("📌 前 8 个层 (Keys) 名称展示（用于辅助判断架构）:")
for i, key in enumerate(state_dict.keys()):
    if i >= 8:
        break
    print(f"  - {key}")
print("=" * 40 + "\n")

# 2. 搜寻目标模型范围（可优先测试 tiny_vit 或 convnext 等）
# 如果确定是 TinyViT 家族：
candidates = timm.list_models("*tiny_vit*")
# 如果不确定，想全盘搜索 timm 所有常用模型，可以取消下面这行的注释：
# candidates = timm.list_models()

print(f"正在尝试在 {len(candidates)} 个候选模型中匹配权重...\n")

matched_models = []

for model_name in candidates:
    try:
        # 创建模型实例
        model = timm.create_model(model_name, pretrained=False)

        # 尝试加载 state_dict
        msg = model.load_state_dict(state_dict, strict=False)

        # 检查 missing_keys 和 unexpected_keys
        missing = len(msg.missing_keys)
        unexpected = len(msg.unexpected_keys)

        if missing == 0 and unexpected == 0:
            print(f"✅ 【完全匹配】成功锁定模型名称: {model_name}")
            matched_models.append(model_name)
            break
        elif missing < 5 and unexpected < 5:
            # 可能是分类头（head.fc / head.l）通道数不一致（比如训练时修改了类别数 num_classes）
            print(
                f"⚠️  【接近匹配】{model_name} -> 缺失 key: {missing} 个, 多余 key: {unexpected} 个"
            )
            print(f"     缺失: {msg.missing_keys}")
            print(f"     多余: {msg.unexpected_keys}")

    except Exception:
        continue

if not matched_models:
    print(
        "\n❌ 在候选列表中未找到 100% 完全匹配的模型。"
    )
    print(
        "💡 提示：如果你的分类类别数（num_classes）和预训练模型默认的不同，看上面打印的【接近匹配】即可确定主干网络！"
    )
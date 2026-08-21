import torch

ckp="/data/users/dyx/Myproject/FetalPlaneClass/output/conv2_attn_method/polar_right0.3/f1_96.39190-epoch_193.pth"
checkpoint = torch.load(ckp, map_location="cpu")

# 1. 查看当前保存的状态信息
print(f"Epoch: {checkpoint.get('epoch')}")
print(f"Best F1: {checkpoint.get('best_f1')}\n")

# 2. 提取模型权重 state_dict
state_dict = checkpoint["model_state_dict"]

# 3. 打印每一层的名称与张量维度
print(f"{'Layer Name':<60} | {'Shape':<30} | {'Param Count'}")
print("-" * 105)

total_params = 0
for name, tensor in state_dict.items():
    shape_str = str(list(tensor.shape))
    num_params = tensor.numel()
    total_params += num_params
    print(f"{name:<60} | {shape_str:<30} | {num_params}")

print("-" * 105)
print(f"Total Parameters: {total_params:,}")
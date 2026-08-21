import os
import sys
import torch


import model.convnextv2 as convnextv2
# ================= 配置参数 =================
# 输入权重路径 (请根据实际情况修改)
CKPT_PATH = "/data/users/dyx/Myproject/FetalPlaneClass/output/conv2/tiny/f1_3.41711-epoch_1.pth"

# 输出 ONNX 路径
ONNX_PATH = "/data/users/dyx/Myproject/FetalPlaneClass/output/conv2/tiny/model.onnx"

# 模型参数 (必须与训练时完全一致)
NUM_CLASSES = 22
IMG_SIZE = 224  # 如果你的 transform 中用了其他尺寸，请修改这里
# ============================================

def export_to_onnx():
    print("=" * 60)
    print("开始导出 ONNX 模型...")
    print(f"权重文件: {CKPT_PATH}")
    print(f"输出路径: {ONNX_PATH}")
    print("=" * 60)

    # 1. 检查权重文件是否存在
    if not os.path.exists(CKPT_PATH):
        print(f"❌ 错误: 找不到权重文件 {CKPT_PATH}")
        return

    # 2. 初始化模型
    print("➡️ 正在初始化 ConvNeXtV2-tiny 模型...")
    # 注意：这里不需要 pretrained=True，因为我们马上要加载自己的权重
    model = convnextv2.convnextv2_tiny(num_classes=NUM_CLASSES)
    
    # 3. 加载权重
    print("➡️ 正在加载训练好的权重...")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    
    # 你的 train.py 中保存的 key 是 "model_state_dict"
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # 加载权重 (strict=True 确保所有权重都正确匹配)
    model.load_state_dict(state_dict, strict=True)
    print("✅ 权重加载成功！")

    # 4. 设置为评估模式 (非常重要：关闭 Dropout 和固定 BatchNorm)
    model.eval()

    # 5. 创建 Dummy Input (模拟输入)
    # 格式: [Batch_Size, Channels, Height, Width]
    # 导出时通常设 Batch_Size=1，后续可通过 dynamic_axes 支持动态 Batch
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    
    # 6. 执行导出
    print("➡️ 正在执行 torch.onnx.export...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_PATH,
            opset_version=17,          # 推荐使用较新的 opset (17)
            input_names=["input"],     # 输入节点名称
            output_names=["output"],   # 输出节点名称
            dynamic_axes={
                "input": {0: "batch_size"},   # 允许 batch_size 维度动态变化
                "output": {0: "batch_size"}
            },
            verbose=False              # 设为 True 可打印详细的网络结构
        )
        print("✅ ONNX 导出成功！")
    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")
        return

    # 7. (可选) 简单验证导出的 ONNX 模型
    print("➡️ 正在验证 ONNX 模型...")
    try:
        import onnx
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 模型结构验证通过！")
        
        # 打印模型输入输出信息
        print("\n--- ONNX 模型信息 ---")
        print(f"Inputs:  {onnx_model.graph.input}")
        print(f"Outputs: {onnx_model.graph.output}")
        print("---------------------\n")
        
    except ImportError:
        print("⚠️ 未安装 onnx 库，跳过结构验证。 (可运行: pip install onnx)")
    except Exception as e:
        print(f"❌ ONNX 模型验证失败: {e}")

    print("=" * 60)
    print(f"🎉 转换完成！文件已保存至: {ONNX_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    export_to_onnx()
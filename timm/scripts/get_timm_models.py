# """
# 列出 timm 中带有预训练权重的 EfficientNet 和 Swin 模型 id
# 用法:
#     python get_timm_models.py
# 或
#     python -m scripts.get_timm_models
# """

# import timm


# def list_pretrained(pattern):
#     """尝试使用 timm.list_models(filter, pretrained=True) 列出带预训练权重的模型。
#     若当前 timm 版本不支持 pretrained 参数，则回退为列出所有匹配模型并提示。"""
#     try:
#         models = timm.list_models(pattern, pretrained=True)
#         return sorted(models)
#     except TypeError:
#         # timm 版本较旧，pretrained 参数可能不可用
#         models = timm.list_models(pattern)
#         print(
#             "注意: 当前 timm.list_models 不支持 `pretrained=True` 参数，\n"
#             "脚本将返回所有匹配的 model-id；其中不一定都包含预训练权重。"
#         )
#         return sorted(models)
#     except Exception as e:
#         print(f"列出模型时出错: {e}")
#         return []


# def main():
#     patterns = {
#         "EfficientNet (classic & v2)": ["efficientnet*", "tf_efficientnet*", "efficientnetv2*"] ,
#         "Swin Transformer": ["swin*"]
#     }

#     for group_name, pats in patterns.items():
#         print("=" * 60)
#         print(group_name)
#         print("=" * 60)
#         all_names = []
#         for p in pats:
#             names = list_pretrained(p)
#             if names:
#                 print(f"Pattern '{p}' -> {len(names)} models")
#                 for n in names:
#                     print(f"  {n}")
#                 all_names.extend(names)
#             else:
#                 print(f"Pattern '{p}' -> no models found")
#         if not all_names:
#             print("（未找到任何模型）")
#         print()


# if __name__ == '__main__':
#     main()
import timm

# Search for all registered Swin V2 models
swin_v2_models = timm.list_models('*swinv2*')
print(swin_v2_models)
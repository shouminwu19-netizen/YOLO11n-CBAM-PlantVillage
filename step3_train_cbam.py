import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import os.path as osp

# ==============================================
# 1. 定义CBAM模块
# ==============================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size ==7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

# ==============================================
# 2. 主函数（修复所有小问题）
# ==============================================
if __name__ == '__main__':
    # --------------------------
    # 0. 提前创建保存目录（核心修复）
    # --------------------------
    PROJECT_DIR = r"C:\Ddeeplearning\ultralytics-8.3.163\runs\disease_detect"
    SAVE_DIR = osp.join(PROJECT_DIR, "yolo11n_cbam_rtx3060_final")
    os.makedirs(SAVE_DIR, exist_ok=True)  # 提前创建目录，避免保存图片失败

    # --------------------------
    # 1. 验证GPU环境
    # --------------------------
    print("="*60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    print("="*60)

    # --------------------------
    # 2. 加载YOLO11n并插入CBAM
    # --------------------------
    from ultralytics import YOLO

    print("\n正在加载官方 YOLO11n 模型...")
    model = YOLO("yolo11n.pt")
    torch_model = model.model

    print("正在手动插入CBAM模块...")
    backbone = torch_model.model[:11]
    neck_and_head = torch_model.model[11:]
    cbam_module = CBAM(1024)
    new_model_list = list(backbone) + [cbam_module] + list(neck_and_head)
    torch_model.model = nn.Sequential(*new_model_list)
    print("✅ CBAM模块已成功插入！")

    # --------------------------
    # 3. 固定随机种子
    # --------------------------
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # --------------------------
    # 4. 路径配置
    # --------------------------
    DATA_YAML = r"C:\PlantVillage-Dataset\PlantVillage_Det\plantvillage_det.yaml"

    # --------------------------
    # 5. 开始GPU训练（关闭可视化避免报错）
    # --------------------------
    print(f"\n🚀 开始使用 RTX 3060 训练 YOLO11n-CBAM 模型！")
    results = model.train(
        data=DATA_YAML,
        epochs=10,                # GPU训练10轮
        imgsz=640,                # 输入图片尺寸
        batch=8,                  # 6G显存稳定值
        device=0,                 # 使用RTX 3060
        seed=seed,                # 固定种子
        project=PROJECT_DIR,      # 结果保存目录
        name="yolo11n_cbam_rtx3060_final",
        exist_ok=True,            # 覆盖已有实验
        plots=True,               # 仍生成最终损失曲线
        workers=0,                # Windows多进程修复
        save=False,               # 临时关闭训练过程中的图片保存（避免报错）
        save_period=-1,           # 只在训练结束后保存权重
    )

    # --------------------------
    # 6. 训练完成
    # --------------------------
    print("\n" + "="*60)
    print("🎉 YOLO11n-CBAM 训练完成！")
    print(f"📂 结果保存路径：{SAVE_DIR}")
    print(f"📊 验证集 mAP@0.5：{results.results_dict['metrics/mAP50(B)']:.4f}")
    print("="*60)
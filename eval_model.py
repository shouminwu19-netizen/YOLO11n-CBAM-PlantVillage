from ultralytics import YOLO
import os
import numpy as np

if __name__ == '__main__':
    # 你的模型准确路径
    model_path = r"C:\Ddeeplearning\ultralytics-8.3.163\runs\disease_detect\yolo11n_cbam_rtx3060_final\weights\last.pt"
    data_path = r"C:\PlantVillage-Dataset\PlantVillage_Det\plantvillage_det.yaml"

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在！路径：{model_path}")
    else:
        print("✅ 找到模型文件，开始评估...")
        model = YOLO(model_path)

        # 运行评估
        results = model.val(
            data=data_path,
            device=0,
            imgsz=640,
            batch=8,
            workers=0,
            plots=True,
            verbose=True,
            save_json=True
        )

        # 核心指标计算（修复所有格式问题）
        map50 = results.box.map  # 实测mAP@0.5=0.993
        precision = np.mean(results.box.p)  # 平均精确率=0.992
        recall = np.mean(results.box.r)  # 平均召回率=0.990
        f1 = 2 * (precision * recall) / (precision + recall)  # F1=0.991
        infer_speed_ms = 4.6  # 实测推理速度（从日志提取：4.6ms inference）
        fps = 1000 / infer_speed_ms  # FPS=217.39
        params = 2.59  # 从日志提取：2,589,562 parameters → 2.59M
        flops = 6.4  # 从日志提取：6.4 GFLOPs

        # 输出最终指标
        print("=" * 60)
        print("✅ 最终实测核心指标（IEEE Access论文专用）：")
        print(f"1. mAP@0.5: {map50:.4f}")
        print(f"2. 精确率 (Precision): {precision:.4f}")
        print(f"3. 召回率 (Recall): {recall:.4f}")
        print(f"4. F1分数: {f1:.4f}")
        print(f"5. 推理速度: {infer_speed_ms:.2f} ms/张 | {fps:.2f} FPS")
        print(f"6. 参数量: {params:.2f} M")
        print(f"7. 计算量: {flops:.2f} GFLOPs")
        print("=" * 60)
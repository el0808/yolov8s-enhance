from ultralytics import YOLO
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def main():
    # 加载模型
    model = YOLO('runs/detect/yolov8s-mbv4-se/weights/last.pt')# 替换为你的模型路径

    # 对 test 集进行评估
    metrics = model.val(data="dataset.yaml", split="test", save=False)

    # 提取 box 相关指标对象
    box_metrics = metrics.box  # <== 重点！

    # 输出总体指标
    print("=== Overall Metrics ===")
    print(f"mAP@0.5: {box_metrics.map50:.4f}")
    print(f"mAP@0.5:0.95: {box_metrics.map:.4f}")
    print(f"Precision (mean): {box_metrics.mp:.4f}")
    print(f"Recall (mean): {box_metrics.mr:.4f}")

    # 输出每个类别的详细指标
    print("\n=== Per-class Metrics ===")
    names = model.names
    ap50_per_class = box_metrics.ap50  # 每类 mAP@0.5
    ap_per_class = box_metrics.ap      # 每类 mAP@0.5:0.95
    p_per_class = box_metrics.p          # 每类 precision
    r_per_class = box_metrics.r          # 每类 recall

    for i, name in names.items():
        print(f"Class: {name:<15}  "
              f"P: {p_per_class[i]:.4f}  "
              f"R: {r_per_class[i]:.4f}  "
              f"mAP50: {ap50_per_class[i]:.4f}  "
              f"mAP50-95: {ap_per_class[i]:.4f}")

    # 保存为 CSV（可选）
    df = pd.DataFrame({
        "Class": [names[i] for i in names],
        "Precision": p_per_class,
        "Recall": r_per_class,
        "mAP50": ap50_per_class,
        "mAP50-95": ap_per_class
    })
    df.to_csv("test_metrics.csv", index=False)
    print("\n=== Saved to test_metrics.csv ===")

if __name__ == "__main__":
    main()

from ultralytics import YOLO

# 加载模型：可以是 YAML 构建的模型，也可以是训练好的 .pt 文件
# model = YOLO('yolov8s.yaml')          # 如果要查看结构定义
# model = YOLO('runs/detect/exp/weights/best.pt')  # 如果查看训练好的模型
model = YOLO('runs/detect/yolov8s-se-test2/weights/last.pt')  # 替换成你的模型路径

# 打印结构与复杂度
model.info(verbose=True)  # verbose=True 会输出详细层级信息

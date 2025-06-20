from ultralytics import YOLO
from ultralytics.nn.modules.block import SPPF

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 加载你的结构定义（从头开始训练）
def main():

    model = YOLO('yolov8s.yaml')
    model.train(
        data='dataset.yaml',     # 数据配置文件路径
        epochs=200,
        imgsz=640,
        batch=16,
        lr0=0.001,           # 推荐初始学习率
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.001,
        device=0,                # GPU编号，若无GPU可设为 'cpu'
        name='yolov8s-se-test',
        pretrained=False,
        auto_augment='',  # 主动传空增强器（彻底关闭）
        augment=True,
        mosaic=1.0,
        close_mosaic=40,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        fliplr=0.0, flipud=0.0,
        perspective=0.0, translate=0.0, scale=0.0, shear=0.0,
        copy_paste=0.0, erasing=0.0,

    )
if __name__ == '__main__':
    main()
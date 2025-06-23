
# 文件概览

该项目实现了一个改进的 YOLOv8 模型，将骨干网络替换成了 MobileNetV4-Small ，并且加入了SE注意力机制。
制作者：黄汉麟
特别鸣谢：https://github.com/jaiwei98/MobileNetV4-pytorch


## 使用方法

1.安装 ultralytics 库
2.打开以下目录 "C:\Users\你的用户名\anaconda3\Lib\site-packages\ultralytics"
3.将本项目中 ultralytics 文件夹内容复制进去，选择替换。
4.将 YOLO 标准格式数据集放入 dataset 文件夹。
5.运行数据增强程序 enhance.py，输出数据集在 augmented 文件夹中。
6.打开 main.py ，根据自己的数据集大小和需求设置训练参数，然后运行。
7.训练的结果保存在 runs/detect 文件夹下。


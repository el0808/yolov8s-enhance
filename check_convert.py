import os
import cv2

# 输入路径（根据你项目结构调整）
image_dir = 'augmented1/images/train'
label_dir = 'augmented1/labels/train'
output_dir = 'check_results_train'

os.makedirs(output_dir, exist_ok=True)

# 支持的图像扩展名
valid_exts = ['.jpg', '.jpeg', '.png']

# 遍历图像目录
for img_file in os.listdir(image_dir):
    if not any(img_file.lower().endswith(ext) for ext in valid_exts):
        continue

    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 图像读取失败：{img_path}")
        continue

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"⚠️ 标签不存在，跳过：{label_path}")
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            cls, xc, yc, bw, bh = map(float, line.strip().split())
        except:
            print(f"⚠️ 标签格式错误：{label_path}")
            continue

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Class {int(cls)}", (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = os.path.join(output_dir, img_file)
    cv2.imwrite(out_path, img)
    print(f"✅ 已检查并保存：{out_path}")

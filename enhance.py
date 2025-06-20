import os
import cv2
import albumentations as A
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ========== 路径设置 ==========
input_image_dir = 'dataset/images/test'   # 原图目录
input_label_dir = 'dataset/labels/test'   # 标签目录（YOLO格式）
output_image_dir = 'augmented1/images/test'
output_label_dir = 'augmented1/labels/test'

# ========== 创建输出目录 ==========
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# ========== 环境设置 ==========
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# ========== 定义增强器列表 ==========
augmentors = [
    A.Compose([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1)),

    A.Compose([
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.7, 1.0), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1)),

    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1)),

    A.Compose([
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1)),

    A.Compose([
        A.HorizontalFlip(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1)),

    A.Compose([
        A.Rotate(limit=10, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1)),

    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.7, 1.0), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1)),

    A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Rotate(limit=10, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1)),

    A.Compose([
        A.GaussianBlur(blur_limit=4, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        A.HueSaturationValue(p=0.8),
        A.Sharpen(p=0.8),
        A.HorizontalFlip(p=0.8),
        A.Rotate(limit=10, p=0.8),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))
]
for i, aug in enumerate(augmentors):
    print(f"Augmentor {i} contains {len(aug.transforms)} transforms")

# ========== 读取图像 ==========
image_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in tqdm(image_files, desc="Augmenting images"):
    img_path = os.path.join(input_image_dir, img_file)
    label_path = os.path.join(input_label_dir, os.path.splitext(img_file)[0] + '.txt')

    image = cv2.imread(img_path)
    if image is None or not os.path.exists(label_path):
        continue

    # 读取标签
    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = map(float, parts)
        bboxes.append([x, y, w, h])
        class_labels.append(int(cls))

    # 保存原图
    cv2.imwrite(os.path.join(output_image_dir, img_file), image)
    with open(os.path.join(output_label_dir, os.path.splitext(img_file)[0] + '.txt'), 'w') as f:
        for line in lines:
            f.write(line if line.endswith('\n') else line + '\n')

    # 进行9次增强
    for i, augmenter in enumerate(augmentors):
        augmented = augmenter(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']

        # 没有有效目标则跳过
        if not aug_bboxes:
            continue

        aug_img_name = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
        aug_label_name = f"{os.path.splitext(img_file)[0]}_aug{i}.txt"

        cv2.imwrite(os.path.join(output_image_dir, aug_img_name), aug_img)
        with open(os.path.join(output_label_dir, aug_label_name), 'w') as f:
            for cls, bbox in zip(class_labels, aug_bboxes):
                f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")

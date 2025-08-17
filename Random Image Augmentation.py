import os
import cv2
import random
import shutil

def flip_horizontal(img):
    return cv2.flip(img, 1)

def flip_vertical(img):
    return cv2.flip(img, 0)

def transform_yolo_label(label_path, mode):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        vals = line.strip().split()
        if len(vals) != 5:
            continue
        cls, xc, yc, ww, hh = vals
        xc = float(xc)
        yc = float(yc)
        ww = float(ww)
        hh = float(hh)
        if mode == 'Hor':
            xc = 1.0 - xc
        elif mode == 'Ver':
            yc = 1.0 - yc
        new_lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
    return new_lines

img_folder = r'images\aba'       # Original images
label_folder = r'label\YOLO_txt\aba_txt'     # Original labels
out_img_folder = r'images_agu\aba_agu'       # Output folder for augmented images
out_label_folder = r'\label_agu\YOLO_txt\aba_agu_txt'    # Output folder for augmented labels

os.makedirs(out_img_folder, exist_ok=True)
os.makedirs(out_label_folder, exist_ok=True)

modes = {
    'Hor': flip_horizontal,
    'Ver': flip_vertical
}

img_names = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg'))]
for img_name in img_names:
    name_base, ext = os.path.splitext(img_name)
    img_path = os.path.join(img_folder, img_name)
    label_path = os.path.join(label_folder, name_base + '.txt')
    out_img_path = os.path.join(out_img_folder, img_name)
    out_label_path = os.path.join(out_label_folder, name_base + '.txt')

    mode = random.choice(list(modes.keys()))
    aug_img = modes[mode](cv2.imread(img_path))
    aug_img_name = f"{name_base}_{mode}{ext}"
    aug_img_path = os.path.join(out_img_folder, aug_img_name)
    cv2.imwrite(aug_img_path, aug_img)

    if os.path.exists(label_path):
        aug_label_lines = transform_yolo_label(label_path, mode)
        aug_label_name = f"{name_base}_{mode}.txt"
        aug_label_path = os.path.join(out_label_folder, aug_label_name)
        with open(aug_label_path, 'w') as f:
            f.writelines(aug_label_lines)

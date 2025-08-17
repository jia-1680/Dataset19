import os
import json
import base64
from PIL import Image

txt_dir = r'label\YOLO_txt\aba_txt'         # Folder containing YOLO TXT annotation files
img_dir = r'images\aba'                     # Folder containing original images
json_dir = r'label\Labelme_json\aba_json'   # Output folder for LabelMe JSON files
os.makedirs(json_dir, exist_ok=True)

class_map = {0: "aba"}

for txt_name in os.listdir(txt_dir):
    if not txt_name.lower().endswith('.txt'):
        continue
    base = os.path.splitext(txt_name)[0]
    txt_path = os.path.join(txt_dir, txt_name)
    img_path = os.path.join(img_dir, base + '.jpg')
    json_path = os.path.join(json_dir, base + '.json')

    if not os.path.exists(img_path):
        print(f'Image not found: {img_path}, skipping.')
        continue

    with Image.open(img_path) as img:
        image_width, image_height = img.size

    with open(img_path, "rb") as img_f:
        img_bytes = img_f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    shapes = []
    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            cls, x_center, y_center, w, h = map(float, parts)
            label = class_map[int(cls)]
            x1 = (x_center - w / 2) * image_width
            y1 = (y_center - h / 2) * image_height
            x2 = (x_center + w / 2) * image_width
            y2 = (y_center + h / 2) * image_height
            shape = {
                "label": label,
                "points": [
                    [x1, y1],
                    [x2, y2]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            }
            shapes.append(shape)

    json_data = {
        "version": "5.2.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": img_b64,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"Generated: {json_path}  Resolution: {image_width}x{image_height}")

print("Batch conversion completed!")

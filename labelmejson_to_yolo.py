import os
import numpy as np
import json
import cv2
from glob import glob

classes = ["aba"]

labelme_path = r"label\labelme_json\aba_json"   # Folder containing your Labelme JSON label files
image_path = r"images\aba"                      # Folder containing your image files
output_path = r"label\YOLO_txt\aba_txt"         # Output folder for YOLO TXT files

os.makedirs(output_path, exist_ok=True)

files = glob(os.path.join(labelme_path, "*.json"))
files = [os.path.splitext(os.path.basename(f))[0] for f in files]

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def ConvertLabelmeToYOLO(files):
    for name in files:
        json_filename = os.path.join(labelme_path, name + ".json")
        txt_path = os.path.join(output_path, name + ".txt")
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))

        image_file = os.path.basename(json_file.get("imagePath", name + ".jpg"))
        img_full_path = os.path.join(image_path, image_file)
        image = cv2.imread(img_full_path)
        if image is None:
            print(f"Failed to read image: {img_full_path}")
            continue
        height, width, _ = image.shape

        with open(txt_path, 'w', encoding='utf-8') as out_file:
            for shape in json_file["shapes"]:
                if shape["shape_type"] == "rectangle":
                    points = np.array(shape["points"])
                    xmin = max(min(points[:, 0]), 0)
                    xmax = max(points[:, 0])
                    ymin = max(min(points[:, 1]), 0)
                    ymax = max(points[:, 1])
                    label = shape["label"]
                    if label not in classes or xmax <= xmin or ymax <= ymin:
                        continue
                    cls_id = classes.index(label)
                    box = (float(xmin), float(xmax), float(ymin), float(ymax))
                    yolo_box = convert((width, height), box)
                    out_file.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in yolo_box]) + "\n")

ConvertLabelmeToYOLO(files)
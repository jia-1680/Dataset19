import os
import xml.etree.ElementTree as ET
from PIL import Image

txt_folder = r"label\YOLO_txt\aba_txt"      # Folder for TXT annotation files
img_folder = r"images\aba"                  # Folder for image files
xml_folder = r"label\Pascal VOC_xml\aba_xml" # Folder for output XML files

if not os.path.exists(xml_folder):
    os.makedirs(xml_folder)

class_names = {0: "aba"}   # Extend class mapping

def yolo_to_voc_bbox(x_center, y_center, w, h, img_w, img_h):
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h
    xmin = int(x_center - w / 2)
    ymin = int(y_center - h / 2)
    xmax = int(x_center + w / 2)
    ymax = int(y_center + h / 2)
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, img_w - 1)
    ymax = min(ymax, img_h - 1)
    return xmin, ymin, xmax, ymax

for file in os.listdir(txt_folder):
    if file.endswith(".txt"):
        txt_path = os.path.join(txt_folder, file)
        base_name = os.path.splitext(file)[0]
        img_path = os.path.join(img_folder, f"{base_name}.jpg")
        output_xml = os.path.join(xml_folder, f"{base_name}.xml")
        if not os.path.exists(img_path):
            print(f"Image not found, skipping: {img_path}")
            continue

        with Image.open(img_path) as img:
            img_width, img_height = img.size
            img_depth = len(img.getbands())

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = img_folder
        ET.SubElement(root, "filename").text = f"{base_name}.jpg"
        ET.SubElement(root, "path").text = os.path.join(img_folder, f"{base_name}.jpg")

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(img_width)
        ET.SubElement(size, "height").text = str(img_height)
        ET.SubElement(size, "depth").text = str(img_depth)

        with open(txt_path, "r") as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 5:
                    continue  # Skip abnormal lines
                class_id = int(items[0])
                x_center, y_center, w, h = map(float, items[1:])
                xmin, ymin, xmax, ymax = yolo_to_voc_bbox(x_center, y_center, w, h, img_width, img_height)

                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = class_names.get(class_id, "unknown")
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(xmin)
                ET.SubElement(bndbox, "ymin").text = str(ymin)
                ET.SubElement(bndbox, "xmax").text = str(xmax)
                ET.SubElement(bndbox, "ymax").text = str(ymax)

        tree = ET.ElementTree(root)
        tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        print(f"Pascal VOC XML annotation file generated: {output_xml}")

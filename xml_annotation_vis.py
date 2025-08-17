import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

xml_folder = r"label\Pascal VOC_xml\aba_xml"        # Folder containing XML annotation files
img_folder = r"images\aba"                          # Folder containing image files
out_folder = r"aba"                                 # Output folder for visualized images

os.makedirs(out_folder, exist_ok=True)

for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith('.xml'):
        continue
    xml_path = os.path.join(xml_folder, xml_file)
    img_name = os.path.splitext(xml_file)[0] + ".jpg"
    img_path = os.path.join(img_folder, img_name)
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        draw.rectangle([xmin, ymin, xmax, ymax], outline='green', width=3)
        draw.text((xmin, ymin - 10), name, fill='green')

    out_path = os.path.join(out_folder, img_name)
    image.save(out_path)
    print(f"Saved: {out_path}")

print("Batch visualization completed. All images have been saved to:", out_folder)

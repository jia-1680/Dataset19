import os
import json
import glob

input_folder = r"label\Labelme_json\All_json"   # Folder containing Labelme JSON files
output_json  = r"label\coco_json\coco.json"      # Output COCO format JSON file

categories_map = {
    "aba": 1, "eco": 2, "kpn": 3, "sau": 4, "ses": 5, "sma": 6, "sep": 7, "efa": 8, "stm": 9,
    "eca": 10, "enc": 11, "cst": 12, "mmo": 13, "bce": 14, "spy": 15, "sag": 16, "ppu": 17,
    "spn": 18, "bcp": 19
}

def bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    return [
        int(round(x_min)),
        int(round(y_min)),
        int(round(w)),
        int(round(h))
    ]

coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

for name, cid in categories_map.items():
    coco["categories"].append({"id": cid, "name": name})

ann_id = 1
img_id = 1

for json_file in glob.glob(os.path.join(input_folder, "*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data.get("imagePath", os.path.splitext(os.path.basename(json_file))[0] + ".jpg")
    image_width = data.get("imageWidth", 0)
    image_height = data.get("imageHeight", 0)

    coco["images"].append({
        "id": img_id,
        "file_name": image_name,
        "width": image_width,
        "height": image_height
    })

    for shape in data.get("shapes", []):
        label = shape["label"]
        category_id = categories_map.get(label, None)
        if category_id is None:
            continue

        points = shape["points"]
        bbox = bbox_from_points(points)

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        ann_id += 1

    img_id += 1

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(coco, f, ensure_ascii=False, indent=2)

print(f"✅ COCO JSON file generated: {output_json}")

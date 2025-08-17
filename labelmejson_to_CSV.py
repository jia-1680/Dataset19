import os
import json
import csv

input_folder = r"label\Labelme_json\aba_json"  # Folder containing Labelme JSON files
output_csv   = r"label\CSV\label_aba.csv"      # Target CSV file

HEADERS = [
    "image_name", "label_name",
    "bbox_x", "bbox_y", "bbox_width", "bbox_height",
    "image_width", "image_height"
]

def bbox_from_points(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (
        int(round(x_min)),
        int(round(y_min)),
        int(round(x_max - x_min)),
        int(round(y_max - y_min))
    )

rows = []

for fname in os.listdir(input_folder):
    if not fname.lower().endswith(".json"):
        continue

    fpath = os.path.join(input_folder, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name   = data.get("imagePath", "")
    image_width  = data.get("imageWidth", "")
    image_height = data.get("imageHeight", "")

    for shp in data.get("shapes", []):
        label = shp.get("label", "")
        pts   = shp.get("points", [])
        if not pts:
            continue

        x, y, w, h = bbox_from_points(pts)

        rows.append([
            image_name,
            label,
            str(x), str(y), str(w), str(h),
            image_width, image_height
        ])

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(HEADERS)
    writer.writerows(rows)

print(f"✅ Generated: {output_csv}, total {len(rows)} records")

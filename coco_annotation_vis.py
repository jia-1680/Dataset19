import os
import json
import glob
import cv2
from collections import defaultdict

images_root = r"images\aba"           # Root directory of images (may contain subfolders)
ann_path    = r"label\coco_json\coco.json"      # Single json file or a folder containing json files
out_dir     = r"vis_coco"       # Output directory for visualized results
os.makedirs(out_dir, exist_ok=True)

IMG_EXTS = {".jpg"}

# Color / Style
GREEN      = (0, 255, 0)
BOX_THICK  = 3
FONT       = cv2.FONT_HERSHEY_TRIPLEX
FONT_SCALE = 1.0
TEXT_THICK = 2
GAP        = 2
PAD        = 4

def build_image_index(root):
    index = {}
    for r, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                key = fn.lower()
                if key not in index:
                    index[key] = os.path.join(r, fn)
    return index

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def draw_label_bar(img, x1, y1, text):
    (tw, th), base = cv2.getTextSize(text, FONT, FONT_SCALE, TEXT_THICK)
    H, W = img.shape[:2]
    bar_x1 = x1
    bar_y2 = max(0, y1 - GAP)
    bar_y1 = max(0, bar_y2 - (th + PAD*2))

    cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), GREEN, thickness=-1)
    org = (bar_x1 + PAD, bar_y2 - PAD - base + 1)
    cv2.putText(img, text, org, FONT, FONT_SCALE, (0, 0, 0), TEXT_THICK, cv2.LINE_AA)

def load_coco_jsons(path):
    json_files = []
    if os.path.isdir(path):
        json_files = sorted(glob.glob(os.path.join(path, "*.json")))
    else:
        json_files = [path]

    datasets = []
    for jp in json_files:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = {img["id"]: img for img in data.get("images", [])}

        cat_id2name = {}
        for c in data.get("categories", []):
            cat_id2name[int(c["id"])] = str(c.get("name", c["id"]))

        anns_by_img = defaultdict(list)
        for a in data.get("annotations", []):
            anns_by_img[int(a["image_id"])].append(a)

        datasets.append((images, anns_by_img, cat_id2name, jp))
    return datasets

image_index = build_image_index(images_root)
datasets = load_coco_jsons(ann_path)

total_drawn = 0
miss_images = 0

for images, anns_by_img, cat_id2name, jp in datasets:
    print(f"[INFO] JSON: {jp} | images={len(images)} anns={sum(len(v) for v in anns_by_img.values())}")

    for img_id, imginfo in images.items():
        filename = str(imginfo.get("file_name") or "")
        key = os.path.basename(filename).lower()
        img_path = image_index.get(key)
        if not img_path:
            miss_images += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            miss_images += 1
            continue

        H, W = img.shape[:2]

        for a in anns_by_img.get(int(img_id), []):
            bx, by, bw, bh = a.get("bbox", [0, 0, 0, 0])
            x1 = clip(int(round(bx)), 0, W - 1)
            y1 = clip(int(round(by)), 0, H - 1)
            x2 = clip(int(round(bx + bw)), 0, W - 1)
            y2 = clip(int(round(by + bh)), 0, H - 1)

            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, BOX_THICK, cv2.LINE_AA)

            cat_name = cat_id2name.get(int(a.get("category_id", -1)), str(a.get("category_id", "")))
            draw_label_bar(img, x1, y1, cat_name)

        save_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img)
        total_drawn += 1
        print(f"[SAVED] {save_path}")

print(f"Done. visualized={total_drawn}, images_not_found={miss_images}, out_dir={out_dir}")

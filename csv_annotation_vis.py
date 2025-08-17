import os
import csv
import cv2
from collections import defaultdict

images_root = r"images\aba"   # Root directory of images (can contain subfolders)
labels_path = r"label\CSV\labels.csv"  # Path to the CSV file containing image annotations (bounding boxes and labels)
out_dir     = r"vis\aba"  # Output directory for visualized images with drawn bounding boxes

os.makedirs(out_dir, exist_ok=True)

fallback_exts = [".jpg"]

def to_int_pixel(v):
    try:
        return int(round(float(v)))
    except Exception:
        return int(v)

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def build_image_index(root):
    idx = {}
    for r, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in fallback_exts:
                idx.setdefault(fn, []).append(os.path.join(r, fn))
    return idx

def find_image_path(index, name):
    if name in index:
        return index[name][0]

    base, ext = os.path.splitext(name)
    if not ext:
        for e in fallback_exts:
            cand = base + e
            if cand in index:
                return index[cand][0]
    return None

def _norm_header(s: str) -> str:
    s = (s or "")
    s = s.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    s = s.strip().lower()
    s = s.replace(" ", "_")
    return s

def read_labels_csv(path):
    boxes_by_image = defaultdict(list)
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                headers = next(reader)
                headers = [_norm_header(h) for h in headers]

                name2idx = {h: i for i, h in enumerate(headers)}

                required = ["image_name","label_name","bbox_x","bbox_y","bbox_width","bbox_height"]
                missing = [c for c in required if c not in name2idx]
                if missing:
                    raise ValueError(f"missing columns: {missing}; got headers={headers}")

                for row in reader:
                    if not row:
                        continue
                    img_name = str(row[name2idx["image_name"]]).strip()
                    label    = str(row[name2idx["label_name"]]).strip()
                    x = to_int_pixel(row[name2idx["bbox_x"]])
                    y = to_int_pixel(row[name2idx["bbox_y"]])
                    w = to_int_pixel(row[name2idx["bbox_width"]])
                    h = to_int_pixel(row[name2idx["bbox_height"]])
                    boxes_by_image[img_name].append((label, x, y, w, h))
            return boxes_by_image
        except Exception as e:
            last_err = e
            continue
    raise last_err

boxes_by_image = read_labels_csv(labels_path)
image_index = build_image_index(images_root)

font         = cv2.FONT_HERSHEY_TRIPLEX   # Times New Roman-like
FONT_SCALE   = 1.5                        # a bit larger
TEXT_THICK   = 2
GREEN        = (0, 255, 0)
BOX_THICK    = 2
GAP          = 2                          # gap between box and label bar
PAD          = 4                          # padding inside label bar

for img_name, items in boxes_by_image.items():
    img_path = find_image_path(image_index, img_name)
    if not img_path:
        print(f"[skip] image not found: {img_name}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[skip] read failed: {img_path}")
        continue

    H, W = img.shape[:2]

    for (label, x, y, w, h) in items:
        x1 = clip(x, 0, W - 1)
        y1 = clip(y, 0, H - 1)
        x2 = clip(x + w, 0, W - 1)
        y2 = clip(y + h, 0, H - 1)

        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, thickness=BOX_THICK, lineType=cv2.LINE_AA)

        (tw, th), base = cv2.getTextSize(label, font, FONT_SCALE, TEXT_THICK)

        bar_x1 = x1
        bar_y2 = max(0, y1 - GAP)
        bar_y1 = max(0, bar_y2 - (th + PAD*2))
        bar_x2 = min(W - 1, bar_x1 + tw + PAD*2)

        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), GREEN, thickness=-1)

        text_org = (bar_x1 + PAD, bar_y2 - PAD - base + 1)
        cv2.putText(img, label, text_org, font, FONT_SCALE, (0, 0, 0), TEXT_THICK, cv2.LINE_AA)

    save_path = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, img)
    print(f"[saved] {save_path}")

print("Done.")

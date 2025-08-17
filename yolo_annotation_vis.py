import cv2
import os

img_dir = r"E:\scientific data\images\bce"      # Input images folder
label_dir = r"E:\scientific data\label\YOLO_txt\bce_txt"  # YOLO labels folder
output_dir = r"E:\scientific data\keshihuaceshi\bce_txt"  # Output results folder
os.makedirs(output_dir, exist_ok=True)

# Mapping dictionary of class IDs to class names (modify according to your own classes)
class_dict = {0: 'aba', 1: 'eco', 2: 'kpn', 3: 'sau', 4: 'ses', 5: 'sma', 6: 'sep',
              7: 'efa', 8: 'stm', 9: 'sep', 10: 'enc', 11: 'cst', 12: 'mmo', 13: 'bce', 14: 'spy'}

img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(label_dir, label_file)
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to read image: {img_path}")
        continue

    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])

        class_name = class_dict.get(class_id, str(class_id))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
        cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    save_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + "_result.jpg")
    cv2.imwrite(save_path, image)
    print(f"Visualization image saved: {save_path}")

print("All done!")

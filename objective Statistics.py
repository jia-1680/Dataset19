import os
import json
from collections import defaultdict

def count_labels_in_folder(folder_path):
    label_counts = defaultdict(int)
    total = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    shapes = data.get("shapes", [])
                    for shape in shapes:
                        label = shape.get("label")
                        if label:
                            label_counts[label] += 1
                            total += 1
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")

    print("Label counts across all JSON files:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    print(f"Total number of labels: {total}")

    return label_counts

json_folder = r"label\Labelme_json\aba_json" # Replace with your JSON folder path
count_labels_in_folder(json_folder)

import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

classes_file_path = "/mnt/data/manual_hypertension_AHA_without_diabetes.csv"  # Обновите путь к вашему файлу

source_images_path = "/mnt/data/cfi"

target_path = "/mnt/data/cfi_manual_hypertension_AHA_without_diabetes"
os.makedirs(target_path, exist_ok=True)

classes_data = pd.read_csv(classes_file_path)

classes_data["image_path"] = classes_data["participant_id"].astype(str).apply(
    lambda x: [file for file in os.listdir(source_images_path) if file.startswith(x)]
)

expanded_rows = []
for _, row in classes_data.iterrows():
    for image_name in row["image_path"]:
        expanded_rows.append({
            "participant_id": row["participant_id"],
            "class": row["class"],
            "image_path": os.path.join(source_images_path, image_name),
        })
expanded_data = pd.DataFrame(expanded_rows)

train_data, temp_data = train_test_split(
    expanded_data, test_size=0.3, random_state=42, stratify=expanded_data["class"]
)
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, random_state=42, stratify=temp_data["class"]
)

splits = {"train": train_data, "val": val_data, "test": test_data}
for split_name, split_data in splits.items():
    split_path = os.path.join(target_path, split_name)
    os.makedirs(split_path, exist_ok=True)
    for class_label in split_data["class"].unique():
        class_path = os.path.join(split_path, str(class_label))
        os.makedirs(class_path, exist_ok=True)

for split_name, split_data in splits.items():
    for _, row in split_data.iterrows():
        source_file = row["image_path"]
        target_file = os.path.join(target_path, split_name, str(row["class"]), os.path.basename(source_file))

        try:
            shutil.copy(source_file, target_file)
        except FileNotFoundError:
            print(f"Image not found: {source_file}")

print("Images successfully organized into train, val, and test directories.")

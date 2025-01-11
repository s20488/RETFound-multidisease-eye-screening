import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Загрузка файла с кластеризированными данными
clustered_data_path = "/mnt/data/clustered_data.csv"
data = pd.read_csv(clustered_data_path)

# Путь к исходным изображениям
source_images_path = "/mnt/data/cfi"

# Путь для сохранения изображений по кластерам и наборам
target_path = "/mnt/data/cfi_target"
os.makedirs(target_path, exist_ok=True)

# Создание папок для train, val, test
splits = ["train", "val", "test"]
for split in splits:
    split_dir = os.path.join(target_path, split)
    os.makedirs(split_dir, exist_ok=True)

# Деление данных на train, val, test
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data["cluster"])
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data["cluster"])

data_splits = {"train": train_data, "val": val_data, "test": test_data}

# Создание папок для каждого кластера в каждом наборе
clusters = data["cluster"].unique()
for split in splits:
    for cluster in clusters:
        cluster_dir = os.path.join(target_path, split, str(cluster))
        os.makedirs(cluster_dir, exist_ok=True)

# Копирование изображений в соответствующие папки
for split, split_data in data_splits.items():
    for _, row in split_data.iterrows():
        image_path = row["image_path"]
        cluster = row["cluster"]

        source_file = os.path.join(source_images_path, os.path.basename(image_path))
        target_file = os.path.join(target_path, split, str(cluster), os.path.basename(image_path))

        try:
            shutil.copy(source_file, target_file)
        except FileNotFoundError:
            print(f"Image not found: {source_file}")

print("Images successfully organized into train, val, and test directories.")

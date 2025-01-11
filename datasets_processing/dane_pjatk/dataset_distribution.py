import os
import pandas as pd
import shutil

# Загрузка файла с кластеризированными данными
clustered_data_path = "/mnt/data/clustered_data.csv"
data = pd.read_csv(clustered_data_path)

# Путь к исходным изображениям
source_images_path = "/mnt/data/cfi"

# Путь для сохранения изображений по кластерам
target_path = "/mnt/data/cfi_target"
os.makedirs(target_path, exist_ok=True)

# Создание папок для кластеров
clusters = data["cluster"].unique()
for cluster in clusters:
    cluster_dir = os.path.join(target_path, str(cluster))
    os.makedirs(cluster_dir, exist_ok=True)

# Копирование изображений в соответствующие папки
for _, row in data.iterrows():
    image_path = row["image_path"]
    cluster = row["cluster"]

    source_file = os.path.join(source_images_path, os.path.basename(image_path))
    target_file = os.path.join(target_path, str(cluster), os.path.basename(image_path))

    try:
        shutil.copy(source_file, target_file)
    except FileNotFoundError:
        print(f"Image not found: {source_file}")

print("Images successfully organized into cluster directories.")

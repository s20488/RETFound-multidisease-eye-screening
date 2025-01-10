import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np


# Подготовка модели
def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    from models_vit import __dict__ as models_vit
    model = models_vit[arch](
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=True,
    )
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()  # Перевод в режим оценки
    return model


chkpt_dir = './RETFound_cfp_weights.pth'
model = prepare_model(chkpt_dir)


# Преобразование PNG в вектор признаков
def extract_features_from_png(image_path, model):
    # Загружаем изображение
    image = Image.open(image_path).convert("RGB")  # Убедимся, что изображение в формате RGB

    # Преобразование изображения в тензор
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = transform(image).unsqueeze(0)  # Добавляем batch dimension

    # Извлечение признаков
    with torch.no_grad():
        features = model(image).squeeze().numpy()  # Преобразуем в numpy
    return features


# Обработка всех PNG файлов
png_folder = "path_to_png_files"  # Путь к папке с PNG файлами
png_paths = [os.path.join(png_folder, fname) for fname in os.listdir(png_folder) if fname.endswith(".png")]

image_features = []
for png_path in png_paths:
    features = extract_features_from_png(png_path, model)
    image_features.append(features)

# Преобразуем в DataFrame
image_features_df = pd.DataFrame(image_features)

# Загрузка биомаркеров
biomarkers_data = pd.read_csv("processed_data.csv")

# Проверяем соответствие количества строк
if len(biomarkers_data) != len(image_features_df):
    raise ValueError("Количество изображений не совпадает с количеством записей в биомаркерах!")

# Объединяем биомаркеры и признаки изображений
final_data = pd.concat([biomarkers_data, image_features_df], axis=1)

# Сохраняем результат
final_data.to_csv("final_data_with_features.csv", index=False)

print("Обработка завершена. Данные сохранены в 'final_data_with_features.csv'.")

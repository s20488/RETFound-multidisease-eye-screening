import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd


# Подготовка модели
def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    from models_vit import __dict__ as models_vit
    model = models_vit[arch](
        img_size=224,
        num_classes=4,
        drop_path_rate=0,
        global_pool=True,
    )
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


chkpt_dir = './RETFound_cfp_weights.pth'
model = prepare_model(chkpt_dir)


# Преобразование PNG в вектор признаков
def extract_features_from_png(image_path, model):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.73126347, 0.42223816, 0.26234768], std=[0.1707951, 0.16099306, 0.13655258]),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).squeeze().numpy()
    return features


# Извлечение признаков изображений
png_folder = "/mnt/data/cfi"  # Путь к папке с PNG файлами
png_paths = [os.path.join(png_folder, fname) for fname in os.listdir(png_folder) if fname.endswith(".png")]

image_features = []
for png_path in png_paths:
    features = extract_features_from_png(png_path, model)
    image_features.append({"image_path": png_path, "features": features})

# Преобразуем в DataFrame
image_features_df = pd.DataFrame(image_features)

# Извлекаем participant_id из имени файла
image_features_df["participant_id"] = image_features_df["image_path"].apply(lambda x: os.path.basename(x).split("_")[0])

# Загрузка таблицы биомаркеров
biomarkers_data = pd.read_csv("/mnt/data/hypertension_clustering_processed.csv")

# Объединяем биомаркеры и признаки изображений
final_data = image_features_df.merge(biomarkers_data, on="participant_id", how="left")

# Проверяем соответствие данных
if final_data.isnull().any().any():
    print("Есть строки без соответствующих биомаркеров! Проверьте данные.")

# Разворачиваем признаки изображений в отдельные столбцы
feature_columns = [f"feature_{i}" for i in range(len(image_features_df['features'].iloc[0]))]
final_features = pd.DataFrame(final_data["features"].tolist(), columns=feature_columns)
final_data = pd.concat([final_data.drop(columns=["features"]), final_features], axis=1)

# Сохраняем результат
final_data.to_csv("final_data_with_image_features.csv", index=False)

print("Обработка завершена. Данные сохранены в 'final_data_with_image_features.csv'.")

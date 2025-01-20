import os
import pandas as pd

# Путь к папке с изображениями
base_folder = "/mnt/data/cfi_manual_diabetes_0.1/"  # Папка, содержащая train, val, test

# Загружаем файл с признаками
csv_file_path = "/mnt/data/processed_data_with_id.csv"
data = pd.read_csv(csv_file_path)

# Списки для хранения обновлённых строк
updated_rows = []

# Обрабатываем каждую папку (train, val, test)
for dataset_type in ['train', 'val', 'test']:
    dataset_path = os.path.join(base_folder, dataset_type)  # Путь к train/val/test
    for label in ['normal', 'diabetes']:  # Метки: normal и diabetes
        label_path = os.path.join(dataset_path, label)  # Путь к train/normal, train/diabetes и т.д.

        if not os.path.exists(label_path):  # Если папка не существует
            print(f"Папка не найдена: {label_path}")
            continue

        # Обрабатываем файлы в папке
        for file_name in os.listdir(label_path):
            if file_name.endswith(".png"):  # Учитываем только PNG
                # Извлекаем participant_id из имени файла
                participant_id = file_name.split("_")[0]

                # Проверяем, существует ли participant_id в CSV
                if participant_id in data['participant_id'].astype(str).values:
                    # Извлекаем строку из CSV
                    matching_rows = data[data['participant_id'] == int(participant_id)]

                    # Для каждого изображения создаём дубликат строки с путём до изображения
                    for _, row in matching_rows.iterrows():
                        new_row = row.copy()
                        new_row['image_path'] = os.path.join(label_path, file_name)  # Путь к изображению
                        new_row['label'] = 0 if label == 'normal' else 1  # Метка: 0 для normal, 1 для diabetes
                        updated_rows.append(new_row)

# Создаём DataFrame из обновлённых данных
updated_data = pd.DataFrame(updated_rows)

# Сохраняем результат в новый CSV
output_csv_path = "/mnt/data/updated_data_with_images.csv"
updated_data.to_csv(output_csv_path, index=False)

print(f"Обновлённый CSV сохранён в файл: {output_csv_path}")

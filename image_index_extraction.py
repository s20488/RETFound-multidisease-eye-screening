import os
import pandas as pd

# Путь к папке с изображениями
image_folder = "cfi_manual_diabetes_0.1/"  # Укажи папку с изображениями

# Загрузка исходного CSV
csv_file_path = "/mnt/data/processed_data_with_id.csv"
data = pd.read_csv(csv_file_path)

# Новый список для обновлённых данных
updated_rows = []

# Проход по всем изображениям
for file_name in os.listdir(image_folder):
    if file_name.endswith(".png"):  # Проверяем, что файл - изображение
        # Извлекаем participant_id из имени файла
        participant_id = file_name.split("_")[0]

        # Проверяем, существует ли participant_id в CSV
        if participant_id in data['participant_id'].astype(str).values:
            # Извлекаем строку из исходного CSV для этого participant_id
            matching_rows = data[data['participant_id'] == int(participant_id)]

            # Дублируем строку и добавляем путь до изображения
            for _, row in matching_rows.iterrows():
                new_row = row.copy()
                new_row['participant_id'] = os.path.join(image_folder, file_name)  # Сохраняем полный путь
                updated_rows.append(new_row)

# Создаём DataFrame из обновлённых данных
updated_data = pd.DataFrame(updated_rows)

# Сохраняем результат в новый CSV
output_csv_path = "updated_data_with_image_paths.csv"
updated_data.to_csv(output_csv_path, index=False)

print(f"Обновлённый CSV сохранён в файл: {output_csv_path}")

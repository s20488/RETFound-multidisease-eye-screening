import os
import pydicom


def check_image_sizes(directory, expected_size=(3456, 5184)):
    """
    Проверяет, соответствуют ли размеры всех DICOM-изображений в директории ожидаемому размеру.

    :param directory: Путь к директории с DICOM-файлами.
    :param expected_size: Ожидаемый размер изображений (высота, ширина).
    """
    mismatched_files = []  # Список файлов с несоответствующим размером

    for filename in os.listdir(directory):
        if filename.endswith(".dcm"):  # Проверяем только DICOM-файлы
            file_path = os.path.join(directory, filename)
            ds = pydicom.dcmread(file_path)
            image_array = ds.pixel_array

            if image_array.shape != expected_size:
                mismatched_files.append((filename, image_array.shape))

    if mismatched_files:
        print("Найдены изображения с несоответствующим размером:")
        for file, size in mismatched_files:
            print(f"Файл: {file}, Размер: {size}")
    else:
        print(f"Все изображения имеют ожидаемый размер: {expected_size}")


# Путь к директории с DICOM-изображениями
directory_path = "/mnt/data/cfi"

# Ожидаемый размер
expected_image_size = (3456, 5184)  # Высота, ширина

# Запуск проверки
check_image_sizes(directory_path, expected_size=expected_image_size)

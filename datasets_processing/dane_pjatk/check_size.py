import os
import pydicom


def check_image_sizes(directory, expected_size=(3456, 5184)):
    """
    Проверяет, соответствуют ли размеры всех DICOM-изображений в директории ожидаемому размеру.

    :param directory: Путь к директории с DICOM-файлами.
    :param expected_size: Ожидаемый размер изображений (высота, ширина).
    """
    mismatched_files = []  # Список файлов с несоответствующим размером
    error_files = []  # Файлы, которые не удалось обработать

    for filename in os.listdir(directory):
        if filename.endswith(".dcm"):  # Проверяем только DICOM-файлы
            file_path = os.path.join(directory, filename)
            try:
                ds = pydicom.dcmread(file_path)
                image_array = ds.pixel_array

                # Вывод размера текущего изображения
                print(f"Файл: {filename}, Размер: {image_array.shape}")

                if image_array.shape != expected_size:
                    mismatched_files.append((filename, image_array.shape))
            except Exception as e:
                error_files.append((filename, str(e)))

    if mismatched_files:
        print("\nНайдены изображения с несоответствующим размером:")
        for file, size in mismatched_files:
            print(f"Файл: {file}, Размер: {size}")
    else:
        print(f"\nВсе изображения имеют ожидаемый размер: {expected_size}")

    if error_files:
        print("\nФайлы с ошибками обработки:")
        for file, error in error_files:
            print(f"Файл: {file}, Ошибка: {error}")


# Путь к директории с DICOM-изображениями
directory_path = "/mnt/data/cfi"

# Ожидаемый размер
expected_image_size = (3456, 5184)  # Высота, ширина

# Запуск проверки
check_image_sizes(directory_path, expected_size=expected_image_size)

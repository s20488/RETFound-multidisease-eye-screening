import os
import pydicom


def check_image_sizes(directory, expected_size=(3456, 5184)):
    """
    Проверяет, соответствуют ли размеры всех DICOM-изображений в директории ожидаемому размеру.

    :param directory: Путь к директории с DICOM-файлами.
    :param expected_size: Ожидаемый размер изображений (высота, ширина).
    """
    error_files = []  # Файлы, которые не удалось обработать

    for filename in os.listdir(directory):
        if filename.endswith(".dcm"):  # Проверяем только DICOM-файлы
            file_path = os.path.join(directory, filename)
            try:
                # Читаем только заголовки DICOM для ускорения
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                rows = int(ds.get("Rows", 0))
                cols = int(ds.get("Columns", 0))

                # Проверяем размеры и выводим только отличающиеся
                if (rows, cols) != expected_size:
                    print(f"Файл: {filename}, Размер: ({rows}, {cols})")
            except Exception as e:
                error_files.append((filename, str(e)))


# Путь к директории с DICOM-изображениями
directory_path = "/mnt/data/cfi"

# Ожидаемый размер
expected_image_size = (3456, 5184)  # Высота, ширина

# Запуск проверки
check_image_sizes(directory_path, expected_size=expected_image_size)

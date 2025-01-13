from PIL import Image
import os
import pydicom
from torchvision import transforms

def remove_black_borders(image, border_fraction):
    # Обрезаем черные границы (по пропорциям)
    width, height = image.size
    border_width = int(width * border_fraction)
    border_height = int(height * border_fraction)

    cropped_image = image.crop((
        border_width,
        border_height,
        width - border_width,
        height - border_height
    ))
    return cropped_image

def crop_to_square(image):
    # Преобразуем изображение в квадрат (по меньшему измерению)
    width, height = image.size
    min_side = min(width, height)

    # Обрезаем так, чтобы изображение стало квадратным
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side

    return image.crop((left, top, right, bottom))

def process_dicom_images(input_dir, output_dir, border_fraction=0.04, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize(target_size),  # Сжимаем до целевого размера
    ])

    for filename in os.listdir(input_dir):
        if filename.endswith(".dcm"):
            dicom_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".dcm", ".png"))

            ds = pydicom.dcmread(dicom_path)
            image_array = ds.pixel_array

            image = Image.fromarray(image_array)

            # Сначала обрезаем черные границы
            image_without_borders = remove_black_borders(image, border_fraction)

            # Затем обрезаем до квадратной формы
            image_square = crop_to_square(image_without_borders)

            # Применяем изменение размера
            processed_image = transform(image_square)

            # Сохраняем изображение
            processed_image.save(output_path)

            print(f"Processed and saved: {output_path}")

# Путь к папкам
input_dir = r"/mnt/data/cfi"
output_dir = r"/mnt/data/cfi"

process_dicom_images(input_dir, output_dir, border_fraction=0.04, target_size=(224, 224))

import os
import pydicom
from torchvision import transforms
from PIL import Image


def remove_black_borders(image, border_fraction):
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


def process_dicom_images(input_dir, output_dir, border_fraction=0.04, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.CenterCrop(min(target_size)),
        transforms.Resize(target_size),
    ])

    for filename in os.listdir(input_dir):
        if filename.endswith(".dcm"):
            dicom_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".dcm", ".png"))

            ds = pydicom.dcmread(dicom_path)
            image_array = ds.pixel_array

            image = Image.fromarray(image_array)

            image_without_borders = remove_black_borders(image, border_fraction)

            processed_image = transform(image_without_borders)

            processed_image.save(output_path)

            print(f"Processed and saved: {output_path}")


input_dir = "/mnt/data/cfi"
output_dir = "/mnt/data/cfi"

process_dicom_images(input_dir, output_dir, border_fraction=0.04, target_size=(224, 224))

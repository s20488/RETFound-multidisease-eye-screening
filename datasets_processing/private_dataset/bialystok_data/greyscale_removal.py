import os
import pydicom
from concurrent.futures import ThreadPoolExecutor


def is_grayscale_dicom(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path)
        photometric_interpretation = ds.get("PhotometricInterpretation", None)
        if photometric_interpretation in ["MONOCHROME1", "MONOCHROME2"]:
            return True
        return False
    except Exception as e:
        print(f"Error reading DICOM file {dicom_path}: {e}")
        return False


def process_dicom(file_path):
    try:
        if is_grayscale_dicom(file_path):
            os.remove(file_path)
            print(f"Deleted black and white DICOM image: {file_path}")
    except Exception as e:
        print(f"Error in processing of {file_path}: {e}")


def remove_grayscale_dicom_parallel(directory):
    with ThreadPoolExecutor() as executor:
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
        executor.map(process_dicom, file_paths)


image_directory = "/mnt/data/cfi"
remove_grayscale_dicom_parallel(image_directory)

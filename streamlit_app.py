import os
import streamlit as st
import torch
from PIL import Image
import models_vit
import util.datasets
from util.pos_embed import interpolate_pos_embed
from timm.models.vision_transformer import trunc_normal_
import random
import numpy as np


# Ustawienie seeda dla reprodukowalności
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# A simplified class for passing parameters
class Args:
    def __init__(self):
        self.input_size = 224  # Rozmiar obrazu wejściowego


# Paths to model weights for each disease
WEIGHTS_PATHS = {
    "Nadciśnienie": "/mnt/data/Anastasiia_Ponkratova/RETFound_MAE/results/finetune_cfi_manual_hypertension_AHA/checkpoint-best.pth",
    "Cukrzyca": "/mnt/data/Anastasiia_Ponkratova/RETFound_MAE/results/finetune_cfi_manual_diabetes/checkpoint-best.pth",
    "Zaćma": "/mnt/data/Anastasiia_Ponkratova/RETFound_MAE/results/finetune_cfi_manual_cataract/checkpoint-best.pth",
    "Jaskra": "/mnt/data/Anastasiia_Ponkratova/RETFound_MAE/results/finetune_cfi_manual_glaucoma/checkpoint-best.pth",
}


# Check if weight files exist
def check_weights_files(weights_paths):
    for disease, path in weights_paths.items():
        if not os.path.exists(path):
            st.error(f"Plik wag dla choroby '{disease}' nie został znaleziony: {path}")
            st.stop()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model with the given weights
def load_model(weights_path):
    model = models_vit.__dict__['vit_large_patch16'](
        num_classes=2,
        drop_path_rate=0.2,
        global_pool=True,
    )

    checkpoint = torch.load(weights_path, map_location=device)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    # Remove incompatible keys from the checkpoint
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Usuwanie klucza {k} z wczytanego modelu")
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)

    trunc_normal_(model.head.weight, std=2e-5)

    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    return model


# Predict disease from the input image
def predict(image, model, args):
    # Create a transformation pipeline for the image
    transform = util.datasets.build_transform(is_train=False, args=args)
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    probability = torch.sigmoid(output[:, 1]).item()
    prediction = probability > 0.5

    return prediction, probability


# Main Streamlit application
def main():
    st.title("Analiza chorób dna oka")

    # Disease selection dropdown
    disease = st.selectbox("Wybierz chorobę", list(WEIGHTS_PATHS.keys()))

    # Image uploader
    uploaded_file = st.file_uploader("Prześlij obraz dna oka", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Przesłany obraz', use_column_width=True)

        # Analyze button
        if st.button("Analizuj"):
            weights_path = WEIGHTS_PATHS[disease]
            model = load_model(weights_path)

            # Create an Args object
            args = Args()
            result = predict(image, model, args)  # Pass args to predict
            if result:
                st.write(f"Wynik dla choroby '{disease}': **True** (Choroba wykryta)")
            else:
                st.write(f"Wynik dla choroby '{disease}': **False** (Choroba niewykryta)")


# Run the application
if __name__ == "__main__":
    check_weights_files(WEIGHTS_PATHS)
    main()

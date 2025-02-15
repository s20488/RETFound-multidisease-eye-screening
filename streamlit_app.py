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

set_seed(42)  # Ustalamy seed

# Uproszczona klasa do przekazywania parametrów
class Args:
    def __init__(self):
        self.input_size = 224  # Rozmiar obrazu wejściowego

# Ścieżki do wag modelu dla każdej choroby
WEIGHTS_PATHS = {
    "Nadciśnienie": "/mnt/data/Anastasiia_Ponkratova/RETFound_MAE/results/finetune_cfi_manual_hypertension_AHA/checkpoint-best.pth",
    "Cukrzyca": "/mnt/data/Anastasiia_Ponkratova/RETFound_MAE/results/finetune_cfi_manual_diabetes/checkpoint-best.pth",
    "Zaćma": "/mnt/data/Anastasiia_Ponkratova/RETFound_MAE/results/finetune_cfi_manual_cataract/checkpoint-best.pth",
    "Jaskra": "/mnt/data/Anastasiia_Ponkratova/RETFound_MAE/results/finetune_cfi_manual_glaucoma/checkpoint-best.pth",
}

# Sprawdzenie istnienia plików wag
def check_weights_files(weights_paths):
    for disease, path in weights_paths.items():
        if not os.path.exists(path):
            st.error(f"Plik wag dla choroby '{disease}' nie został znaleziony: {path}")
            st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ładowanie modelu z podanymi wagami
def load_model(weights_path):
    model = models_vit.__dict__['vit_large_patch16'](
        num_classes=2,
        drop_path_rate=0.2,
        global_pool=True,
    )

    checkpoint = torch.load(weights_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    # Usuwanie niezgodnych kluczy z checkpoint
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Usuwanie klucza {k} z wczytanego modelu")
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)

    # Inicjalizacja head.weight z ustalonym seedem
    trunc_normal_(model.head.weight, std=2e-5)

    # Ładowanie wag do modelu
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    return model

# Przewidywanie choroby na podstawie obrazu
def predict(image, model, args):
    # Przekształcenie obrazu
    transform = util.datasets.build_transform(is_train=False, args=args)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Przełączamy model w tryb inferencji
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    # Obliczanie prawdopodobieństwa i przewidywania
    probability = torch.sigmoid(output[:, 1]).item()
    prediction = probability > 0.5

    return prediction, probability

# Główna aplikacja Streamlit
def main():
    st.title("Analiza chorób dna oka")

    # Wybór choroby
    disease = st.selectbox("Wybierz chorobę", list(WEIGHTS_PATHS.keys()))

    # Przesyłanie obrazu
    uploaded_file = st.file_uploader("Prześlij obraz dna oka", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Przesłany obraz', use_column_width=True)

        # Przycisk analizy
        if st.button("Analizuj"):
            weights_path = WEIGHTS_PATHS[disease]
            model = load_model(weights_path)

            # Tworzymy obiekt Args
            args = Args()
            prediction, probability = predict(image, model, args)

            # Wyświetlamy wynik
            st.write(f"Prawdopodobieństwo dla choroby '{disease}': {probability:.4f}")
            if prediction:
                st.write(f"Wynik: **True** (Choroba wykryta)")
            else:
                st.write(f"Wynik: **False** (Choroba niewykryta)")

# Uruchomienie aplikacji
if __name__ == "__main__":
    check_weights_files(WEIGHTS_PATHS)
    main()

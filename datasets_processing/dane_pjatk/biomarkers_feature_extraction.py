import pandas as pd
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from gensim.models import FastText
import numpy as np

# Загрузка данных
data = pd.read_excel("C:/Users/Anastasiia/Desktop/Praca_dyplomowa/hypertension_task/hypertension_clustering.xlsx")

# Сохраняем original_columns, чтобы позже сравнить
original_columns = set(data.columns)

# Сохраняем participant_id
participant_id = data["participant_id"]
data = data.drop(columns=["participant_id"])

# Определяем категориальные, числовые и текстовые данные
categorical_columns = ["gender", "intm_w10_1", "intm_w10_2", "intm_w10_3", "intm_w10_5",
                       "intm_w10_6", "intm_w10_7", "intm_w10_8", "intm_w10_9", "intm_w10_11",
                       "intm_w10_12", "intm_sn1_1", "intm_sn1_3", "intm_sn1_5", "intm_sn13",
                       "intm_sn21", "intm_sn22_6", "intm_sn26", "intm_sn29", "intm_wo2_1",
                       "intm_wo4_1", "intm_wo4_2", "intm_wo4_4", "intm_wo4_6", "intm_wo6",
                       "intm_wo7", "intm_wo9", "intm_wo10", "intm_wo15_1", "inth1_h1d_99",
                       "inth1_h2f3_18_1", "inth1_h2f3_19_1", "rr_hand", "aha_bfnerv", "aha_maku",
                       "aha_bfmaku", "aha_netz", "aha_bfnetz", "aha_ve_ar", "aha_bf_ve_ar",
                       "aha_bfzusatz", "aha_faf", "aha_nerv_li", "aha_bfnerv_li", "aha_maku_li",
                       "aha_bfmaku_li", "aha_netz_li", "aha_bfnetz_li", "aha_ve_ar_li",
                       "aha_bf_ve_ar_li", "aha_bfzusatz_li", "aha_faf_li"]

text_columns = ["aha_bfnote", "aha_bfnote_li"]

numerical_columns = [
    "CASP-3_OID00630", "FAS_OID00615", "IL-6RA_OID00602", "LDL receptor_OID00564",
    "MCP-1_OID00576", "MMP-2_OID00614", "MMP-3_OID00644", "MMP-9_OID00568",
    "MPO_OID00600", "NT-proBNP_OID00131", "PECAM-1_OID00652", "RETN_OID00603",
    "SELE_OID00596", "SELP_OID00574", "TIMP4_OID00585", "TNF-R1_OID00649",
    "TNF-R2_OID00567", "vWF_OID00651", "rr_ps1", "rr_pd1", "rr_hr1", "rr_ps2",
    "rr_pd2", "rr_hr2", "rr_ps4", "rr_pd4", "rr_hr4", "rr_ps5", "rr_pd5", "rr_hr5",
    "L_AR_ARMedian_Cylinder", "L_AR_ARMedian_Sphere", "L_AR_ARPeriData_Axis",
    "L_AR_ARPeriData_Cylinder", "L_AR_ARPeriData_Sphere", "L_PS_PSList_Size",
    "L_AC_Sphere", "L_AC_MaxPS", "L_AC_MinPS", "L_RI_COIA", "L_RI_COIH", "L_RI_POI",
    "L_KM1_KMMedian_R1_Axis", "L_KM1_KMMedian_R1_Power", "L_KM1_KMMedian_R1_Radius",
    "L_KM1_KMMedian_R2_Axis", "L_KM1_KMMedian_R2_Power", "L_KM1_KMMedian_R2_Radius",
    "L_KM1_KMMedian_Average_Power", "L_KM1_KMMedian_Average_Radius",
    "L_KM1_KMMedian_KMCylinder_Axis", "L_KM1_KMMedian_KMCylinder_Power",
    "L_KM2_KMMedian_Average_Power", "L_KM2_KMMedian_Average_Radius",
    "L_KM2_KMMedian_KMCylinder_Axis", "L_KM2_KMMedian_KMCylinder_Power",
    "L_CS_CSList_Size", "PD_PDList_FarPD", "L_NT_NTAverage_mmHg",
    "L_NT_CorrectedIOP_Corrected_mmHg", "L_PACHY_PACHYAverage_Thickness",
    "R_AR_ARMedian_Axis", "R_AR_ARMedian_Cylinder", "R_AR_ARMedian_Sphere",
    "R_AR_ARPeriData_Axis", "R_AR_ARPeriData_Cylinder", "R_AR_ARPeriData_Sphere",
    "R_PS_PSList_Size", "R_AC_Sphere", "R_AC_MaxPS", "R_AC_MinPS", "R_RI_COIA",
    "R_RI_COIH", "R_RI_POI", "R_KM1_KMMedian_R1_Axis", "R_KM1_KMMedian_R1_Power",
    "R_KM1_KMMedian_R1_Radius", "R_KM1_KMMedian_R2_Axis", "R_KM1_KMMedian_R2_Power",
    "R_KM1_KMMedian_R2_Radius", "R_KM1_KMMedian_Average_Power",
    "R_KM1_KMMedian_Average_Radius", "R_KM1_KMMedian_KMCylinder_Axis",
    "R_KM1_KMMedian_KMCylinder_Power", "R_KM2_KMMedian_Average_Power",
    "R_KM2_KMMedian_Average_Radius", "R_KM2_KMMedian_KMCylinder_Axis",
    "R_KM2_KMMedian_KMCylinder_Power", "R_CS_CSList_Size", "R_NT_NTAverage_mmHg",
    "R_NT_CorrectedIOP_Corrected_mmHg", "R_PACHY_PACHYAverage_Thickness"
]


# --- ПРЕДОБРАБОТКА ТЕКСТА ---
def clean_text(text):
    """
    Очищает текст: убирает пунктуацию, приводит к нижнему регистру, удаляет лишние пробелы.
    """
    if pd.isna(text):  # Если текст отсутствует, возвращаем пустую строку
        return ""
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r"[^\w\s]", "", text)  # Удаление пунктуации
    text = re.sub(r"\s+", " ", text)  # Удаление лишних пробелов
    return text.strip()


# Применение очистки ко всем текстовым колонкам
for column in text_columns:
    data[column] = data[column].apply(clean_text)

# --- ОБРАБОТКА ПРОПУСКОВ ---
cat_imputer = SimpleImputer(strategy="most_frequent")
data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])

num_imputer = SimpleImputer(strategy="median")
data[numerical_columns] = num_imputer.fit_transform(data[numerical_columns])

# Преобразование категориальных данных в строки
data[categorical_columns] = data[categorical_columns].astype(str)

# --- ОБРАБОТКА ЧИСЛОВЫХ ДАННЫХ ---
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# --- КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ДАННЫХ ---
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_categorical = pd.DataFrame(
    encoder.fit_transform(data[categorical_columns]),
    columns=encoder.get_feature_names_out(categorical_columns),
    index=data.index
)


# --- ОБРАБОТКА ТЕКСТА С ПОМОЩЬЮ FASTTEXT ---
def prepare_text_embeddings(data, text_columns):
    """
    Создаёт FastText-эмбеддинги для указанных текстовых колонок.
    """
    tokenized_texts = data[text_columns].fillna("").apply(lambda x: x.str.split())

    # Проверяем, есть ли хоть какие-то токены
    sentences = tokenized_texts.sum().tolist()
    if not any(sentences):
        raise ValueError("Нет доступного текста для обучения модели.")

    # Обучение модели FastText
    model = FastText(sentences=sentences, vector_size=100, min_count=1, window=3, sg=1)

    # Создание эмбеддингов
    text_embeddings = []
    for column in text_columns:
        column_embeddings = tokenized_texts[column].apply(
            lambda tokens: model.wv[tokens].mean(axis=0) if tokens else np.zeros(100)
        )
        column_embeddings = pd.DataFrame(
            column_embeddings.tolist(),
            index=data.index,
            columns=[f"{column}_emb_{i}" for i in range(100)]
        )
        text_embeddings.append(column_embeddings)

    return pd.concat(text_embeddings, axis=1)


try:
    text_embeddings = prepare_text_embeddings(data, text_columns)
except ValueError as e:
    print(f"Ошибка обработки текста: {e}")
    text_embeddings = pd.DataFrame()  # Создаем пустую таблицу, если текста нет

# --- ОБЪЕДИНЕНИЕ ВСЕХ ДАННЫХ ---
processed_data = pd.concat([data[numerical_columns], encoded_categorical, text_embeddings], axis=1)
processed_data["participant_id"] = participant_id

# --- СОХРАНЕНИЕ РЕЗУЛЬТАТА ---
processed_data.to_csv(
    "C:/Users/Anastasiia/Desktop/Praca_dyplomowa/hypertension_task/hypertension_clustering_processed.csv", index=False)

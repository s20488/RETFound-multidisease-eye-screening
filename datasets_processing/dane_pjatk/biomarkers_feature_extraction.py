import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Шаг 1: Загрузка данных
data = pd.read_excel("C:/Users/Anastasiia/Desktop/Praca_dyplomowa/hypertension_task/hypertension_clustering.xlsx")

# Разделение колонок
categorical_columns = ["gender", "age", "intm_w10_1", "intm_w10_2", "intm_w10_3", "intm_w10_5",
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

numerical_columns = ["CASP-3_OID00630", "FAS_OID00615", "IL-6RA_OID00602", "LDL receptor_OID00564",
                     "MCP-1_OID00576", "MMP-2_OID00614", "MMP-3_OID00644", "MMP-9_OID00568",
                     "MPO_OID00600", "NT-proBNP_OID00131", "PECAM-1_OID00652", "RETN_OID00603",
                     "SELE_OID00596", "SELP_OID00574", "TIMP4_OID00585", "TNF-R1_OID00649",
                     "TNF-R2_OID00567", "vWF_OID00651", "rr_ps1", "rr_pd1", "rr_hr1", "rr_ps2",
                     "rr_pd2", "rr_hr2", "rr_ps4", "rr_pd4", "rr_hr4", "rr_ps5", "rr_pd5", "rr_hr5"]

# Сохраняем participant_id для последующего объединения
participant_id = data["participant_id"]

# Удаляем participant_id из данных для обработки
data = data.drop(columns=["participant_id"])

# Шаг 2: Обработка пропусков (NaN)
# Для числовых колонок: медиана
num_imputer = SimpleImputer(strategy="median")
data[numerical_columns] = num_imputer.fit_transform(data[numerical_columns])

# Для категориальных колонок: мода (наиболее частое значение)
cat_imputer = SimpleImputer(strategy="most_frequent")
data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])

# Для текстовых колонок: заполняем "unknown"
data[text_columns] = data[text_columns].fillna("unknown")

# Шаг 3: Масштабирование числовых данных
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Шаг 4: Кодирование категориальных данных
# Преобразуем категориальные данные в числовые с помощью OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_categorical = pd.DataFrame(encoder.fit_transform(data[categorical_columns]),
                                    columns=encoder.get_feature_names_out(categorical_columns),
                                    index=data.index)

# Удаляем исходные категориальные колонки и добавляем закодированные
data = data.drop(columns=categorical_columns)
data = pd.concat([data, encoded_categorical], axis=1)

# Шаг 5: Сохранение обработанных данных
data["participant_id"] = participant_id  # Возвращаем participant_id для идентификации
data.to_csv("C:/Users/Anastasiia/Desktop/Praca_dyplomowa/hypertension_task/processed_data.csv", index=False)

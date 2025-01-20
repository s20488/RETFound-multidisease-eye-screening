import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Загружаем данные
data = pd.read_csv(r"C:\Users\Anastasiia\Desktop\Praca_dyplomowa\diabetes_task\by features\dibetes_with_features.csv")

# 2. Сохраняем participant_id для дальнейшей идентификации
participant_ids = data['participant_id']  # Сохраняем колонку participant_id

# 3. Признаки и метки
X = data.drop(columns=['intm_cu1', 'participant_id'])  # Убираем метку и идентификатор
y = data['intm_cu1']  # Целевая переменная (0 — нет диабета, 1 — есть диабет)

# 4. Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Сохраняем обработанные данные
processed_data = pd.DataFrame(X_scaled, columns=X.columns)  # Признаки
processed_data['participant_id'] = participant_ids  # Возвращаем participant_id
processed_data['target'] = y.values  # Добавляем целевую переменную
processed_data.to_csv(r'C:\Users\Anastasiia\Desktop\Praca_dyplomowa\diabetes_task\by features\processed_data_with_id.csv', index=False)

print("Обработанные данные сохранены в файл 'processed_data_with_id.csv'.")

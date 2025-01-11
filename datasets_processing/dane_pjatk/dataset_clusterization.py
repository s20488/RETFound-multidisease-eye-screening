import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Установка глобального random seed
np.random.seed(42)

# Загрузка файла с признаками изображений и биомаркерами
data = pd.read_csv("C:/Users/Anastasiia/Desktop/Praca_dyplomowa/hypertension_task/final_data_with_image_features.csv")

# Удаляем ненужные столбцы
data_for_pca = data.drop(columns=["image_path", "participant_id"])

# Применение PCA с фиксированным random_state
pca = PCA(random_state=42)
reduced_data_full = pca.fit_transform(data_for_pca)

# Накопленная объясненная дисперсия
explained_variance = pca.explained_variance_ratio_.cumsum()
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Explained')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Choosing Number of Components')
plt.legend()
plt.grid()
plt.savefig("C:/Users/Anastasiia/Desktop/Praca_dyplomowa/hypertension_task/variance_explained.png", dpi=300)
plt.show()

# Оптимальное количество компонент для 90% объясненной дисперсии
optimal_components = next(i for i, v in enumerate(explained_variance) if v >= 0.9) + 1
print(f"Optimal number of components for 90% explained variance: {optimal_components}")

# Повторное применение PCA с оптимальным количеством компонент
pca = PCA(n_components=optimal_components, random_state=42)
reduced_data = pca.fit_transform(data_for_pca)

# Кластеризация KMeans с фиксированным random_state и количеством инициализаций
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(reduced_data)

# Добавляем кластеры в оригинальный датасет
data["cluster"] = clusters

# 2D-визуализация кластеров
pca_2d = PCA(n_components=2, random_state=42)
reduced_data_2d = pca_2d.fit_transform(data_for_pca)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=reduced_data_2d[:, 0],
    y=reduced_data_2d[:, 1],
    hue=clusters,
    palette='deep',
    style=clusters,
    markers=['o', 's', '^', 'P'],
    s=100,
    alpha=0.7
)
plt.title("2D Cluster Visualization (PCA)")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.legend(title="Clusters")
plt.grid(True)
plt.savefig("C:/Users/Anastasiia/Desktop/Praca_dyplomowa/hypertension_task/2d_cluster_visualization.png", dpi=300)
plt.show()

# 3D-визуализация кластеров
pca_3d = PCA(n_components=3, random_state=42)
reduced_data_3d = pca_3d.fit_transform(data_for_pca)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Цвета для кластеров
colors = sns.color_palette("deep", n_clusters)

for cluster_id in range(n_clusters):
    cluster_points = reduced_data_3d[clusters == cluster_id]
    ax.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        cluster_points[:, 2],
        s=50,
        alpha=0.7,
        color=colors[cluster_id],
        label=f'Cluster {cluster_id}'
    )

# Оформление графика
ax.set_title("3D Cluster Visualization (PCA)")
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.set_zlabel("Third Principal Component")
ax.legend(title="Clusters")
plt.savefig("C:/Users/Anastasiia/Desktop/Praca_dyplomowa/hypertension_task/3d_cluster_visualization.png", dpi=300)
plt.show()

# Сохраняем результат
data.to_csv("C:/Users/Anastasiia/Desktop/Praca_dyplomowa/hypertension_task/clustered_data.csv", index=False)

print("Processing completed. Data saved in 'clustered_data.csv'.")

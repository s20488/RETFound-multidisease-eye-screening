import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = '/mnt/data/cfi_manual_diabetes'

data = []

for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if os.path.isdir(category_path):
        for label in os.listdir(category_path):
            label_path = os.path.join(category_path, label)
            if os.path.isdir(label_path):
                count = len([f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))])
                data.append({"Category": category, "Label": label, "Count": count})

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=df, x='Category', y='Count', hue='Label', palette='viridis', ax=ax)

ax.set_axisbelow(True)
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.grid(axis='x', linestyle='-', alpha=0.7)

plt.xlabel("Kategorie", fontsize=15)
plt.ylabel("Ilość", fontsize=15)
plt.legend(
    title="Kategorie",
    labels=["Cukrzyca", "Brak chorób"],
    loc="upper right",
    fontsize=15
)

ax.tick_params(axis='both', which='major', labelsize=15)

plt.savefig('/mnt/data/dataset_distribution_diabetes.png', dpi=300)
plt.show()
plt.close()

import pandas as pd
import json
import matplotlib.pyplot as plt

csv_file_path = 'C:/Users/Anastasiia/Desktop/results/finetune_REFUGE2/_metrics_val.csv'
txt_file_path = 'C:/Users/Anastasiia/Desktop/results/finetune_REFUGE2/output_dir/log.txt'

csv_data = pd.read_csv(csv_file_path, header=None)
csv_data.columns = ['Acc', 'Sensitivity', 'Specificity', 'Precision', 'AUC-ROC', 'AUC-PR', 'F1', 'MCC', 'Loss']

csv_data['Loss_main'] = csv_data['Loss'].str.extract(r'^([\d.]+)').astype(float)
csv_data['Loss_in_brackets'] = csv_data['Loss'].str.extract(r'\(([\d.]+)\)').astype(float)

loss_in_brackets = csv_data['Loss_in_brackets'].to_numpy().flatten()

txt_data = []
with open(txt_file_path, 'r') as txt_file:
    for line in txt_file:
        txt_data.append(json.loads(line))
txt_df = pd.DataFrame(txt_data)

txt_df['epoch'] = pd.to_numeric(txt_df['epoch'], errors='coerce')
txt_df['train_loss'] = pd.to_numeric(txt_df['train_loss'], errors='coerce')
txt_df = txt_df.dropna()

x_values = txt_df['epoch'].to_numpy().flatten()
y_values = txt_df['train_loss'].to_numpy().flatten()

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='Training loss')
plt.plot(range(len(loss_in_brackets)), loss_in_brackets, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig('REFUGE2_loss_comparison.png', dpi=300)
plt.show()

from sklearn.preprocessing import LabelEncoder

labels = ['cataract', 'normal']
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
print(encoded_labels)  # [0, 1]
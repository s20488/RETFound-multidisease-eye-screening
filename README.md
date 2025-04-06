## Automated Detection of Systemic and Ophthalmic Pathologies Using RETFound in Fundus Imaging

This repository contains the implementation and research materials for a scientific-practical study employing deep learning methods to analyze retinal fundus images for automated diagnosis of systemic (hypertension, diabetes) and ophthalmic (cataracts, glaucoma, diabetic retinopathy, and age-related macular degeneration) diseases. Utilizing the advanced RETFound model - specifically designed for medical image analysis - this interdisciplinary project enables comprehensive detection of various ocular diseases and systemic conditions through fundus image interpretation.

### Key Features

- Interdisciplinary approuch: medicine + IT
- multiple diseases detection: identification of systemic and ophthalmic diseases
- State-of-the-art techologies: RETFond model
- Comprehensive datasets: private and public datasets
- Transfer learning: fine-turning approach was used

### Dataset Information

Private Dataset (with the support of Medical University of Białystok)
21,410 fundus images from 3,214 patients, 407 biomarkers were hand distributed and wewe divised manually on:
- hypertansion (n=12911)
- diabetes (n=2863)
- cataract (n=2206)
- glaucoma (n=745)

Public Datasets
- Kaggle https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data
- REFUGE2 https://refuge.grand-challenge.org/
- ADAM  https://amd.grand-challenge.org/

All datasets were distributed on 70% train, 15% val and 15% test. 

### Results 

The model achieved good performance across all datasets, espeshally on public dataset:

| Dataset      |  AUC  | Accuracy |
| ---          | ---   | ---      |
| Hypertension | 0.788 |   0.718  |
| Diabetes     | 0.813 |   0.739  |
| Glaucoma     | 0.856 |   0.777  |
| Cataracts    | 0.945 |   0.864  |
| Kaggle       | 0.990 |   0.947  |
| REFUGE2      | 0.976 |   0.947  |
| ADAM         | 0.976 |   0.945  |

## Visualization

For each dataset was build Confusion matrix, AUC ROC, PR AUC and distributin diagramms you can see [here](link).

Kaggle dataset visualization example is below:

<img src="./documents/flair.png" width = "750" alt="" align=center /> <br/>

### 📃Citation

```
@article{zhou2023foundation,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and Ayhan, Murat S and Williamson, Dominic J and Struyven, Robbert R and Liu, Timing and Xu, Moucheng and Lozano, Mateo G and Woodward-Court, Peter and others},
  journal={Nature},
  volume={622},
  number={7981},
  pages={156--163},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## Automated Detection of Systemic and Ophthalmic Pathologies Using RETFound in Fundus Imaging

This repository contains the implementation and research materials for a scientific-practical study employing deep learning methods to analyze retinal fundus images for automated diagnosis of systemic diseases (hypertension, diabetes) and ophthalmic pathologies (cataracts, glaucoma, diabetic retinopathy, and age-related macular degeneration). Utilizing the advanced RETFound model - specifically designed for medical image analysis - this interdisciplinary project enables comprehensive detection of various ocular diseases and systemic conditions through fundus image interpretation.

### Key Features

- Interdisciplinary approuch: medicine + IT
- multiple diseases detection: identification of systemic and ophthalmic diseases
- State-of-the-art techologies: RETFond model
- Comprehensive datasets: Private and public datasets
- Transfer learning: fine-turning approach

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

### Results 

The model achieved good performance across all datasets, espeshally on public dataset:

| Dataset      |  AUC  | Accuracy |
| ---          | ---   | ---      |
| Hypertension | 0.788 |   0.718  |
| Diabetes     | 0.813 |   0.739  |
| Cataracts    | 0.945 |   0.864  |
| Glaucoma     | 0.856 |   0.777  |
| Kaggle       | 0.990 |   0.947  |
| REFUGE2      | 0.976 |   0.947  |
| ADAM         | 0.976 |   0.945  |








Official repo for [RETFound: a foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x), which is based on [MAE](https://github.com/facebookresearch/mae):

Please contact 	**ykzhoua@gmail.com** or **yukun.zhou.19@ucl.ac.uk** if you have questions.

Keras version implemented by Yuka Kihara can be found [here](https://github.com/uw-biomedical-ml/RETFound_MAE)


### 📝Key features

- RETFound is pre-trained on 1.6 million retinal images with self-supervised learning
- RETFound has been validated in multiple disease detection tasks
- RETFound can be efficiently adapted to customised tasks





### 🌱Fine-tuning with RETFound weights

To fine tune RETFound on your own data, follow these steps:

1. Download the RETFound pre-trained weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">Colour fundus image</td>
<td align="center"><a href="https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view?usp=sharing">download</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">OCT</td>
<td align="center"><a href="https://drive.google.com/file/d/1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2/view?usp=sharing">download</a></td>
</tr>
</tbody></table>



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

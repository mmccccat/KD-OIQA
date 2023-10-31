**Omnidirectional Image Quality Assessment with Knowledge Distillation**

[![license](https://img.shields.io/badge/-LICENSE-green)](https://github.com/mmccccat/KD-OIQA/blob/main/LICENSE)
# Description
Pytorch implementation for the following paper:  
 [L. Liu, P. Ma, C. Wang and D. Xu, “Omnidirectional Image Quality Assessment with Knowledge Distillation,” IEEE Signal Processing Letters, doi:10.1109/LSP.2023.3327908.](https://doi.org/10.1109/LSP.2023.3327908)

# Requirements

Python 3.6  
torch 1.10.1  
torchvision 0.11.2  
numpy 1.19.5  
scipy 1.5.4  
pandas 1.1.5  
tensorborad 2.6.0  
tqdm 4.64.1  

# Usages

## 1. Data
We train and test KD-OIQA on three databases:  
CVIQ: [W. Sun, X. Min, G. Zhai, K. Gu, H. Duan and S. Ma, “MC360IQA: A Multi-channel CNN for Blind 360-Degree Image Quality Assessment,” IEEE Journal of Selected Topics in Signal Processing, vol. 14, no. 1, pp. 64-77, Jan. 2020, doi: 10.1109/JSTSP.2019.2955024](https://doi.org/10.1109/JSTSP.2019.2955024)  
OIQA: [H. Duan, G. Zhai, X. Min, Y. Zhu, Y. Fang and X. Yang, “Perceptual Quality Assessment of Omnidirectional Images,” in 2018 IEEE International Symposium on Circuits and Systems (ISCAS), 2018, pp. 1-5, doi: 10.1109/ISCAS.2018.8351786.](https://doi.org/10.1109/ISCAS.2018.8351786)  
IQA-ODI: [L. Yang, M. Xu, X. Deng and B. Feng, “Spatial Attention-Based Non-Reference Perceptual Quality Prediction Network for Omnidirectional Images,” in 2021 IEEE International Conference on Multimedia and Expo (ICME), 2021, pp. 1-6, doi: 10.1109/ICME51207.2021.9428390.](https://doi.org/10.1109/ICME51207.2021.9428390)  

## 2. Model Training and evaluation
We have provided the features extracted with the pretrained teacher network, which can be seen under the "features" directory.
Thus you can easily train the student network with following command:  
```
# train and evaluate the student network with mask distillation
python mask_train.py --database=OIQA
```

You can also train a new teacher network, but before that you need to extract viewport images to prepare the data:  
```
cd viewport
OIQA2viewport.m
```

In order to make the training process faster, we recommand save the resized viewport images (?→256×256) and ERP images (?→1024×512) in advance.  
We have provided a demo python script "img_resize.py" for resizing images.

Then you can train the teacher network with following command:  
```
# train the teacher network
python teacher_train.py --database=OIQA
```

In order to make the mask distillation training process faster, we also extract the features in advance with the pretrained teacher network.
```
# extract the features with the trained teacher network
python feat_save.py
```

## 3. Cross-database evaluation
You can evaluate the cross-database performance of KD-OIQA with following steps:
```
# note to modify the root path of the database to your own path
# train the teacher network
python teacher_crossdb_train.py --database_train=CVIQ --database_test=OIQA
# extract the features with the trained teacher network
python crossdb_feat.py --database_train=CVIQ --database_test=OIQA
# train the student network with mask distillation
# if you train on other database, note to modify the feat path in dataset_cross.py Line:42
python crossdb_train.py --database_train=CVIQ --database_test=OIQA
```

# Contact
lxliu@bit.edu.cn  
pcma@bit.edu.cn

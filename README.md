**Omnidirectional Image Quality Assessment with Knowledge Distillation**

# Description
Pytorch implementation for the following paper:  
 [L. Liu, P. Ma, C. Wang and D. Xu, "Omnidirectional Image Quality Assessment with Knowledge Distillation," IEEE Signal Processing Letters, doi:10.1109/LSP.2023.3327908.](https://doi.org/10.1109/LSP.2023.3327908)

# Requirements

**Python 3.6**
**torch 1.10.1**
**torchvision 0.11.2**
**numpy 1.19.5**
**scipy 1.5.4**
**pandas 1.1.5**
**tensorborad 2.6.0**
**tqdm 4.64.1**

# Usages

## 1. Train the Teacher network

```
# train the teacher network
python teacher_train.py
# extract the features with the trained teacher network
python feat_save.py
```
We provided the features extracted with the pretrained teacher network, which can be seen under the "features" directory.

## 2. Train a new model

```
# train the student network with mask distillation
python mask_train.py
```
We provided our train-test split results of different databases, which can be seen under the "data" directory.

## 3. Cross-database evaluation

```
# train the teacher network
python teacher_crossdb_train.py
# extract the features with the trained teacher network
python crossdb.feat.py
# train the student network with mask distillation
python crossdb_train.py
```

# Contact
lxliu@bit.edu.cn
pcma@bit.edu.cn

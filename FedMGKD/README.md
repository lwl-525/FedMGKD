# FedMGKD


## Overview
<p align="center">
  <img src="assets/overview.png" width="50%" alt="Overview">
</p>

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy

## Datasets
We conduct experiments on six datasets:
- MNIST
- CIFAR-10
- CIFAR-100
- FashionMNIST 
- OfficeCaltech10
- DomainNet

## Training
```
python main.py --dataset Cifar100 --num_clients 20 --global_epochs 200 --join_ratio 1.0 --partition dir --alpha 0.1 --test 4

```



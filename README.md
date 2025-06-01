# ALETHEIA: A Multi-Frequency Eddy Current Pulsed Thermography Dataset for Neural Operator Learning in Nondestructive Testing

## Abstract
Learning neural solvers for spatiotemporal partial differential equations (PDEs) under real-world constraints remains 
a key challenge in scientific machine learning, especially for inverse tasks with sparse and noisy boundary observations. 
We present the **Aletheia** dataset, the first 3D benchmark for learning data-driven solvers in the context of 
**nondestructive testing (NDT)**. The dataset simulates eddy-current-induced heating in conductive solids and models 
the resulting transient heat propagation governed by the heat equation. Aletheia contains over 4,700 high-resolution samples 
across 10 excitation frequencies (1–100\,kHz), each providing volumetric heat source and temperature fields over time. 
It supports both forward prediction of temperature evolution and inverse reconstruction of internal heat sources or defects 
from surface infrared measurements. Real infrared thermography data from cracked rail specimens are included for calibration 
and generalization studies. We define three canonical tasks on both regular and irregular grids and benchmark them using 
various neural operators. Aletheia establishes a unified platform for evaluating neural PDE solvers under 
realistic NDT conditions, enabling progress in reliable, data-driven inverse modeling.

## Datasets
You could get Aletheia Dataset from:

- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part2
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part3
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part4
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part5
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part6
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part7
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part8
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part9
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part10
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part11
- [Aletheia] https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part12

## Example
```bash
python main.py --data_root  /dataset path/ --data_num  600 --data_type unstructured_data --model FNO3d --lr 0.0001 --mode T2Q
```

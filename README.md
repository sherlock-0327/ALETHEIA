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

## Dataset Overview

The Aletheia dataset is designed to support neural operator learning for NDT applications, focusing on eddy current pulsed thermography (ECPT). Key features include:

- **Size**: Over 4,700 samples across 10 excitation frequencies, as described in the paper.
- **Data Types**: Volumetric heat source (Q) and temperature (T) simulated fields, surface temperature simulated data.
- **Tasks**: Three canonical tasks:
  - **T2Q**: Temperature-to-heat source reconstruction (inverse task).
  - **Q2T**: Heat-to-temperature prediction (forward task).
  - **T2T**: Temperature-to-temperature prediction.
- **Grid Types**: Regular and irregular grids, supporting both full-frequency and out-of-distribution (OOD) settings.
- **Metrics**: Evaluated using PDEBench and SSIM metrics.

The dataset is hosted on Kaggle and accessible via the GitHub repository, with scripts and documentation for training and evaluation.

## Datasets Access
The Aletheia dataset is split into multiple parts due to size constraints and is available on Kaggle:

- [Aletheia-part0](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset)
- [Aletheia-part1](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part2)
- [Aletheia-part2](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part3)
- [Aletheia-part3](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part4)
- [Aletheia-part4](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part5)
- [Aletheia-part5](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part6)
- [Aletheia-part6](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part7)
- [Aletheia-part7](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part81)
- [Aletheia-part8](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part8)
- [Aletheia-part9](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part9)
- [Aletheia-part10](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part10)
- [Aletheia-part11](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part11)
- [Aletheia-part12](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part12)

## Simulated Data Generation

1. **Create a Geometric Model**:
   - Construct a 3D model of a steel rail (100 × 70 × 45 mm) in COMSOL Multiphysics, incorporating customizable defects.
   - Define defect parameters to generate diverse defect cases, as described in paper.

2. **Set Material Properties**:
   - Assign standard physical properties for steel rails, including:
     - Electrical conductivity: 1.3 × 10^7 S/m
     - Thermal conductivity: 48 W/m·K
     - Density: 8000 kg/m³
     - Other thermal and electromagnetic properties as specified in paper.

3. **Define Electromagnetic Induction Heating**:
   - Configure coil parameters in COMSOL:
     - Excitation: Current-driven coil.
     - Wire model: Single wire.
     - Coil current: Variable parameter for 600 ms sinusoidal pulses at 1–100 kHz.
   - Apply electromagnetic field equations to simulate eddy current induction heating, as detailed in paper.

4. **Grid Generation**:
   - Generate a free tetrahedral grid, with finer meshing in defect areas and regions of eddy current penetration depth.
   - Implement adaptive grid refinement strategies to capture defect details accurately, as described in paper.

5. **Set Up the Solver**:
   - Use a time-dependent solver with a simulation time span of 0 to 0.6 seconds and a fixed time step of 0.05 seconds.
   - Employ the Backward Differentiation Formula (BDF) for time integration to ensure accurate capture of thermal diffusion dynamics.
   - Set boundary conditions:
     - Thermal convection with ambient air, using a heat transfer coefficient of 5 W/m²·K.

6. **Export Results**:
   - Output simulation results in VTU format, including volumetric heat source (Q) and temperature (T) fields for each of the 10 excitation frequencies.
   - Export both surface and internal data for all simulation cases, as detailed in paper.
   - Use the provided Python script `/comsol/comsol_control.py` to batch configure defect shapes and export results, enabling efficient generation of large-scale simulation samples.

7. **Alternative for Non-COMSOL Users**:
   - Preprocessed VTU files are available in the Kaggle dataset.
   - You can directly use our data in benchmark tasks.

This pipeline ensures reproducibility of the Aletheia dataset’s simulation. Detailed instructions and scripts are provided in the `/comsol` directory of the repository to facilitate replication.

## Example

**Running Benchmarks**:
   - This repository contains implementations of models such as FNO, Transolver and MLP for six tasks.
     - Supported `--data_type` options: `unstructured_data`, `structured_data`
     - Supported `--mode` options: `T2Q`, `Q2T`, `T2T`.
     - Supported `--model` options: `FNO3d`, `Transolver`, `MLP`, `DeepONet`, `LNO`, `FCNO`, `FFNO`, `GeoFNO`
     - Supported `--OOD` option: Out-Of-Distribution Training
   - Example command for the T2Q task using FNO:
     ```bash
     python main.py --data_root  /dataset path/ --data_num  600 --data_type unstructured_data --model FNO3d --lr 0.0001 --mode T2Q
     ```

**Evaluation**:
   - Evaluation scripts use PDEBench and SSIM metrics to reproduce results.
   - Run:
     ```bash
     python main.py --data_root  /dataset path/ --data_num  600 --data_type unstructured_data --model FNO3d --lr 0.0001 --mode T2Q --eval
     ```

## Compute Resources

The benchmark experiments were conducted using the following resources:
- **Hardware**: NVIDIA 4090 GPU, 24 GB memory.
- **Training Time**: Approximately 0.5 to 2 hours per model for 600 samples, depending on the task and model.


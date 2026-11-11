# ALETHEIA: A Multi-Frequency Eddy Current Pulsed Thermography Dataset for Neural Operator Learning in Nondestructive Testing

## Abstract

Learning neural solvers for spatiotemporal partial differential equations (PDEs) under real-world constraints remains a key challenge in scientific machine learning, especially for inverse tasks with sparse and noisy boundary observations. We present the Aletheia dataset, the first 3D benchmark for learning data-driven solvers in the context of nondestructive testing (NDT). The dataset simulates eddy-current-induced heating in conductive solids and models the resulting transient heat propagation governed by the heat equation. Aletheia contains over 4,700 high-resolution samples across 10 excitation frequencies (1-100 kHz), each providing volumetric heat source and temperature fields over time. It supports both forward prediction of temperature evolution and inverse reconstruction of internal heat sources or defects from surface infrared measurements. Real infrared thermography data from cracked rail specimens are included for calibration and comparison. We define four canonical tasks on both regular and irregular grids and benchmark them using various neural operators. Aletheia establishes a unified platform for evaluating neural PDE solvers under realistic NDT conditions, enabling progress in reliable, data-driven inverse modeling. The dataset and code are publicly available at https://github.com/sherlock-0327/ALETHEIA.

## Dataset Overview

The Aletheia dataset is designed to support neural operator learning for NDT applications, focusing on eddy current pulsed thermography (ECPT). Key features include:

- **Size**: Over 4,700 samples across 10 excitation frequencies, as described in the paper.
- **Data Types**: Volumetric heat source (Q) and temperature (T) simulated fields, surface temperature simulated data.
- **Tasks**: Four canonical tasks:
  - **T2Q**: Temperature-to-heat source reconstruction (inverse task).
  - **Q2T**: Heat-to-temperature prediction (forward task).
  - **T2T**: Temperature-to-temperature prediction.
  - **S2Q**: Volumetric heat source reconstruction from surface temperature observations.
- **Grid Types**: Regular and irregular grids, supporting both full-frequency and out-of-distribution (OOD) settings.
- **Metrics**: Evaluated using PDEBench and SSIM metrics.

The dataset is hosted on Kaggle and accessible via the GitHub repository, with scripts and documentation for training and evaluation.

## Datasets Access

The six defect types are distributed across the following Kaggle parts. The directory names match those used by the benchmark scripts.

| Type ID | Defect type | Directory | Kaggle downloads |
| --- | --- | --- | --- |
| 1 | Single-layer | `single_layer` | [Part 0](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset) |
| 2 | Type I double-layer | `typeI_double-layer` | [Part 2](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part3), [Part 11](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part11), [Part 12](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part12) |
| 3 | Type II double-layer | `typeII_double-layer` | [Part 2](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part3) |
| 4 | Type III double-layer | `typeIII_double-layer` | [Part 3](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part4), [Part 4](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part5), [Part 5](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part6), [Part 6](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part7), [Part 7](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part81) |
| 5 | Type I multi-layer | `typeI_multi-layer` | [Part 1](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part2) |
| 6 | Type II multi-layer | `typeII_multi-layer` | [Part 8](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part8), [Part 9](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part9), [Part 10](https://www.kaggle.com/datasets/xiaojie1232/ecptdataset-part10) |

Part 2 contains both Type I and Type II double-layer cases. After extraction, place the simulation case folders directly under their corresponding defect-type directories in a common dataset root. The loader reads `structured_data/` or `unstructured_data/` within each case folder and uses `surf_input/` for S2Q.

## Simulated Data Generation

See the [COMSOL README](comsol/README.md) for the simulation workflow and the two experimental infrared thermography recordings.

## Example

**Running Benchmarks**:
   - This repository contains implementations of models such as FNO, Transolver and MLP for four tasks.
     - Supported `--data_type` options: `unstructured_data`, `structured_data`
     - Supported `--mode` options: `T2Q`, `Q2T`, `T2T`, `S2Q`.
     - For S2Q, pass `--mode S2Q --use_surf` to load surface temperature observations.
     - Supported `--model` options: `FNO3d`, `Transolver`, `MLP`, `DeepONet`, `LNO`, `FCNO`, `FFNO`, `GeoFNO`
     - Supported `--OOD` option: Out-Of-Distribution Training
   - The examples use 600 samples, with 20% held out for testing.
   - Example command for the T2Q task using FNO:
     ```bash
     DATA_ROOT="/path/to/Aletheia"
     python main.py --data_root "$DATA_ROOT/typeI_double-layer" --crack_type 2 \
       --data_num 600 --test_split 0.2 --data_type unstructured_data \
       --model FNO3d --lr 0.0001 --mode T2Q --in_channels 13 --out_channels 1
     ```

**Evaluation**:
   - Evaluation scripts use PDEBench and SSIM metrics to reproduce results.
   - After training, use the same settings with `--eval` to load the saved checkpoint:
     ```bash
     python main.py --data_root "$DATA_ROOT/typeI_double-layer" --crack_type 2 \
       --data_num 600 --test_split 0.2 --data_type unstructured_data \
       --model FNO3d --lr 0.0001 --mode T2Q --in_channels 13 --out_channels 1 --eval
     ```
   - Confirm that the output reports a successful checkpoint load before using the metrics.

**Running the benchmark scripts**

Run the scripts from the repository root in a Bash shell with the benchmark Python environment activated. Set `DATA_BASE` to the common dataset root and `GPU` to your GPU index in the selected script. For geometric OOD, set `TRAIN_DIRS` to the five training defect directories, `TEST_DIR` to the held-out defect directory, and `TEST_CRACK_TYPE` to its type ID. Quote paths that contain spaces.

| Experiment | Python entry point | S2Q command |
| --- | --- | --- |
| Single defect type, full-frequency and frequency-band OOD | [`run_all_models_exps.py`](run_all_models_exps.py) | `bash scripts/run_all_models_S2Q.sh` |
| Mixed defect types | [`multi_crack_exps.py`](multi_crack_exps.py) | `bash scripts/run_multi_crack_S2Q.sh` |
| Single-frequency OOD | [`run_all_models_exps.py`](run_all_models_exps.py) | `bash scripts/run_S2Q_OOD_sfo.sh` |
| Geometric OOD | [`cracks_OOD_exps.py`](cracks_OOD_exps.py) | `bash scripts/run_S2Q_OOD-fsplit.sh` |

The [`scripts/`](scripts/) directory contains the corresponding launchers for T2Q, Q2T, and T2T. For a direct `main.py` run, set `--in_channels` and `--out_channels` to `13 → 1` for T2Q or S2Q, `1 → 13` for Q2T, and `11 → 2` for T2T.

## Compute Resources

The benchmark experiments were conducted using the following resources:
- **Hardware**: NVIDIA 4090 GPU, 24 GB memory.
- **Training Time**: Approximately 0.5 to 2 hours per model for 600 samples, depending on the task and model.

## Citation

Changbin Sun, Jie Xiao, Xiaotian Chen, and Yuankai Wu. *ALETHEIA: A Multi-Frequency Eddy Current Pulsed Thermography Dataset for Neural Operator Learning in Nondestructive Testing*. NeurIPS 2026, Evaluations and Datasets Track.

[arXiv:2611.11252](https://arxiv.org/abs/2611.11252)

```bibtex
@misc{sun2026aletheiamultifrequencyeddycurrent,
  title={ALETHEIA: A Multi-Frequency Eddy Current Pulsed Thermography Dataset for Neural Operator Learning in Nondestructive Testing},
  author={Changbin Sun and Jie Xiao and Xiaotian Chen and Yuankai Wu},
  year={2026},
  eprint={2611.11252},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2611.11252},
}
```


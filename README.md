# Rethinking Molecular OOD Generalization via Target-Aware Source Selection

[![Dataset](https://img.shields.io/badge/Dataset-SCOPE--Bench-blue)](#)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS_2026-green)](#)


This repository contains the official PyTorch implementation for the paper: **"Rethinking Molecular OOD Generalization via Target-Aware Source Selection"** (Submitted to NeurIPS 2026).

## 📌 Overview

Robust prediction of molecular properties under extreme out-of-distribution (OOD) scenarios is a pivotal bottleneck in AI-driven drug discovery. In this repository, we provide:
1. **SCOPE-Bench**: A rigorous OOD benchmark based on explicit physicochemical clustering that eliminates evaluation biases and shortcut learning.
2. **POMA**: A policy-driven framework (**Policy Optimization for Multi-source Adaptation**) that overcomes negative transfer through a combinatorial selection policy and dual-scale decoupled domain adaptation.

## 📂 Codebase Structure

Our repository is organized to ensure reproducibility and independent execution:
- `SCOPE_Bench/`: Scripts for generating the strict OOD data splits based on physicochemical descriptors.
- `SupervisedLearning/`: Implementation of the pure supervised baselines for 3 SOTA 3D equivariant backbones (ViSNet, ETNN, GotenNet).
- `POMA/`: The core implementation of our policy-driven pipeline, decoupled into:
  - `POMA_GRPO`: One-time offline policy optimization via Group Relative Policy Optimization.
  - `POMA_DA`: Online dual-scale decoupled multi-source domain adaptation on target tasks.

*(Note: To ensure the independent execution of different training protocols without environment variable conflicts, the core backbone files are intentionally duplicated within their respective execution folders.)*

## ⚙️ Environment Installation

Due to the specific topological dependencies of ETNN and the general 3D equivariant requirements of ViSNet/GotenNet, we provide two separate requirement files based on our exact experimental setup.

**Option A: For ETNN (Python 3.10)**
*Includes topological data analysis dependencies like `gudhi` and `toponetx`.*
```bash
conda create -n etnn_env python=3.10
conda activate etnn_env
pip install -r requirements_etnn.txt
```

**Option B: For ViSNet & GotenNet (Python 3.8)**
```bash
conda create -n visnet_env python=3.8
conda activate visnet_env
pip install -r requirements_visnet.txt
```

## 🚀 Usage

### 1. Data Preparation (SCOPE-Bench)
You can generate the pre-processed dataset locally:
```bash
cd SCOPE_Bench
python splitter.py 
```

### 2. Running Supervised Baselines
To train and evaluate the pure supervised models (e.g., ETNN):
```bash
cd SupervisedLearning/etnn
python train.py --config config.py 
```

### 3. Running the POMA Framework
**Phase 1: Offline Policy Optimization (GRPO)**
Train the RL combinatorial selection policy on source-domain proxy targets:
```bash
cd POMA/etnn/POMA_GRPO
python controller.py --config config.py
```

**Phase 2: Online Task Adaptation (MSDA)**
Load the pre-trained policy to assign optimal source subsets and perform dual-scale domain adaptation without validation set intervention:
```bash
cd POMA/etnn/POMA_DA
python train.py --config config.py
```

## 📊 Main Results

Overall Mean Absolute Error (MAE, eV) measured on the QM9 dataset using SCOPE-Bench splits (with a fixed random seed of 42). Lower is better.

| Method | Property | ViSNet | ETNN | GotenNet |
| :--- | :---: | :---: | :---: | :---: |
| Supervised | HOMO $\downarrow$ | 0.1621 | 0.1349 | 0.1580 |
| **POMA (Ours)** | HOMO $\downarrow$ | **0.1540** | **0.1233** | **0.1535** |
| | | *Gain: +5.0%* | *Gain: +8.6%* | *Gain: +2.8%* |
| Supervised | LUMO $\downarrow$ | 0.1829 | 0.1312 | 0.1902 |
| **POMA (Ours)** | LUMO $\downarrow$ | **0.1673** | **0.1165** | **0.1728** |
| | | *Gain: +8.5%* | *Gain: +11.2%* | *Gain: +9.1%* |
| Supervised | Gap $\downarrow$ | 0.2303 | 0.1814 | 0.2059 |
| **POMA (Ours)** | Gap $\downarrow$ | **0.2270** | **0.1681** | **0.2021** |
| | | *Gain: +1.4%* | *Gain: +7.3%* | *Gain: +1.8%* |

## 📄 License
This project is licensed under the MIT License.

<br>
<hr>
<br>

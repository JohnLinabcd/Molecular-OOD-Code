# Learning to Select Source Domains: Proxy-Rewarded Policy Optimization for Molecular OOD Generalization

[![Dataset](https://img.shields.io/badge/Dataset-SCOPE--Bench-blue)](#)

This repository contains the official PyTorch implementation for the paper: **"Learning to Select Source Domains: Proxy-Rewarded Policy Optimization for Molecular OOD Generalization"** .

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

### Physicochemical Descriptor Representation

Each molecular scaffold is represented by four high-level physicochemical descriptor groups:

1. `V_macro`: Global topology-related descriptors.
2. `V_element`: Elemental composition and polarizability-related descriptors.
3. `V_conn`: Connectivity, electronic delocalization, and rigidity-related descriptors.
4. `V_flex`: Conformational flexibility-related descriptors.

These four conceptual descriptor groups contain multiple numerical subfeatures. After concatenation, they form an **11-dimensional numerical descriptor vector**:

\[
f_k=V_{\mathrm{macro}}\oplus V_{\mathrm{element}}
\oplus V_{\mathrm{conn}}\oplus V_{\mathrm{flex}}
\in\mathbb{R}^{11}.
\]

Thus, four refers to the number of high-level physicochemical descriptor groups, while eleven refers to the dimensionality of the expanded numerical representation.

### SCOPE-Bench Partition

The final SCOPE-Bench benchmark consists of **12 disjoint scaffold clusters**, partitioned as follows:

| Partition | Number of Clusters |
| :--- | :---: |
| Source | 6 |
| Validation | 1 |
| Target | 5 |
| **Total** | **12** |

The target partition is further used to construct 15 target-scaffold evaluation tasks.

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
python train.py
```

### 3. Running the POMA Framework

**Phase 1: Offline Policy Optimization (GRPO)**

Train the RL combinatorial selection policy on source-domain proxy targets.

The actual GRPO training entry point is `lookfor.py`:
```bash
cd POMA/etnn/POMA_GRPO
python lookfor.py
```

**Phase 2: Online Task Adaptation (MSDA)**

Load the pre-trained policy to assign optimal source subsets and perform dual-scale domain adaptation without validation set intervention:
```bash
cd POMA/etnn/POMA_DA
python train.py
```

### Checkpoint Selection and Final Results

The target-adaptation implementation generates two types of model checkpoints:

- **`best_model.pt`**: A monitoring checkpoint saved when the monitored target MAE reaches a new minimum during training. It is retained for monitoring purposes and is **not used to generate the POMA results reported in the paper**.
- **`last_model.pth`**: The final checkpoint saved after the predetermined training schedule is completed. This is the checkpoint **used for the reported POMA predictions and quantitative results**.

The reported results therefore use the final scheduled checkpoint rather than selecting the best-performing epoch based on target MAE.

*Implementation note: In the current ETNN training implementation, target-test MAE is evaluated during training for monitoring and for saving `best_model.pt`. This monitoring value does not determine the final `last_model.pth` checkpoint, optimizer updates, or adaptive loss-weight updates.*

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

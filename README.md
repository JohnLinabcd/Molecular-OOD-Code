# Rethinking Molecular OOD Generalization via Target-Aware Source Selection

[![Dataset](https://img.shields.io/badge/Dataset-SCOPE--Bench-blue)](#)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS_2026-green)](#)

[👉 **点击这里查看中文版 README (Chinese Version)**](#中文版-chinese-version)

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

<h2 id="中文版-chinese-version">🇨🇳 中文版</h2>

# 通过目标感知源域选择反思分子分布外泛化

本仓库包含论文 **"Rethinking Molecular OOD Generalization via Target-Aware Source Selection"** (NeurIPS 2026 投稿) 的官方 PyTorch 实现。

## 📌 概览

在极端分布外（OOD）场景下对分子属性进行鲁棒预测是 AI 驱动药物发现的核心瓶颈。在本仓库中，我们提供：
1. **SCOPE-Bench**: 一个基于理化描述符显式聚类构建的严格 OOD 评估基准，彻底消除了评估偏差与捷径学习。
2. **POMA**: 一种策略驱动的自适应框架，通过组合源域选择策略和双尺度解耦域自适应，有效克服了负迁移。

## 📂 代码结构

- `SCOPE_Bench/`: 用于生成严格 OOD 数据划分的脚本。
- `SupervisedLearning/`: 3 种 SOTA 3D 几何等变主干网络的纯监督基线实现。
- `POMA/`: 本文核心策略管道的实现，解耦为：
  - `POMA_GRPO`: 使用 Group Relative Policy Optimization 进行的一次性离线策略寻优。
  - `POMA_DA`: 针对真实目标任务的在线 dual-scale 解耦多源域自适应（MSDA）。

## ⚙️ 环境安装

由于 ETNN 依赖拓扑数据分析库，而 ViSNet/GotenNet 环境有所不同，我们根据真实的实验环境提供了两份不同的依赖文件：

**选项 A: 用于 ETNN (Python 3.10)**
*包含 `gudhi`, `toponetx` 等拓扑图神经网络必备依赖。*
```bash
conda create -n etnn_env python=3.10
conda activate etnn_env
pip install -r requirements_etnn.txt
```

**选项 B: 用于 ViSNet 与 GotenNet (Python 3.8)**
```bash
conda create -n visnet_env python=3.8
conda activate visnet_env
pip install -r requirements_visnet.txt
```

## 🚀 运行指南

### 1. 数据准备
运行以下脚本在本地生成预处理数据集：
```bash
cd SCOPE_Bench
python splitter.py 
```

### 2. 运行纯监督基线
```bash
cd SupervisedLearning/etnn
python train.py --config config.py 
```

### 3. 运行 POMA 框架
**第一阶段：离线策略寻优 (GRPO)**
```bash
cd POMA/etnn/POMA_GRPO
python controller.py --config config.py
```

**第二阶段：在线任务自适应 (MSDA)**
```bash
cd POMA/etnn/POMA_DA
python train.py --config config.py
```

# Rethinking Molecular OOD Generalization via Target-Aware Source Selection

[![Dataset](https://img.shields.io/badge/Dataset-SCOPE--Bench-blue)](https://huggingface.co/datasets/tempresearch00/SCOPE-BENCH)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS_2026-green)](#)

[👉 **点击这里查看中文版 README (Chinese Version)**](#中文版-chinese-version)

This repository contains the official PyTorch implementation for the paper: **"Rethinking Molecular OOD Generalization via Target-Aware Source Selection"** (Submitted to NeurIPS 2026).

## 📌 Overview

Robust prediction of molecular properties under extreme out-of-distribution (OOD) scenarios is a pivotal bottleneck in AI-driven drug discovery. In this repository, we provide:
1. **SCOPE-Bench**: A rigorous OOD benchmark based on physicochemical clustering that eliminates evaluation biases (shortcut learning).
2. **SOMA**: A reinforcement learning-guided framework (Strategic Optimization for Multi-source Adaptation) that overcomes negative transfer through target-aware combinatorial source selection.

## 📂 Codebase Structure

Our repository is organized to ensure reproducibility and independent execution:
- `SCOPE_Bench/`: Scripts for generating the strict OOD data splits (clustering based on physicochemical descriptors).
- `SupervisedLearning/`: Implementation of the pure supervised baselines for 3 SOTA 3D equivariant backbones (ViSNet, ETNN, GotenNet).
- `SOMA/`: The core implementation of our framework, decoupled into:
  - `SOMA_GRPO`: One-time offline policy optimization using RL (Round-robin proxy updates).
  - `SOMA_DA`: Online dual-scale multi-source domain adaptation (MSDA) on target tasks.

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
You can directly download our pre-processed dataset from [Hugging Face](https://huggingface.co/datasets/tempresearch00/SCOPE-BENCH), or generate it locally:
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

### 3. Running the SOMA Framework
**Phase 1: Offline Policy Optimization (GRPO)**
Train the RL policy network on source-domain proxy targets:
```bash
cd SOMA/etnn/SOMA_GRPO
python controller.py --config config.py
```

**Phase 2: Online Task Adaptation (MSDA)**
Load the pre-trained policy to assign source subsets and perform target domain adaptation without validation set intervention:
```bash
cd SOMA/etnn/SOMA_DA
python train.py --config config.py
```

## 📊 Main Results

Mean Absolute Error (MAE, eV) on SCOPE-Bench (QM9), averaged over 3 seeds. Lower is better.

| Method | Property | ViSNet | ETNN | GotenNet |
| :--- | :---: | :---: | :---: | :---: |
| Supervised | HOMO $\downarrow$ | 0.1673 | 0.1398 | 0.1659 |
| **SOMA (Ours)** | HOMO $\downarrow$ | **0.1547** | **0.1241** | **0.1594** |
| | | *Gain: +7.5%* | *Gain: +11.2%* | *Gain: +3.9%* |
| Supervised | LUMO $\downarrow$ | 0.1921 | 0.1412 | 0.2046 |
| **SOMA (Ours)** | LUMO $\downarrow$ | **0.1868** | **0.1283** | **0.1926** |
| | | *Gain: +2.8%* | *Gain: +9.1%* | *Gain: +5.9%* |
| Supervised | Gap $\downarrow$ | 0.2398 | 0.1659 | 0.2121 |
| **SOMA (Ours)** | Gap $\downarrow$ | 0.2405 | **0.1804** | **0.2099** |
| | | *Gain: -0.3%* | *Gain: +9.1%* | *Gain: +1.0%* |

## 📄 License
This project is licensed under the MIT License.

<br>
<hr>
<br>

<h2 id="中文版-chinese-version">🇨🇳 中文版 (Chinese Version)</h2>

# 通过目标感知源域选择反思分子分布外泛化

本仓库包含论文 **"Rethinking Molecular OOD Generalization via Target-Aware Source Selection"** (NeurIPS 2026 投稿) 的官方 PyTorch 实现。

## 📌 概览

在极端分布外（OOD）场景下对分子属性进行鲁棒预测是 AI 驱动药物发现的核心瓶颈。在本仓库中，我们提供：
1. **SCOPE-Bench**: 一个基于理化描述符聚类构建的严格 OOD 评估基准，彻底消除了常规拆分中的子结构语义重叠（捷径学习）。
2. **SOMA**: 一种基于强化学习引导的域自适应框架，通过目标感知组合源域选择，有效克服了负迁移。

* **数据集下载:** [Hugging Face: SCOPE-BENCH](https://huggingface.co/datasets/tempresearch00/SCOPE-BENCH)

## 📂 代码结构

- `SCOPE_Bench/`: 用于生成严格 OOD 数据划分的脚本。
- `SupervisedLearning/`: 3 种 SOTA 3D 几何等变主干网络的纯监督基线实现。
- `SOMA/`: 本文核心框架的实现，解耦为：
  - `SOMA_GRPO`: 使用强化学习进行的一次性离线策略寻优。
  - `SOMA_DA`: 针对真实目标任务的在线双尺度多源域自适应（MSDA）。

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

### 1. 数据准备 (SCOPE-Bench)
可直接从 Hugging Face 下载数据集，或在本地运行：
```bash
cd SCOPE_Bench
python splitter.py 
```

### 2. 运行纯监督基线
```bash
cd SupervisedLearning/etnn
python train.py --config config.py 
```

### 3. 运行 SOMA 框架
**第一阶段：离线策略寻优 (GRPO)**
```bash
cd SOMA/etnn/SOMA_GRPO
python controller.py --config config.py
```

**第二阶段：在线任务适配 (MSDA)**
```bash
cd SOMA/etnn/SOMA_DA
python train.py --config config.py
```

## 📄 许可证
本项目采用 MIT 许可证。

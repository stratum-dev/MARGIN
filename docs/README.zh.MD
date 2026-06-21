![](./assets/logo.png)

[English](../README.MD) | [中文](./README.zh.MD) | [Français](./README.fr.MD) | [Español](./README.es.MD) | [Русский](./README.ru.MD) | [العربية](./README.ar.MD)

**M**argin-**A**ware **R**egularized **G**eometry for **I**mbalance Vulnerability Detectio**N**

论文官方实现。本仓库提供完整的训练与评估流程，供复现使用。

## 摘要

![](./assets/banner.png)

软件漏洞检测对保障软件安全与可靠性至关重要。尽管深度学习技术取得了显著进展，真实世界的漏洞数据集仍面临两大严峻挑战：频率不平衡与难度不平衡。我们从嵌入几何的视角重新审视这些挑战，发现这种不平衡会在超球面表示空间中引发几何扭曲。为解决这一问题，我们提出了 MARGIN，一个基于度量学习的框架，通过自适应间隔度量学习和超球面原型建模来学习具有判别力的漏洞表示。MARGIN 根据 von Mises-Fisher 浓度估计的分布结构动态调整几何正则化，使嵌入分布的概率质量与对应的 Voronoi 单元对齐，从而减少几何扭曲并产生更稳定的决策边界。在公开漏洞数据集上的大量实验表明，MARGIN 始终优于强基线方法，在分类和检测任务上均取得了显著提升，尤其在具有挑战性的不平衡数据集上表现突出。进一步分析表明，MARGIN 能产生结构更优的嵌入几何，提高了模型的鲁棒性、可解释性和泛化能力。

## 环境要求

- **GPU 显存**：≥ 24 GB
- **CUDA**：12.4 运行时（驱动版本 ≥ 525 可能通过向下兼容方式运行）
- **CPU**：x86-64，≥ 4 核
- **内存**：≥ 8 GB
- **磁盘空间**：数据集 ≥ 2 GB，骨干模型 ≥ 5 GB
- **操作系统**：Linux（已在 Ubuntu 22.04 / 24.04 上测试）

## 安装

### 1. 创建 conda 环境

```bash
conda env create -f environment.yml
conda activate margin
```

### 2. 安装 PyTorch

（CUDA 12.4 版本 — 托管于 pytorch.org，非 PyPI）

```bash
pip install torch==2.6.0+cu124          \
            torchvision==0.21.0+cu124   \
            torchaudio==2.6.0+cu124     \
            --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装 HuggingFace CLI

```bash
pip install -U "huggingface_hub"
```

## 下载

数据集和骨干模型托管于 HuggingFace。您可通过 CLI 工具手动下载。

### 数据集

```bash
hf download codemetic/MARGIN --repo-type=dataset
```

### 骨干模型

```bash
hf download Salesforce/codet5-base
```

> **中国大陆用户注意：** 在执行任何 `hf download` 命令前，请设置 HuggingFace 镜像端点：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

## 训练

运行 `python main.py --help` 查看所有可用参数。相关超参数已在论文中给出。

训练命令示例：

```bash
python  main.py                                           \
        --dataset_subset          bigvul                  \
        --backbone_name           Salesforce/codet5-base  \
        --max_epochs              200                     \
        --max_checkpoints         3                       \
        --early_stopping_patience 30                      \
        --base_scale              20                      \
        --device                  cuda:0                  \
        --seed                    42                      \
        --batch_size              16
```

**注意：** 由于平台架构、硬件及软件版本的差异，训练和评估结果可能与原论文略有不同。

## 评估

```bash
python eval.py
```

交互式命令行界面 — 使用 `↑` / `↓` 选择训练运行和检查点，然后在评估开始前配置批次大小、随机种子和设备。

## 引用

```bibtex
@misc{zhang2026MARGIN,
      title={MARGIN: Margin-Aware Regularized Geometry for Imbalanced Vulnerability Detection}, 
      author={Yuteng Zhang and Huifang Ma and Jiahui Wei and Qingqing Li and Yafei Yang},
      year={2026},
      eprint={2605.10240},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2605.10240}, 
}
```

## 许可证

本仓库中的源代码和编程实现采用 **CC BY 4.0** 许可证。

本项目引用或使用的数据集**不**包含在该许可范围内。所有这些数据集仍为其各自所有者的知识产权，并按其自身的许可、使用条款和访问条件进行分发。用户在下载、使用或重新分发这些数据集时，有责任遵守原始数据集提供者的适用许可要求。

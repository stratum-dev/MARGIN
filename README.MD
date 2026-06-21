![](./assets/logo.png)

**M**argin-**A**ware **R**egularized **G**eometry for **I**mbalance Vulnerability Detectio**N**

Official implementation of the paper. This artifact repository is for reproduction and it offers the training, evaluation pipelines.

## Abstract

![](./assets/banner.png)

Software vulnerability detection is critical for ensuring software security and reliability. Despite recent advances in deep learning, real-world vulnerability datasets suffer from two severe challenges: frequency imbalance and difficulty imbalance. We reinterpret these challenges from an embedding geometry perspective, observing that such imbalances induce geometric distortions in hyperspherical representation space. To address this issue, we propose MARGIN, a metric-based framework that learns discriminative vulnerability representations through adaptive margin metric learning and hyperspherical prototype modeling. MARGIN dynamically adjusts geometric regularization according to the distribution structure estimated by the von Mises-Fisher concentration, aligning the probability mass of embedding distributions with their corresponding Voronoi cells, thereby reducing geometric distortion and yielding more stable decision boundaries. Extensive experiments on public vulnerability datasets show that MARGIN consistently outperforms strong baselines, achieving notable improvements in classification and detection, especially on challenging, imbalanced datasets. Further analysis demonstrates that MARGIN produces more structured embedding geometries, improving robustness, interpretability, and generalization.

## Requirements

- **GPU Memory**: ≥ 24 GB
- **CUDA**: 12.4 runtime (driver ≥ 525 may also work via backward compatibility)
- **CPU**: x86-64, ≥ 4 cores
- **RAM**: ≥ 8 GB
- **Disk Space**: ≥ 2 GB for the dataset, ≥ 5 GB for backbone models
- **OS**: Linux (tested on Ubuntu 22.04 / 24.04)

## Installation

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate margin
```

### 2. Install PyTorch

(CUDA 12.4 wheels — hosted on pytorch.org, not PyPI)

```bash
pip install torch==2.6.0+cu124          \
            torchvision==0.21.0+cu124   \
            torchaudio==2.6.0+cu124     \
            --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install the HuggingFace CLI

```bash
pip install -U "huggingface_hub"
```

## Download

The dataset and backbone models are hosted on Huggingface. You can download the dataset and backbone models by CLI tools manually.

### Dataset

```bash
hf download codemetic/MARGIN --repo-type=dataset
```

### Backbone Models

```bash
hf download Salesforce/codet5-base
```

> **For users in mainland China:** set the HuggingFace mirror endpoint before any `hf download` command:
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

## Training

Run `python main.py --help` to see all available parameters. The relevant hyperparameters are given in the paper.

Example training command:

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

**NOTE:** Training and evaluation results may differ slightly from the original paper due to variations in platform architecture, hardware, and software versions.

## Evaluation

```bash
python eval.py
```

Interactive CLI — use `↑` / `↓` to select a training run and checkpoint, then configure batch size, seed, and device before evaluation starts.

## Citation

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

## Licence

The source code and programming implementation in this repository are licensed under the **CC BY 4.0 License**.

The datasets referenced or used by this project are **NOT** included under this license. All such datasets remain the intellectual property of their respective owners and are distributed under their own licenses, terms of use, and access conditions. Users are responsible for complying with the applicable licensing requirements of the original dataset providers when downloading, using, or redistributing these datasets.

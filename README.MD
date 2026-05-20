# MARGIN

**M**argin-**A**ware **R**egularized **G**eometry for **I**mbalance Vulnerability Detectio**N**

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

## Requirement

You need at least **24GB** GPU Memory to launch this project under **CUDA 12.4**.

## Getting Start

Ensure you have installed the huggingface-cli on your environment and download the dataset.

```bash
hf download codemetic/MARGIN --repo-type=dataset
```

Start training with such command. Here is the example.

```bash
python main.py                               \
--dataset_subset bigvul                      \
--backbone_name Salesforce/codet5-base       \
--max_epochs 200                             \
--max_checkpoints 3                          \
--early_stop_patience 30                     \
--base_scale 20                              \
--device cuda:0                              \
--batch_size 16 
```

Also, you can lookup all parameter descriptions with

```bash
python main.py --help
```

## Licence

The source code and programming implementation is under the licence of [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://interoperable-europe.ec.europa.eu/licence/creative-commons-attribution-40-international-cc-40)



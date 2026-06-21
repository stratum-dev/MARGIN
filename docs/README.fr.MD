![](../assets/logo.png)

[English](../README.MD) | [中文](./README.zh.MD) | [Français](./README.fr.MD) | [Español](./README.es.MD) | [Русский](./README.ru.MD) | [العربية](./README.ar.MD)

**M**argin-**A**ware **R**egularized **G**eometry for **I**mbalance Vulnerability Detectio**N**

Implémentation officielle de l'article. Ce dépôt fournit les pipelines d'entraînement et d'évaluation pour la reproduction.

## Résumé

![](../assets/banner.png)

La détection de vulnérabilités logicielles est essentielle pour garantir la sécurité et la fiabilité des logiciels. Malgré les progrès récents de l'apprentissage profond, les jeux de données de vulnérabilités du monde réel souffrent de deux défis majeurs : le déséquilibre de fréquence et le déséquilibre de difficulté. Nous réinterprétons ces défis du point de vue de la géométrie de l'espace de représentation, en observant que de tels déséquilibres induisent des distorsions géométriques dans l'espace de représentation hypersphérique. Pour résoudre ce problème, nous proposons MARGIN, un cadre basé sur l'apprentissage métrique qui apprend des représentations discriminatives des vulnérabilités via un apprentissage métrique à marge adaptative et une modélisation de prototypes hypersphériques. MARGIN ajuste dynamiquement la régularisation géométrique en fonction de la structure de distribution estimée par la concentration de von Mises-Fisher, alignant la masse de probabilité des distributions de représentations avec leurs cellules de Voronoi correspondantes, réduisant ainsi la distorsion géométrique et produisant des frontières de décision plus stables. Des expériences approfondies sur des jeux de données publics de vulnérabilités montrent que MARGIN surpasse systématiquement les références solides, réalisant des améliorations notables en classification et en détection, en particulier sur les jeux de données déséquilibrés et difficiles. Une analyse plus poussée démontre que MARGIN produit des géométries de représentations plus structurées, améliorant la robustesse, l'interprétabilité et la généralisation.

## Prérequis

- **Mémoire GPU** : ≥ 24 Go
- **CUDA** : runtime 12.4 (pilote ≥ 525 peut également fonctionner via compatibilité descendante)
- **CPU** : x86-64, ≥ 4 cœurs
- **RAM** : ≥ 8 Go
- **Espace disque** : ≥ 2 Go pour le jeu de données, ≥ 5 Go pour les modèles backbone
- **Système d'exploitation** : Linux (testé sur Ubuntu 22.04 / 24.04)

## Installation

### 1. Créer l'environnement conda

```bash
conda env create -f environment.yml
conda activate margin
```

### 2. Installer PyTorch

(Roues CUDA 12.4 — hébergées sur pytorch.org, pas sur PyPI)

```bash
pip install torch==2.6.0+cu124          \
            torchvision==0.21.0+cu124   \
            torchaudio==2.6.0+cu124     \
            --index-url https://download.pytorch.org/whl/cu124
```

### 3. Installer le CLI HuggingFace

```bash
pip install -U "huggingface_hub"
```

## Téléchargement

Le jeu de données et les modèles backbone sont hébergés sur HuggingFace. Vous pouvez les télécharger manuellement via les outils CLI.

### Jeu de données

```bash
hf download codemetic/MARGIN --repo-type=dataset
```

### Modèles Backbone

```bash
hf download Salesforce/codet5-base
```

> **Pour les utilisateurs en Chine continentale :** configurez le miroir HuggingFace avant toute commande `hf download` :
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

## Entraînement

Exécutez `python main.py --help` pour voir tous les paramètres disponibles. Les hyperparamètres pertinents sont donnés dans l'article.

Exemple de commande d'entraînement :

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

**REMARQUE :** Les résultats d'entraînement et d'évaluation peuvent différer légèrement de l'article original en raison des variations d'architecture de plateforme, de matériel et de versions logicielles.

## Évaluation

```bash
python eval.py
```

CLI interactive — utilisez `↑` / `↓` pour sélectionner une session d'entraînement et un point de contrôle, puis configurez la taille de lot, la graine aléatoire et le périphérique avant le lancement de l'évaluation.

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

Le code source et l'implémentation de programmation dans ce dépôt sont sous licence **CC BY 4.0**.

Les jeux de données référencés ou utilisés par ce projet ne sont **PAS** inclus dans cette licence. Tous ces jeux de données restent la propriété intellectuelle de leurs propriétaires respectifs et sont distribués selon leurs propres licences, conditions d'utilisation et conditions d'accès. Les utilisateurs sont responsables du respect des exigences de licence applicables des fournisseurs de jeux de données originaux lors du téléchargement, de l'utilisation ou de la redistribution de ces jeux de données.

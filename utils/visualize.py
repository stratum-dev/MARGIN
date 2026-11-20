"""
Visualisation utilities for prototype geometry and embedding UMAP plots.

Functions
---------
- ``draw_prototype_dispersion`` / ``_no_num`` — heatmap of pairwise cosine
  similarities between geometric-median prototypes.
- ``draw_prototype_alignment`` / ``_no_num`` — cross-similarity heatmap between
  learnable weight prototypes and geometric-median prototypes.
- ``draw_umap`` — 2-D UMAP projection of learned embeddings, coloured by label.
"""

from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns
import umap


def draw_prototype_dispersion_no_num(
    geometric_median_prototypes: torch.Tensor,
    id2label: dict,
    title: str,
    filepath: str,
):
    """
    Cosine similarity matrix of geometric-median prototypes (no annotations).

    Uses ``imshow`` instead of ``heatmap`` — faster for large numbers of
    classes but does not display numeric values.

    Parameters
    ----------
    geometric_median_prototypes : torch.Tensor
        Prototype matrix of shape ``(C, D)``.
    id2label : dict
        Class-index-to-label mapping.
    title : str
        Plot title.
    filepath : str
        Output file path (SVG recommended).
    """
    geo = geometric_median_prototypes.cpu().numpy()

    # cosine similarity matrix
    sim_matrix = geo @ geo.T
    sim_matrix = sim_matrix * 100

    plt.figure(figsize=(10, 8))

    plt.imshow(
        sim_matrix,
        cmap="viridis",
        vmin=-100,
        vmax=100,
        interpolation="nearest",
        aspect="auto",
    )

    cbar = plt.colorbar()
    cbar.set_label("Cosine Similarity (%)")

    plt.xlabel("Class (sorted by sample count: high → low)")
    plt.ylabel("Class (sorted by sample count: high → low)")

    plt.title(title)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def draw_prototype_dispersion(
    geometric_median_prototypes: torch.Tensor, id2label: dict, title: str, filepath: str
):
    """
    Annotated heatmap of pairwise cosine similarities between prototypes.

    Uses seaborn ``heatmap`` with numeric annotations (upper-triangle masked
    to avoid duplication).  Cosine similarity is scaled to percentage (×100).

    Parameters
    ----------
    geometric_median_prototypes : torch.Tensor
        Prototype matrix of shape ``(C, D)``.
    id2label : dict
        Class-index-to-label mapping.
    title : str
        Plot title.
    filepath : str
        Output file path.
    """
    geo_medians = geometric_median_prototypes.cpu().numpy()
    sim_matrix = np.matmul(geo_medians, geo_medians.T)
    sim_matrix = sim_matrix * 100
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
    n = len(id2label)
    size = max(6, n * 0.5)
    plt.figure(figsize=(size, size), constrained_layout=True)
    sns.heatmap(
        sim_matrix,
        annot=True,
        mask=mask,
        fmt=".0f",
        cmap="viridis",
        vmin=-100,
        vmax=100,
        xticklabels=[id2label[i] for i in range(n)],
        yticklabels=[id2label[i] for i in range(n)],
        cbar_kws={"label": "Similarity (%)"},
        square=True,
    )
    plt.title(title, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def draw_prototype_alignment(
    geometric_median_prototypes: torch.Tensor,
    weight_prototypes: torch.Tensor,
    id2label: dict,
    title: str,
    filepath: str,
):
    """
    Cross-similarity heatmap between geometric-median and learnable weight prototypes.

    Rows = geometric-median prototypes (data-driven), columns = weight vectors
    (classifier).  High diagonal values indicate good alignment between the two
    prototype sets.

    Parameters
    ----------
    geometric_median_prototypes : torch.Tensor
        Shape ``(C, D)``.
    weight_prototypes : torch.Tensor
        Learnable classifier weight vectors, shape ``(C, D)``.
    id2label : dict
        Class-index-to-label mapping.
    title : str
        Plot title.
    filepath : str
        Output file path.
    """
    weight_protos = weight_prototypes.detach()
    sim_matrix = torch.matmul(geometric_median_prototypes.detach(), weight_protos.t())
    sim_matrix = (sim_matrix * 100).cpu().numpy()

    n = len(id2label)
    size = max(6, n * 0.5)
    plt.figure(figsize=(size, size), constrained_layout=True)
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".0f",
        cmap="viridis",
        vmin=-100,
        vmax=100,
        xticklabels=[f"W-{id2label[i]}" for i in range(n)],
        yticklabels=[f"G-{id2label[i]}" for i in range(n)],
        cbar_kws={"label": "Similarity (%)"},
        square=True,
    )

    plt.title(title, pad=20)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def draw_prototype_alignment_no_num(
    geometric_median_prototypes: torch.Tensor,
    weight_prototypes: torch.Tensor,
    id2label: dict,
    title: str,
    filepath: str,
):
    """
    Cross-similarity heatmap (no annotations), same data as
    :func:`draw_prototype_alignment` but using ``imshow``.

    Faster rendering for many classes.
    """
    geo = geometric_median_prototypes.detach()
    weight = weight_prototypes.detach()

    # cosine similarity
    sim_matrix = torch.matmul(geo, weight.t())
    sim_matrix = (sim_matrix * 100).cpu().numpy()

    plt.figure(figsize=(10, 8))

    plt.imshow(
        sim_matrix,
        cmap="coolwarm",
        vmin=-100,
        vmax=100,
        interpolation="nearest",
        aspect="auto",
    )

    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity (%)")

    plt.xlabel("Classifier Prototypes (sorted by sample count: high → low)")
    plt.ylabel("Geometric Median Prototypes (sorted by sample count: high → low)")

    plt.title(title)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def draw_umap(
    features: torch.Tensor,
    pred_label_idx: list,
    id2label: dict,
    title: str,
    filepath: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
):
    """
    2-D UMAP projection of learned embeddings, coloured by predicted label.

    Class 0 ("Non-vul") is always plotted in gray with low alpha; other classes
    use the tab10 colour cycle.

    Parameters
    ----------
    features : torch.Tensor
        Embedding vectors, shape ``(N, D)``.
    pred_label_idx : list of int
        Predicted class index for each sample.
    id2label : dict
        Class-index-to-label mapping.
    title : str
        Plot title.
    filepath : str
        Output file path.
    n_neighbors : int
        UMAP ``n_neighbors`` parameter.
    min_dist : float
        UMAP ``min_dist`` parameter.
    random_state : int
        Seed for reproducible UMAP layout.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    )
    embedding = reducer.fit_transform(features)

    plt.figure(figsize=(6, 5))

    unique_labels = sorted(set(pred_label_idx))

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels) - 1))
    color_map = {}
    idx = 0
    for label in unique_labels:
        if label == 0:
            color_map[label] = "gray"
        else:
            color_map[label] = colors[idx % len(colors)]
            idx += 1

    mask = np.array(pred_label_idx) == 0
    plt.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c="gray",
        label=id2label.get(0, "Non-vul"),
        alpha=0.3,
        s=30,
        edgecolors="none",
    )

    for label in unique_labels:
        if label != 0:
            mask = np.array(pred_label_idx) == label
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color_map[label]],
                label=id2label[label],
                alpha=0.9,
                s=20,
                edgecolors="none",
            )

    plt.legend(loc="best", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

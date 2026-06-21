"""
Evaluation metrics for vulnerability classification models.

This module provides functions to compute:
- Classification metrics (macro, per-class, binary) for multi-class vulnerability detection.
- Clustering quality metrics (NMI, ARI, AMI, FMI, V-measure, angular silhouette).
- ETF (Equiangular Tight Frame) structure metrics for prototype geometry analysis.
- Statistical summary metrics for model parameters (kappa, margin, scale).
"""

import numpy as np
import math
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize


def compute_classification_metrics(truth_label_idx, pred_label_idx, idx2label: dict):
    """
    Compute comprehensive classification metrics for multi-class vulnerability detection.

    Produces three levels of metrics:
    - ``global_macro``: Macro-averaged precision, recall, F1, MCC, accuracy, plus
      per-class breakdown (TP/FP/TN/FN, support, precision, recall, F1, MCC).
    - ``positive_macro``: Macro-averaged metrics over positive classes only
      (class 0 — "non-vulnerable" — is excluded).
    - ``binary``: Binary classification metrics treating class 0 as negative and
      all other classes as positive.

    Parameters
    ----------
    truth_label_idx : array-like of int
        Ground-truth class indices.
    pred_label_idx : array-like of int
        Predicted class indices.
    idx2label : dict[int, str]
        Mapping from class index to human-readable label name.

    Returns
    -------
    dict
        Nested dictionary with keys ``"global_macro"``, ``"positive_macro"``,
        and ``"binary"``.
    """

    all_label_idx = list(range(len(idx2label)))
    metrics = {}

    # ------------------------------------------------------------------
    # Helper: One-vs-All confusion matrix for a single class
    # ------------------------------------------------------------------
    def ova_confusion(y_true, y_pred, class_idx):
        """
        Compute one-vs-all confusion table for a given class.

        Treats ``class_idx`` as positive, all other classes as negative.

        Returns
        -------
        tuple
            (tp, fp, tn, fn, y_true_bin, y_pred_bin)
        """
        y_true_bin = (np.array(y_true) == class_idx).astype(int)
        y_pred_bin = (np.array(y_pred) == class_idx).astype(int)

        tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

        return tp, fp, tn, fn, y_true_bin, y_pred_bin

    # ==================================================================
    # GLOBAL MACRO: macro-averaged metrics across all classes
    # ==================================================================
    metrics["global_macro"] = {
        "mcc": float(matthews_corrcoef(truth_label_idx, pred_label_idx)),
        "f1": float(
            f1_score(truth_label_idx, pred_label_idx, average="macro", zero_division=0)
        ),
        "precision": float(
            precision_score(
                truth_label_idx, pred_label_idx, average="macro", zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                truth_label_idx, pred_label_idx, average="macro", zero_division=0
            )
        ),
        "accuracy": float(accuracy_score(truth_label_idx, pred_label_idx)),
        "per_class": {},
    }

    # Collect per-class FNR/FPR for macro-averaging
    fnr_list, fpr_list = [], []

    for c in all_label_idx:
        # One-vs-all confusion table for class c
        tp, fp, tn, fn, y_true_bin, y_pred_bin = ova_confusion(
            truth_label_idx, pred_label_idx, c
        )

        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fnr_list.append(fnr)
        fpr_list.append(fpr)

        # Binary metrics for this class (OVA view — class c vs rest)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mcc = (
            matthews_corrcoef(y_true_bin, y_pred_bin)
            if (tp + fp + fn + tn) > 0
            else 0.0
        )

        label_name = idx2label[c]
        metrics["global_macro"]["per_class"][label_name] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "support": int(tp + fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": float(mcc),
        }

    metrics["global_macro"]["fnr"] = float(np.mean(fnr_list))
    metrics["global_macro"]["fpr"] = float(np.mean(fpr_list))

    # ==================================================================
    # POSITIVE MACRO: macro-averaged metrics over positive classes only
    # (excludes class 0 — the "non-vulnerable" / background class)
    # ==================================================================
    positive_label_idx = list(range(1, len(idx2label)))

    pos_precisions = []
    pos_recalls = []
    pos_f1s = []
    pos_mccs = []

    for c in positive_label_idx:
        label_name = idx2label[c]
        cls_metrics = metrics["global_macro"]["per_class"][label_name]

        pos_precisions.append(cls_metrics["precision"])
        pos_recalls.append(cls_metrics["recall"])
        pos_f1s.append(cls_metrics["f1"])
        pos_mccs.append(cls_metrics["mcc"])

    metrics["positive_macro"] = {
        "precision": float(np.mean(pos_precisions)),
        "recall": float(np.mean(pos_recalls)),
        "f1": float(np.mean(pos_f1s)),
        "mcc": float(np.mean(pos_mccs)),
    }

    # ==================================================================
    # BINARY: collapse all classes into "non-vulnerable" (class 0) vs
    # "vulnerable" (any positive class)
    # ==================================================================
    y_true_bin = [0 if y == 0 else 1 for y in truth_label_idx]
    y_pred_bin = [0 if y == 0 else 1 for y in pred_label_idx]

    tp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics["binary"] = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "mcc": float(matthews_corrcoef(y_true_bin, y_pred_bin)),
    }

    return metrics


def compute_clustering_metrics(truth_label_idx, pred_label_idx, features=None):
    """
    Compute clustering quality metrics comparing predicted vs ground-truth assignments.

    Computes NMI, ARI, AMI, FMI, V-measure, and the angular (cosine) silhouette score.
    Silhouette score requires at least 2 unique predicted labels; if fewer, it is set
    to NaN.

    Parameters
    ----------
    truth_label_idx : array-like of int
        Ground-truth class indices.
    pred_label_idx : array-like of int
        Predicted cluster/class indices.
    features : np.ndarray or torch.Tensor, optional
        Feature vectors for silhouette computation. Expected shape ``(n_samples, d)``.

    Returns
    -------
    dict
        Keys: ``"nmi"``, ``"ari"``, ``"ami"``, ``"fmi"``, ``"v_measure"``,
        ``"angular_silhouette_score"``.
    """
    truth_label_idx = np.array(truth_label_idx)
    pred_label_idx = np.array(pred_label_idx)

    clustering_metrics = {}

    clustering_metrics["nmi"] = normalized_mutual_info_score(
        truth_label_idx, pred_label_idx
    )
    clustering_metrics["ari"] = adjusted_rand_score(truth_label_idx, pred_label_idx)
    clustering_metrics["ami"] = adjusted_mutual_info_score(
        truth_label_idx, pred_label_idx
    )
    clustering_metrics["fmi"] = fowlkes_mallows_score(truth_label_idx, pred_label_idx)
    clustering_metrics["v_measure"] = v_measure_score(truth_label_idx, pred_label_idx)
    # L2-normalize features for angular (cosine) distance
    features_normalized = normalize(features, norm="l2", axis=1)
    # Silhouette score needs ≥2 clusters; guard against degenerate single-cluster case
    n_unique_labels = len(set(pred_label_idx))
    if n_unique_labels >= 2:
        angular_sh = silhouette_score(
            features_normalized, pred_label_idx, metric="cosine"
        )
    else:
        angular_sh = float("nan")
    clustering_metrics["angular_silhouette_score"] = angular_sh
    return clustering_metrics


def compute_etf_metrics(prototypes: torch.Tensor):
    """
    Measure how closely class prototypes follow an Equiangular Tight Frame (ETF) structure.

    An ETF is a geometric configuration where K unit vectors in R^d are maximally
    separated: all pairwise inner products equal -1/(K-1).  This function computes:

    - Frobenius deviation of the Gram matrix from the ideal ETF Gram.
    - Statistics of off-diagonal cosine similarities (variance, std, deviation from ideal).
    - Angular deviation from the ideal ETF angle (radians).
    - Eigenvalue statistics and condition number of the Gram matrix.

    Parameters
    ----------
    prototypes : torch.Tensor
        Prototype vectors of shape ``(K, d)`` where K is the number of classes.

    Returns
    -------
    dict
        Keys: ``"etf_error"``, ``"etf_error_norm"``, ``"cosine_variance"``,
        ``"cosine_std"``, ``"avg_cosine_deviation"``, ``"max_cosine_deviation"``,
        ``"avg_angle_deviation"``, ``"max_angle_deviation"``, ``"eig_var"``,
        ``"eig_mean"``, ``"cond_num"``.
    """
    P = F.normalize(prototypes, dim=1)
    K, d = P.shape

    # Gram matrix: pairwise inner products between normalized prototypes
    G = P @ P.T

    # 1. Frobenius error from the ideal ETF Gram matrix
    #    Ideal: diag=1, off-diag=-1/(K-1)
    target = torch.full((K, K), -1 / (K - 1), device=P.device)
    target.fill_diagonal_(1)
    etf_error = torch.norm(G - target, p="fro").item()
    etf_error_norm = etf_error / K  # Per-class normalized error

    # 2. Extract off-diagonal cosine similarities
    mask = ~torch.eye(K, dtype=bool, device=P.device)
    cosines = G[mask]

    # 3. Dispersion of off-diagonal cosines
    cosine_variance = cosines.var().item()
    cosine_std = cosines.std().item()

    # 4. Absolute deviation of cosines from the ideal ETF value
    avg_cosine_deviation = torch.mean(torch.abs(cosines - (-1 / (K - 1)))).item()
    max_cosine_deviation = torch.max(torch.abs(cosines - (-1 / (K - 1)))).item()

    # 5. Angular deviation from the ideal ETF angle (radians)
    angles = torch.acos(cosines.clamp(-1, 1))
    etf_angle = math.acos(-1 / (K - 1))
    avg_angle_deviation = torch.mean(torch.abs(angles - etf_angle)).item()
    max_angle_deviation = torch.max(torch.abs(angles - etf_angle)).item()

    # 6. Eigenvalue statistics of the Gram matrix (non-zero only)
    eigvals = torch.linalg.eigvalsh(G)
    non_zero_eig = eigvals[eigvals > 1e-6]
    eig_var = non_zero_eig.var().item()
    eig_mean = non_zero_eig.mean().item()

    # 7. Gram matrix condition number (smallest → 0 means collapse)
    cond_num = torch.linalg.cond(G).item()

    return {
        "etf_error": etf_error,
        "etf_error_norm": etf_error_norm,
        "cosine_variance": cosine_variance,
        "cosine_std": cosine_std,
        "avg_cosine_deviation": avg_cosine_deviation,
        "max_cosine_deviation": max_cosine_deviation,
        "avg_angle_deviation": avg_angle_deviation,
        "max_angle_deviation": max_angle_deviation,
        "eig_var": eig_var,
        "eig_mean": eig_mean,
        "cond_num": cond_num,
    }


def compute_statistics_metrics(kappas, margins, scales, id2label):
    """
    Summarise per-class learnable parameters (kappa, margin, scale) of the loss head.

    Parameters
    ----------
    kappas : torch.Tensor
        Per-class concentration (kappa) parameters, shape ``(C,)``.
    margins : torch.Tensor
        Per-class angular margin parameters, shape ``(C,)``.
    scales : torch.Tensor
        Per-class scale parameters, shape ``(C,)``.
    id2label : dict[int, str]
        Mapping from class index to human-readable label name.

    Returns
    -------
    dict
        Nested dictionary with keys ``"per_class"`` (label → kappa/margin/scale dict)
        and ``"summary"`` (mean and std of each parameter across classes).
    """
    # Move tensors to CPU for summary computation
    kappas = kappas.detach().cpu()
    margins = margins.detach().cpu()
    scales = scales.detach().cpu()
    C = kappas.shape[0]

    per_class = {}
    for i in range(C):
        label = id2label[i]
        per_class[label] = {
            "kappa": float(kappas[i]),
            "margin": float(margins[i]),
            "scale": float(scales[i]),
        }

    summary = {
        "kappa_mean": float(kappas.mean()),
        "kappa_std": float(kappas.std()),
        "margin_mean": float(margins.mean()),
        "margin_std": float(margins.std()),
        "scale_mean": float(scales.mean()),
        "scale_std": float(scales.std()),
    }

    return {"per_class": per_class, "summary": summary}

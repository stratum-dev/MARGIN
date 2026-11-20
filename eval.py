#!/usr/bin/env python3
"""
MARGIN Evaluation Script
Interactive CLI to select a training run and checkpoint, then evaluate on the test set.
Outputs metrics JSON, UMAP, prototype-dispersion, and prototype-alignment charts.
"""

import json
import os

import numpy as np
import torch
from datasets import load_dataset
from InquirerPy import inquirer
from torch.utils.data import DataLoader

from utils.dataset import CodeDataset
from utils.evaluation import evaluate_model
from utils.logger import log
from utils.model import MARGINModel
from utils.seed import set_seed
from utils.string import print_dict_pipe
from utils.visualize import (
    draw_prototype_alignment,
    draw_prototype_dispersion,
    draw_umap,
)

OUTPUT_ROOT = "./output"


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def get_immediate_subdirs(root):
    """Return absolute paths of immediate subdirectories under *root*, sorted."""
    if not os.path.isdir(root):
        return []
    entries = sorted(os.listdir(root))
    subdirs = []
    for entry in entries:
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            subdirs.append(full)
    return subdirs


def get_checkpoint_files(run_dir):
    """Return sorted list of .pth checkpoint paths under *run_dir*/checkpoints/."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    names = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]

    # sort by epoch number embedded in filename
    def _epoch_key(name):
        try:
            return int(name.replace("epoch_", "").replace(".pth", ""))
        except ValueError:
            return 0

    names.sort(key=_epoch_key)
    return [os.path.join(ckpt_dir, n) for n in names]


def select_from_list(items, prompt="Select an option"):
    """Arrow-key interactive menu powered by InquirerPy.  Returns the chosen item or None."""
    if not items:
        print("No items found.")
        return None

    choices = [
        {"name": os.path.basename(it) if os.path.isabs(it) else str(it), "value": it}
        for it in items
    ]
    try:
        result = inquirer.select(
            message=prompt + ":",
            choices=choices,
            vi_mode=True,
            raise_keyboard_interrupt=True,
        ).execute()
        return result
    except KeyboardInterrupt:
        print("\nAborted.")
        exit(0)


def _prompt_value(label, default="", cast=str, validator=None):
    """Prompt for a single value, returning the parsed result (or default on empty)."""
    prompt = f"{label} (default: {default}): "
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return cast(default) if cast is not str else default
        if raw.lower() == "q":
            print("Aborted.")
            exit(0)
        try:
            val = cast(raw)
        except ValueError:
            print(f"  Invalid value for {label}. Please try again.")
            continue
        if validator is not None:
            checked = validator(val)
            if checked is None:
                print(f"  Invalid value for {label}. Please try again.")
                continue
            val = checked
        return val


# ---------------------------------------------------------------------------
#  checkpoint & model helpers
# ---------------------------------------------------------------------------


def load_checkpoint(path):
    """Load a .pth checkpoint from disk (safe for CPU-only machines)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model_from_checkpoint(ckpt):
    """Reconstruct model + test dataset from checkpoint metadata."""
    cfg = ckpt["config"]
    backbone_name = cfg.get("backbone_name", "microsoft/unixcoder-base")
    dataset_name = cfg.get("dataset_name", "codemetic/MARGIN")
    dataset_subset = cfg.get("dataset_subset", "bigvul")
    base_scale = cfg.get("base_scale", 20)
    alpha = cfg.get("alpha", 0.95)

    # Load HF dataset – only the test split is needed
    hf_dataset = load_dataset(dataset_name, dataset_subset)
    test_hf = hf_dataset["test"]

    # Build CodeDataset (pass test set for all splits to satisfy constructor)
    test_dataset = CodeDataset(backbone_name, test_hf)

    model = MARGINModel(
        backbone=backbone_name,
        base_scale=base_scale,
        alpha=alpha,
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        test_dataset=test_dataset,
    )

    return model, test_dataset


def restore_eval_state(model, ckpt):
    """Restore evaluation-only tensors & mappings that live outside state_dict."""
    num_classes = model.num_classes
    embed_dim = model.embedding_dim
    device = model.weights.device

    model.current_geometric_median_prototypes = ckpt.get(
        "current_geometric_median_prototypes",
        torch.zeros(num_classes, embed_dim),
    ).to(device)

    model.loss_head.kappas = ckpt.get(
        "loss_head_kappas",
        torch.zeros(num_classes),
    ).to(device)

    model.loss_head.margins = ckpt.get(
        "loss_head_margins",
        torch.zeros(num_classes),
    ).to(device)

    model.loss_head.scales = ckpt.get(
        "loss_head_scales",
        torch.full((num_classes,), 20.0),
    ).to(device)

    if "label2id" in ckpt:
        model.label2id = ckpt["label2id"]
    if "id2label" in ckpt:
        # ensure integer keys (torch.save preserves them)
        model.id2label = {int(k): v for k, v in ckpt["id2label"].items()}


# ---------------------------------------------------------------------------
#  output helpers
# ---------------------------------------------------------------------------


def _make_serializable(obj):
    """Recursively convert tensors/arrays to plain Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.item() if obj.size == 1 else obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_metrics_json(metrics, epoch, report_dir):
    """Write evaluation metrics dict to a JSON file."""
    path = os.path.join(report_dir, f"test_metrics_epoch_{epoch}.json")
    os.makedirs(report_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_make_serializable(metrics), f, indent=2)
    log.print(f"Metrics saved to: {path}")
    return path


def print_eval_results(epoch, metrics):
    """Pretty-print evaluation metrics to the logger."""
    cm = metrics["classification_metrics"]
    clm = metrics["clustering_metrics"]
    em = metrics["etf_metrics"]
    sm = metrics["statistics_metrics"]

    log.print(f"\n{'='*50}")
    log.print(f"Epoch {epoch}  Test Set Evaluation Results")
    log.print(f"{'='*50}")

    log.print("Classification Metrics: ---------------------------------")
    log.print(
        f"🐱 Binary - MCC: {cm['binary']['mcc']:.4f}, "
        f"F1: {cm['binary']['f1']:.4f}, "
        f"Prec: {cm['binary']['precision']:.4f}, "
        f"Rec: {cm['binary']['recall']:.4f}"
    )
    log.print(
        f"🐒 Positive-Macro - MCC: {cm['positive_macro']['mcc']:.4f}, "
        f"F1: {cm['positive_macro']['f1']:.4f}, "
        f"Prec: {cm['positive_macro']['precision']:.4f}, "
        f"Rec: {cm['positive_macro']['recall']:.4f}"
    )
    log.print(
        f"🌏 Global-Macro - MCC: {cm['global_macro']['mcc']:.4f}, "
        f"F1: {cm['global_macro']['f1']:.4f}, "
        f"FNR: {cm['global_macro']['fnr']:.4f}, "
        f"FPR: {cm['global_macro']['fpr']:.4f}"
    )

    log.print("Clustering Metrics: ---------------------------------")
    log.print(print_dict_pipe(clm))

    log.print("ETF Metrics: ---------------------------------")
    log.print(print_dict_pipe(em))

    log.print("Statistics Metrics: ---------------------------------")
    log.print(
        f"Margin - Mean: {sm['summary']['margin_mean']:.4f}, "
        f"Std: {sm['summary']['margin_std']:.4f}"
    )
    log.print(
        f"Scale  - Mean: {sm['summary']['scale_mean']:.4f}, "
        f"Std: {sm['summary']['scale_std']:.4f}"
    )
    log.print(
        f"Kappa  - Mean: {sm['summary']['kappa_mean']:.4f}, "
        f"Std: {sm['summary']['kappa_std']:.4f}"
    )


def visualize_and_save(
    model,
    features,
    truth_labels,
    epoch,
    eval_dir,
    umap_n_neighbors,
    umap_min_dist,
    seed,
):
    """Generate UMAP, prototype-dispersion, and prototype-alignment SVGs."""
    umap_dir = os.path.join(eval_dir, "umap")
    proto_align_output_dir = os.path.join(eval_dir, "prototype-alignment")
    proto_disp_output_dir = os.path.join(eval_dir, "prototype-dispersion")

    for d in (umap_dir, proto_align_output_dir, proto_disp_output_dir):
        os.makedirs(d, exist_ok=True)

    draw_prototype_dispersion(
        model.current_geometric_median_prototypes,
        model.id2label,
        f"Epoch {epoch} (Test)",
        os.path.join(proto_disp_output_dir, f"geo_median_sim_epoch_{epoch}_test.svg"),
    )

    draw_prototype_alignment(
        model.current_geometric_median_prototypes,
        model.get_norm_weight_prototypes(),
        model.id2label,
        f"Epoch {epoch} (Test)",
        os.path.join(proto_align_output_dir, f"weight_geo_sim_epoch_{epoch}_test.svg"),
    )

    draw_umap(
        features,
        truth_labels,
        model.id2label,
        f"UMAP Visualization - Epoch {epoch} (Test Set)",
        os.path.join(umap_dir, f"umap_epoch_{epoch}_test.svg"),
        umap_n_neighbors,
        umap_min_dist,
        seed,
    )


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------


def main():
    """
    Interactive evaluation entry point.

    Steps:
    1. Select a training run from ``./output/``.
    2. Choose a checkpoint (``.pth`` file).
    3. Prompt for device, batch size, and random seed.
    4. Reconstruct the model, load weights, and run test-set evaluation.
    5. Save metrics JSON and visualisation SVGs.
    """
    # ---- Step 1: select training run ----
    runs = get_immediate_subdirs(OUTPUT_ROOT)
    if not runs:
        log.print(f"No output directories found under '{OUTPUT_ROOT}'.")
        log.print("Please run training first (main.py).")
        return

    selected_run = select_from_list(runs, "Select a training run")
    if selected_run is None:
        log.print("Aborted.")
        return

    run_name = os.path.basename(selected_run)
    log.print(f"Selected run: {run_name}")

    # ---- Step 2: select checkpoint ----
    ckpt_paths = get_checkpoint_files(selected_run)
    if not ckpt_paths:
        log.print(f"No .pth checkpoints found in '{selected_run}/checkpoints/'.")
        return

    ckpt_display_names = [os.path.basename(p) for p in ckpt_paths]
    chosen_display = select_from_list(ckpt_display_names, "Select a checkpoint")
    if chosen_display is None:
        log.print("Aborted.")
        return

    ckpt_path = os.path.join(selected_run, "checkpoints", chosen_display)
    epoch_num = int(chosen_display.replace("epoch_", "").replace(".pth", ""))
    log.print(f"Selected checkpoint: {chosen_display}  (epoch {epoch_num})")

    # ---- Step 3: load checkpoint ----
    log.print("Loading checkpoint...")
    ckpt = load_checkpoint(ckpt_path)

    # ---- Step 4: user-specified runtime parameters ----
    print()
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = _prompt_value(
        "Device",
        default=default_device,
        validator=lambda v: (
            v
            if v in ("cpu", "cuda", "cuda:0", "cuda:1") or v.startswith("cuda:")
            else None
        ),
    )
    batch_size = _prompt_value("Batch size", default="16", cast=int)
    seed = _prompt_value("Random seed", default="42", cast=int)
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1

    set_seed(seed)

    # ---- Step 5: build model & load weights ----
    log.print("Building model and loading test dataset...")
    model, test_dataset = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    restore_eval_state(model, ckpt)

    # ---- Step 6: run evaluation ----
    log.print(
        f"Running test evaluation on device: {device}  batch_size: {batch_size}  seed: {seed}"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    (
        metrics,
        all_features,
        all_truth_label_idx,
        all_pred_label_idx,
        all_raw_labels,
        avg_loss,
    ) = evaluate_model(model, test_loader, f"Checkpoint {epoch_num} Evaluation", device)

    # ---- Step 7: save outputs ----
    eval_dir = os.path.join(selected_run, "evaluation", f"epoch_{epoch_num}")
    report_dir = os.path.join(eval_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    save_metrics_json(metrics, epoch_num, report_dir)
    print_eval_results(epoch_num, metrics)

    log.print("Generating visualizations...")
    visualize_and_save(
        model,
        all_features,
        all_truth_label_idx,
        epoch_num,
        eval_dir,
        UMAP_N_NEIGHBORS,
        UMAP_MIN_DIST,
        seed,
    )

    log.print(f"\n✅ Test evaluation complete. Outputs saved to: {eval_dir}")


if __name__ == "__main__":
    main()

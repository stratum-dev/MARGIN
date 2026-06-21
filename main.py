import argparse
from datetime import datetime
import json
import os
from datasets import load_dataset
from utils.trainer import Trainer, TrainerConfig
from utils.seed import set_seed
from utils.dataset import CodeDataset
from utils.model import MARGINModel
from utils.logger import log


def parse_args():
    parser = argparse.ArgumentParser(description="MARGIN Model Training Script")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="codemetic/MARGIN",
        help="HuggingFace Dataset (Default: codemetic/MARGIN)",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="bigvul",
        help="Subset (Default：bigvul)",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="microsoft/unixcoder-base",
        help="Pretrained Backbone (Default: microsoft/unixcoder-base)",
    )

    # --- Hyperparameter Configuration ---
    parser.add_argument(
        "--base_scale", type=int, default=20, help="Base scale (Default: 20)"
    )
    parser.add_argument(
        "--confidence_alpha",
        type=float,
        default=0.95,
        help="Confidence Alpha (Default: 0.95)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning Rate (Default: 2e-5)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight Decay (Default: 0.01)"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=200, help="Max Epochs (Default: 200)"
    )
    parser.add_argument(
        "--max_checkpoints", type=int, default=3, help="Max Checkpoints (Default: 3)"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=30,
        help="Early Stop Patience (Default: 30)",
    )

    # --- Runtime Environment Configuration ---
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (Default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Your cuda device,e.g., cuda:0 (default: cuda:0)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size (default: 16)"
    )

    # UMAP related parameters
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)

    # Parse arguments
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    TIME_PREFIX = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    OUTPUT_DIR = f"./output/{args.dataset_subset}-{args.backbone_name.split('/')[1]}-{TIME_PREFIX}"
    SETTINGS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "settings.json")
    log.set_log_file(os.path.join(OUTPUT_DIR, "train.log"))

    config = TrainerConfig(
        output_dir=OUTPUT_DIR,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        max_checkpoints=args.max_checkpoints,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        seed=args.seed,
        backbone_name=args.backbone_name,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        base_scale=args.base_scale,
        alpha=args.confidence_alpha,
    )

    # Persist all hyperparameters for reproducibility
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    settings = config.__dict__.copy()
    settings["time_prefix"] = TIME_PREFIX
    with open(SETTINGS_OUTPUT_PATH, "w") as f:
        json.dump(settings, f, indent=2, default=str)
    log.print(f"Settings saved to {SETTINGS_OUTPUT_PATH}")

    set_seed(config.seed)
    log.print(
        f"Training on dataset: {args.dataset_name}, subset: {args.dataset_subset}"
    )
    log.print("Loading dataset...")

    dataset = load_dataset(args.dataset_name, args.dataset_subset)

    train_hf = dataset["train"]
    val_hf = dataset["val"]
    test_hf = dataset["test"]

    log.print(
        f"Train size: {len(train_hf)}, Val size: {len(val_hf)}, Test size: {len(test_hf)}"
    )

    log.print(f"Loading tokenizer and model: {args.backbone_name}")

    train_dataset = CodeDataset(args.backbone_name, train_hf)
    val_dataset = CodeDataset(args.backbone_name, val_hf)
    test_dataset = CodeDataset(args.backbone_name, test_hf)

    label2id = train_dataset.label2idx

    log.print(f"Number of classes: {len(label2id)}")
    log.print(f"Label mapping: {label2id}")

    model = MARGINModel(
        backbone=args.backbone_name,
        base_scale=args.base_scale,
        alpha=args.confidence_alpha,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    trainer = Trainer(model, config)
    trainer.train()


if __name__ == "__main__":
    main()

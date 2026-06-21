from collections import deque
import json
import os
from os import path
import warnings
from datetime import datetime
import time
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.model import MARGINModel
from utils.math import (
    compute_geometric_median,
    compute_vmf_kappa,
)
from utils.string import print_dict_pipe
from utils.visualize import (
    draw_prototype_dispersion,
    draw_prototype_alignment,
    draw_umap,
)
from utils.evaluation import evaluate_model
from utils.logger import log

warnings.filterwarnings("ignore", category=UserWarning)


class TrainerConfig:

    def __init__(
        self,
        batch_size: int,
        learning_rate: int,
        weight_decay: int,
        max_epochs: int,
        max_checkpoints: int,
        early_stopping_patience: int,
        output_dir: str,
        device: str,
        umap_n_neighbors: int,
        umap_min_dist: float,
        seed=int,
        backbone_name: str = "microsoft/unixcoder-base",
        dataset_name: str = "codemetic/MARGIN",
        dataset_subset: str = "bigvul",
        base_scale: float = 20,
        alpha: float = 0.95,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.max_checkpoints = max_checkpoints
        self.output_dir = output_dir
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.seed = seed
        self.backbone_name = backbone_name
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.base_scale = base_scale
        self.alpha = alpha


class Trainer:
    def __init__(self, model: MARGINModel, config: TrainerConfig):
        self.config = config
        self.model = model.to(config.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = GradScaler()

        self.best_global_f1 = float("-inf")
        self.patience_counter = 0
        self.best_model_state = None

        self.time_prefix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def setup_output_dirs(self):
        self.umap_output_dir = os.path.join(self.config.output_dir, "umap")
        self.prototype_alignment_output_dir = os.path.join(
            self.config.output_dir, "prototype-alignment"
        )
        self.prototype_dispersion_output_dir = os.path.join(
            self.config.output_dir, "prototype-dispersion"
        )
        self.report_output_dir = os.path.join(self.config.output_dir, "report")

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.umap_output_dir, exist_ok=True)
        os.makedirs(self.prototype_alignment_output_dir, exist_ok=True)
        os.makedirs(self.prototype_dispersion_output_dir, exist_ok=True)
        os.makedirs(self.report_output_dir, exist_ok=True)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        feature_accumulator = {label: [] for label in range(self.model.num_classes)}

        start_time = time.time()
        log.print(
            f"⏱️ Epoch {epoch} Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
        )

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            label_idx = batch["label_idx"].to(self.config.device)

            self.optimizer.zero_grad()
            with autocast(self.config.device):
                cos_theta, features = self.model(
                    input_ids, attention_mask, return_features=True
                )
                loss = self.model.loss_head(cos_theta, label_idx)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            features_cpu = features.detach().cpu()
            labels_cpu = label_idx.detach().cpu()
            for f, l in zip(features_cpu, labels_cpu):
                feature_accumulator[int(l)].append(f)

        D = features.shape[1]
        C = self.model.num_classes

        mean_prototypes_list = []
        geom_median_prototypes_list = []
        kappas_list = []
        class_counts_list = []

        with torch.no_grad():
            for label_idx in range(C):
                feats = feature_accumulator[label_idx]
                feats_tensor = torch.stack(feats, dim=0)  # [N, D]
                class_counts_list.append(len(feats))
                feats_tensor_norm = F.normalize(feats_tensor, p=2, dim=1)
                mean_proto = F.normalize(feats_tensor_norm.mean(dim=0), dim=0)
                mean_prototypes_list.append(mean_proto)

                # Geometric Median Prototype
                geom_median_proto = compute_geometric_median(feats_tensor_norm)
                geom_median_proto = F.normalize(geom_median_proto, dim=0)
                geom_median_prototypes_list.append(geom_median_proto)

                kappa = compute_vmf_kappa(feats_tensor_norm, mean_proto)
                kappas_list.append(kappa)

            self.model.current_mean_prototypes = torch.stack(
                mean_prototypes_list, dim=0
            ).to(self.config.device)

            self.model.current_geometric_median_prototypes = torch.stack(
                geom_median_prototypes_list, dim=0
            ).to(self.config.device)

            self.model.class_counts = torch.tensor(class_counts_list).to(
                self.config.device
            )  # [C]
            self.model.current_kappas = torch.tensor(kappas_list).to(
                self.config.device
            )  # [C]

            self.model.loss_head.update_adaptive_params(
                self.model.current_kappas,
                self.model.class_counts,
                self.model.current_mean_prototypes,
            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        log.print(
            f"✅ Epoch {epoch} Training finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )
        log.print(f"⏳ Epoch {epoch} Training costs {elapsed_time:.2f} seconds.")

        return total_loss / num_batches

    def evaluate_epoch(self, dataloader, epoch, save_prefix="val"):
        self.model.eval()

        start_time = time.time()
        log.print(
            f"⏱️  Epoch {epoch} Evaluation started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
        )

        (
            metrics,
            all_features,
            all_truth_label_idx,
            all_pred_label_idx,
            all_raw_labels,
            avg_loss,
        ) = evaluate_model(
            self.model, dataloader, f"Epoch {epoch} Evaluating", self.config.device
        )

        classification_metrics = metrics["classification_metrics"]
        clustering_metrics = metrics["clustering_metrics"]
        etf_metrics = metrics["etf_metrics"]
        statistics_metrics = metrics["statistics_metrics"]

        json_path = os.path.join(
            self.report_output_dir, f"{save_prefix}_metrics_epoch_{epoch}.json"
        )
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        log.print(f"Epoch {epoch} Evaluation Results:")
        log.print("Classification Metrics: ---------------------------------")
        log.print(
            f"🐱 Binary - MCC: {classification_metrics['binary']['mcc']:.4f}, F1: {classification_metrics['binary']['f1']:.4f}, "
            f"Prec: {classification_metrics['binary']['precision']:.4f}, Rec: {classification_metrics['binary']['recall']:.4f}"
        )
        log.print(
            f"🐒 Positive-Macro - MCC: {classification_metrics['positive_macro']['mcc']:.4f}, "
            f"F1: {classification_metrics['positive_macro']['f1']:.4f}, "
            f"Prec: {classification_metrics['positive_macro']['precision']:.4f}, "
            f"Rec: {classification_metrics['positive_macro']['recall']:.4f}, "
        )
        log.print(
            f"🌏 Global-Macro - MCC: {classification_metrics['global_macro']['mcc']:.4f}, "
            f"F1: {classification_metrics['global_macro']['f1']:.4f}, "
            f"FNR: {classification_metrics['global_macro']['fnr']:.4f}, "
            f"FPR: {classification_metrics['global_macro']['fpr']:.4f}"
        )

        log.print("Clustering Metrics: ---------------------------------")
        log.print(print_dict_pipe(clustering_metrics))

        log.print("ETF Metrics: ---------------------------------")
        log.print(print_dict_pipe(etf_metrics))

        log.print("Statistics Metrics: ---------------------------------")
        log.print(
            f"Margin - Mean: {statistics_metrics['summary']['margin_mean']:.4f}, Std: {statistics_metrics['summary']['margin_std']:.4f}"
        )
        log.print(
            f"Scale - Mean: {statistics_metrics['summary']['scale_mean']:.4f}, Std: {statistics_metrics['summary']['scale_std']:.4f}"
        )
        log.print(
            f"Kappa - Mean: {statistics_metrics['summary']['kappa_mean']:.4f}, Std: {statistics_metrics['summary']['kappa_std']:.4f}"
        )

        self.visualize_epoch(all_features, all_truth_label_idx, epoch)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log.print(
            f"✅ Epoch {epoch} Evaluation finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )
        log.print(f"⏳ Epoch {epoch} Evaluation costs: {elapsed_time:.2f} seconds")

        return avg_loss, metrics

    def visualize_epoch(self, features, truth_label_idx, epoch):
        draw_prototype_dispersion(
            self.model.current_geometric_median_prototypes,
            self.model.id2label,
            f"Epoch {epoch}",
            os.path.join(
                self.prototype_dispersion_output_dir,
                f"geo_median_sim_epoch_{epoch}.svg",
            ),
        )

        draw_prototype_alignment(
            self.model.current_geometric_median_prototypes,
            self.model.get_norm_weight_prototypes(),
            self.model.id2label,
            f"Epoch {epoch}",
            os.path.join(
                self.prototype_alignment_output_dir, f"weight_geo_sim_epoch_{epoch}.svg"
            ),
        )

        draw_umap(
            features,
            truth_label_idx,
            self.model.id2label,
            f"UMAP Visualization - Epoch {epoch}",
            os.path.join(self.umap_output_dir, f"umap_epoch_{epoch}.svg"),
            self.config.umap_n_neighbors,
            self.config.umap_min_dist,
            self.config.seed,
        )

    def train(self):
        self.setup_output_dirs()
        log.set_log_file(os.path.join(self.config.output_dir, "train.log"))

        self.init_checkpoint_queue()

        for epoch in range(0, self.config.max_epochs + 1):
            log.print(f"\n{'='*50}")
            log.print(f"Epoch {epoch}/{self.config.max_epochs}")
            log.print(f"{'='*50}")

            g = torch.Generator()
            g.manual_seed(self.config.seed)
            train_loader = DataLoader(
                self.model.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=g,
            )
            val_loader = DataLoader(
                self.model.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

            train_loss = self.train_epoch(train_loader, epoch)
            log.print(f"Train Loss: {train_loss:.4f}")

            avg_val_loss, val_metrics = self.evaluate_epoch(val_loader, epoch)
            val_global_f1 = val_metrics["classification_metrics"]["global_macro"]["f1"]
            log.print(f"Val Loss: {avg_val_loss:.4f}")

            # Check if best model needs to be updated
            self.update_best_model(epoch, val_global_f1)

            # Print current best
            if self.best_model_state is not None:
                best_epoch = self.best_model_state["epoch"]
                best_global_f1 = self.best_model_state["val_global_f1"]
                log.print(
                    f"🏆 Current Best: Epoch {best_epoch} | Global F1 {best_global_f1:.4f}"
                )
            else:
                log.print(
                    f"🏆 Current Best: Epoch {epoch} | Global F1 {val_global_f1:.4f}"
                )

            # Early stopping check
            if self.patience_counter >= self.config.early_stopping_patience:
                log.print(f"Early stopping triggered at epoch {epoch}")
                break

        if self.best_model_state is not None:
            log.print(f"Loading best model from epoch {self.best_model_state['epoch']}")
            self.model.load_state_dict(self.best_model_state["model_state_dict"])

        return self.model

    def init_checkpoint_queue(self):
        self.checkpoint_queue = deque()

    def update_best_model(self, epoch, val_global_f1):
        if val_global_f1 > self.best_global_f1:
            self.best_global_f1 = val_global_f1
            self.patience_counter = 0
            self.best_model_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_global_f1": val_global_f1,
            }
            log.print("Model improved.")

            self.save_checkpoint_with_queue(epoch)
        else:
            self.patience_counter += 1
            log.print(
                f"No improvement. Patience: {self.patience_counter}/{self.config.early_stopping_patience}"
            )

    def save_checkpoint_with_queue(self, epoch):
        checkpoint_path = os.path.join(
            self.config.output_dir, "checkpoints", f"epoch_{epoch}.pth"
        )
        self.save_checkpoint_file(checkpoint_path)
        self.checkpoint_queue.append(checkpoint_path)

        while len(self.checkpoint_queue) > self.config.max_checkpoints:
            oldest = self.checkpoint_queue.popleft()
            if os.path.exists(oldest):
                os.remove(oldest)
                log.print(f"Removed oldest checkpoint: {oldest}")

    def save_checkpoint_file(self, path):

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_global_f1": self.best_global_f1,
            "patience_counter": self.patience_counter,
            "best_model_state": self.best_model_state,
            "config": self.config.__dict__,
            "label2id": self.model.label2id,
            "id2label": self.model.id2label,
        }
        torch.save(checkpoint, path)
        log.print(f"Checkpoint saved: {path}")

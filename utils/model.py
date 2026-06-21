"""
MARGIN model architecture.

Contains:
- :class:`MARGINModel`: backbone encoder + learnable class prototypes (weights)
  with adaptive ETF-guided loss.
- :class:`MARGINLossHead`: ArcFace-style loss with dynamic per-class margins and
  scales driven by vMF concentration estimates.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForTextEncoding
from utils.dataset import CodeDataset
from utils.math import (
    compute_margin,
    compute_scale,
)
from utils.logger import log


class MARGINModel(nn.Module):
    """
    MARGIN: Multi-class vulnerability detection with adaptive margins.

    Architecture
    ------------
    1. A pretrained transformer backbone (e.g. UniXcoder) encodes source code
       into hidden states.
    2. The last layer's ``[CLS]`` hidden state is extracted as the code
       embedding.
    3. Cosine similarity between the normalised embedding and learnable class
       weight vectors (prototypes) produces logits.
    4. A :class:`MARGINLossHead` applies adaptive angular margins and per-class
       scaling for the final cross-entropy loss.

    Parameters
    ----------
    backbone : str
        HuggingFace model name (e.g. ``"microsoft/unixcoder-base"``).
    base_scale : float
        Base scale factor for the loss head.
    alpha : float
        Confidence level for vMF margin computation.
    train_dataset : CodeDataset
        Training dataset (used to infer number of classes and label mappings).
    val_dataset : CodeDataset
        Validation dataset.
    test_dataset : CodeDataset
        Test dataset.
    dropout_rate : float
        Dropout probability applied after feature extraction (default 0.0).
    """

    def __init__(
        self,
        backbone: str,
        base_scale: float,
        alpha: float,
        train_dataset: CodeDataset,
        val_dataset: CodeDataset,
        test_dataset: CodeDataset,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.roberta_config = AutoConfig.from_pretrained(
            backbone, output_hidden_states=True
        )
        self.roberta = AutoModelForTextEncoding.from_pretrained(
            backbone, config=self.roberta_config
        )

        self.embedding_dim = self.roberta_config.hidden_size
        self.num_classes = len(train_dataset.label2idx)
        self.dropout = nn.Dropout(dropout_rate)
        self.weights = nn.Parameter(
            F.normalize(torch.Tensor(self.num_classes, self.embedding_dim), p=2, dim=1)
        )
        self.layer_weights = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.weights)

        self.current_kappas = torch.zeros(self.num_classes)
        self.current_mean_prototypes = torch.zeros(self.num_classes, self.embedding_dim)
        self.current_geometric_median_prototypes = torch.zeros(
            self.num_classes, self.embedding_dim
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.label2id = train_dataset.label2idx
        self.id2label = train_dataset.idx2label

        self.loss_head: MARGINLossHead = MARGINLossHead(
            self.num_classes, base_scale, alpha, self.embedding_dim
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input_ids, attention_mask, return_features=False):
        """
        Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token indices, shape ``(B, L)``.
        attention_mask : torch.Tensor
            Attention mask, shape ``(B, L)``.
        return_features : bool
            If True, also return the ``[CLS]`` embeddings.

        Returns
        -------
        torch.Tensor or tuple
            Cosine similarity logits ``(B, C)``, and optionally the ``[CLS]``
            feature vectors ``(B, D)``.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # Select the last hidden layer
        selected_layers = []
        layers_to_concat = [-1]
        for layer_idx in layers_to_concat:
            selected_layers.append(hidden_states[layer_idx])  # (B, L, D)

        # Weighted layer fusion (currently a single layer, extensible to more)
        stacked = torch.stack(selected_layers, dim=0)  # (N_layers, B, L, D)
        norm_weights = self.softmax(self.layer_weights)  # (N_layers,)
        weighted = norm_weights.view(-1, 1, 1, 1) * stacked
        fused = torch.sum(weighted, dim=0)  # (B, L, D)

        # Take [CLS] token representation as the sentence embedding
        features = fused[:, 0, :]  # (B, D)

        # Cosine similarity to learnable class weight vectors (prototypes)
        cos_thetas = torch.matmul(
            F.normalize(features, p=2, dim=1),
            F.normalize(self.weights, p=2, dim=1).t(),
        )  # [B, C]

        if return_features:
            return cos_thetas, features
        return cos_thetas

    def get_norm_weight_prototypes(self):
        """
        Return L2-normalised learnable weight vectors (classifier prototypes).

        Returns
        -------
        torch.Tensor
            Shape ``(C, D)``.
        """
        return F.normalize(self.weights.detach(), p=2, dim=1)


# ==================== ArcFace Loss with Adaptive Margin ====================


class MARGINLossHead(nn.Module):
    """
    ArcFace-style loss head with adaptive per-class margins and scales.

    Given cosine logits ``cos(θ)``, this head:

    1. Adds an **angular margin** *m* to the target-class angle: ``cos(θ + m)``.
    2. Multiplies the result by a **per-class scale** factor.

    Both *m* and the scales are updated adaptively via
    :meth:`update_adaptive_params`, which uses vMF concentration (kappa)
    estimates and ETF geometry.

    Parameters
    ----------
    num_classes : int
        Number of classes (C).
    base_scale : float
        Global base scale factor.
    alpha : float
        Confidence level for the chi-squared based margin computation.
    dim : int
        Embedding dimension.
    """

    def __init__(
        self,
        num_classes: int,
        base_scale: int,
        alpha: float,
        dim: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_scale = base_scale
        self.dim = dim
        self.alpha = alpha

        self.is_initialized = False

        self.margins = torch.zeros(num_classes)
        self.kappas = torch.zeros(num_classes)
        self.scales = torch.full((num_classes,), base_scale, dtype=torch.float32)

    def update_adaptive_params(
        self,
        kappas: torch.Tensor,
        class_counts: torch.Tensor,
        mean_prototypes: torch.Tensor,
    ):
        """
        Recompute adaptive margins and scales from current statistics.

        Called after each training epoch with new kappa estimates.

        Parameters
        ----------
        kappas : torch.Tensor
            Per-class vMF concentration, shape ``(C,)``.
        class_counts : torch.Tensor
            Per-class sample counts, shape ``(C,)``.
        mean_prototypes : torch.Tensor
            Per-class mean prototypes, shape ``(C, D)``.

        Returns
        -------
        tuple
            ``(margins, scales)`` — both ``torch.Tensor`` of shape ``(C,)``.
        """
        new_scales = compute_scale(kappas=kappas, base_scale=self.base_scale)

        new_margins = compute_margin(
            kappas=kappas,
            mean_prototypes=mean_prototypes,
            dim=self.dim,
            alpha=self.alpha,
        )

        # Update state
        self.kappas = kappas
        self.margins = new_margins
        self.scales = new_scales

        log.print(f"Updated margins: {self.margins}")
        log.print(f"Updated scales: {self.scales}")
        log.print(f"Updated kappas: {self.kappas}")

        return self.margins, self.scales

    def forward(self, cos_thetas, label_idxs):
        """
        Compute ArcFace loss with adaptive margins.

        For each sample *i* with target class *y_i*:

            logit_j = s_j · cos(θ_j)           for j ≠ y_i
            logit_yi = s_yi · cos(θ_yi + m_yi)  for the target class

        Parameters
        ----------
        cos_thetas : torch.Tensor
            Cosine logits, shape ``(B, C)``.
        label_idxs : torch.Tensor
            Ground-truth class indices, shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Scalar cross-entropy loss.
        """
        B, C = cos_thetas.shape
        device = cos_thetas.device

        # Clamp for numerical stability
        cos_thetas = torch.clamp(cos_thetas, -1 + 1e-7, 1 - 1e-7)

        # Fetch per-sample margin
        margins_batch = self.margins.to(device)[label_idxs].to(cos_thetas.dtype)
        cos_m = torch.cos(margins_batch)
        sin_m = torch.sin(margins_batch)

        # Extract target-class cosine and compute sin(θ)
        target_cos = cos_thetas[torch.arange(B, device=device), label_idxs]
        target_sin = torch.sqrt(torch.clamp(1.0 - target_cos**2, min=1e-7))

        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        target_cos_margin = target_cos * cos_m - target_sin * sin_m
        target_cos_margin = target_cos_margin.to(cos_thetas.dtype)

        # Replace target-class logit with margin-adjusted version
        logits = cos_thetas.clone()
        logits[torch.arange(B, device=device), label_idxs] = target_cos_margin

        # Apply per-class scale
        logits = logits * self.scales.to(device).unsqueeze(0)

        loss = F.cross_entropy(logits, label_idxs)
        return loss

    # def forward(self, cos_theta, label_idxs):
    #     B, C = cos_theta.shape

    #     # Margin for each sample
    #     margins = self.margins[label_idxs]  # [B]

    #     cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)

    #     # one-hot
    #     one_hot = F.one_hot(label_idxs, C).float()

    #     # CosFace: cos(theta) - m
    #     cos_theta_minus_m = cos_theta - margins.unsqueeze(1)

    #     # Only subtract margin for target class
    #     output = cos_theta * (1 - one_hot) + cos_theta_minus_m * one_hot

    #     # scale
    #     output = output * self.scales.unsqueeze(0)

    #     loss = F.cross_entropy(output, label_idxs)
    #     return loss

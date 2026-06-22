"""
Mathematical utilities for the MARGIN model.

Includes:
- vMF (von Mises-Fisher) concentration estimation.
- Adaptive scale and margin computation for the ArcFace-style loss head.
- Convergence coefficient for per-class training dynamics.
- Geometric median on the unit hypersphere (Weiszfeld algorithm).
"""

import math
import torch
import torch.nn.functional as F
from scipy.stats import chi2


def sigmoid(x):
    """
    Scalar sigmoid function (operates on a single float).

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        sigmoid(x)
    """
    x_tensor = torch.tensor([x], dtype=torch.float32)
    y_tensor = torch.sigmoid(x_tensor)
    return y_tensor.item()


def compute_vmf_kappa(features, prototype):
    """
    Estimate the von Mises-Fisher concentration parameter kappa.

    Uses the method-of-moments estimator based on the mean resultant length *r*
    of the cosine similarities between the normalised *features* and the
    normalised *prototype*.

    Parameters
    ----------
    features : torch.Tensor
        Normalised feature vectors, shape ``(N, d)``.
    prototype : torch.Tensor
        Prototype vector, shape ``(d,)``.

    Returns
    -------
    float
        Estimated kappa (>= 1e-6).  Returns 0.0 for empty input or degenerate
        cases.
    """
    if features.size(0) == 0:
        return 0.0

    features = torch.nn.functional.normalize(features, dim=1)
    prototype = torch.nn.functional.normalize(prototype, dim=0)

    cos_sim = torch.matmul(features, prototype)  # (N,)
    r = torch.mean(cos_sim).item()  # mean resultant length
    d = features.size(1)  # embedding dimensionality
    if r >= 1.0:
        r = 0.999999
    if r <= 0:
        return 0.0
    # Method-of-moments estimator for vMF
    kappa = r * (d - r * r) / (1 - r * r)
    return max(kappa, 1e-6)


def compute_scale(kappas: torch.Tensor, base_scale: float):
    """
    Compute adaptive per-class scale factors from concentration (kappa) values.

    Classes with higher kappa (tighter clusters) receive smaller scale factors,
    and vice versa, via a softmax reweighting that is reversed so that the
    dispersed class gets the largest scale boost.

    Parameters
    ----------
    kappas : torch.Tensor
        Per-class vMF concentration parameters, shape ``(C,)``.
    base_scale : float
        Global base scale factor.

    Returns
    -------
    torch.Tensor
        Per-class scales, shape ``(C,)``.
    """
    u = torch.log(kappas)
    C = kappas.shape[0]
    r = torch.softmax( - u / C, dim=0) * C
    new_scales = base_scale * r
    return new_scales


def compute_margin(
    kappas: torch.Tensor,
    mean_prototypes: torch.Tensor,  # reserved for future use
    dim: int,
    alpha: float = 0.95,
):
    """
    Compute adaptive per-class angular margins.

    Balances two forces:

    1. **vMF uncertainty cone** — derived from a chi-squared confidence region
       at level *alpha*.  Larger kappa (tighter cluster) → smaller cone.
    2. **ETF Voronoi cone** — half the ideal ETF angle, representing the
       minimal separation needed for a simplex ETF configuration.

    The final margin is ``max(excess_over_voronoi, fallback_vs_minimum, 0)``,
    ensuring the margin is never negative.

    Parameters
    ----------
    kappas : torch.Tensor
        Per-class vMF concentrations, shape ``(C,)``.
    mean_prototypes : torch.Tensor
        Per-class mean prototypes (reserved).
    dim : int
        Embedding dimension.
    alpha : float
        Confidence level for the chi-squared uncertainty cone (default 0.95).

    Returns
    -------
    torch.Tensor
        Per-class margins, shape ``(C,)``.
    """
    device = kappas.device
    C = kappas.shape[0]

    # return torch.full((C,), 0, dtype=torch.float32)

    # ==================================================================
    # 1. vMF uncertainty cone: radius of (1-alpha) confidence region
    # ==================================================================
    q = chi2.ppf(alpha, df=dim - 1)
    kappa_eff = torch.clamp(kappas, min=1.0)
    theta_vmf = torch.sqrt(torch.tensor(q, device=device) / kappa_eff)  # [C]

    # ==================================================================
    # 2. ETF Voronoi cone: half the ideal ETF angle
    # ==================================================================
    theta_voronoi = 0.5 * math.acos(-1 / (C - 1))

    # ==================================================================
    # 3. Global minimum cone (tightest uncertainty across all classes)
    # ==================================================================
    theta_minimum_angle = torch.min(theta_vmf)

    # ==================================================================
    # 4. Excess: how much the uncertainty cone exceeds the ETF ideal
    # ==================================================================
    theta_exceed = theta_vmf - theta_voronoi

    # ==================================================================
    # 5. Fallback: excess over the global minimum uncertainty
    # ==================================================================
    theta_fallback = theta_vmf - theta_minimum_angle
    theta_fallback = theta_fallback.expand_as(theta_vmf)

    # ==================================================================
    # 6. Final margin: max(excess, fallback, 0) — ensures non-negative
    # ==================================================================
    margins = torch.maximum(
        torch.maximum(theta_exceed, theta_fallback), torch.zeros_like(theta_vmf)
    )

    return margins


def compute_convergence_coefficient(
    n: int, count_i: int, kappa_i: float, dim: int, alpha: float = 0.95
):
    """
    Compute a per-class convergence coefficient (range [0, 1]).

    Compares the ETF Voronoi cone angle (ideal separation) against the vMF
    predictive uncertainty cone.  A value near 1 indicates that the class
    cluster is tight enough relative to the ideal ETF separation.

    Parameters
    ----------
    n : int
        Total number of classes (K).
    count_i : int
        Number of samples in class *i*.
    kappa_i : float
        vMF concentration for class *i*.
    dim : int
        Embedding dimension.
    alpha : float
        Confidence level for the chi-squared uncertainty cone.

    Returns
    -------
    float
        Convergence coefficient in [0, 1].
    """
    # Predictive uncertainty for this class (accounts for sample count)
    q = chi2.ppf(alpha, df=dim - 1)
    kappa_i_eff = kappa_i * count_i / (count_i + 1)
    theta_vmf = math.sqrt(q / kappa_i_eff)

    # ETF Voronoi cone angle (half the ideal ETF pairwise angle)
    theta_voronoi_cell = 0.5 * math.acos(-1 / (n - 1))
    convergence_coeff = theta_voronoi_cell / (theta_vmf)
    convergence_coeff = max(0.0, min(1.0, convergence_coeff))
    return convergence_coeff


def compute_geometric_median(
    X: torch.Tensor,
    max_iter: int = 100,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the geometric median of points on the unit sphere.

    Uses Weiszfeld's algorithm with projection back to the sphere after each
    iteration.  The geometric median is robust to outliers compared to the
    arithmetic mean.

    Parameters
    ----------
    X : torch.Tensor
        Point set, shape ``(N, D)``.  Should be L2-normalised.
    max_iter : int
        Maximum iterations (default 100).
    eps : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    torch.Tensor
        Geometric median, shape ``(D,)``, L2-normalised.
    """

    # Initialize on the sphere via the arithmetic mean
    y = F.normalize(X.mean(dim=0), dim=0)
    for _ in range(max_iter):
        dist = torch.norm(X - y, dim=1)
        # Prevent division by zero for points coincident with current estimate
        dist = torch.clamp(dist, min=eps)
        inv_dist = 1.0 / dist
        # Weiszfeld update: weighted sum of points
        y_new = (X * inv_dist[:, None]).sum(dim=0) / inv_dist.sum()
        # Project back to the unit sphere (critical for sphere-valued data)
        y_new = F.normalize(y_new, dim=0)
        # Convergence check
        if torch.norm(y - y_new) < eps:
            break
        y = y_new
    return y

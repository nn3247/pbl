"""
GMM モデル学習モジュール (GMM Trainer)

【責務】
- Train データのみを用いた GaussianMixture モデルの初期化とフィッティング (Strict Leakage-Free)
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from .class_conditional import ClassConditionalGMM

import sys
from pathlib import Path
_LEARNING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("clustering.gmm.trainer")


def fit_gmm(
    X_train: np.ndarray,
    n_clusters: int,
    cov_type: str = "full",
    random_state: int = 42,
    max_iter: int = 100,
    n_init: int = 5,
) -> GaussianMixture:
    """
    GMM モデルを初期化し、Train データのみで学習します。
    """
    log.info(f"GMM 学習開始 (k = {n_clusters}, 共分散: {cov_type}, max_iter: {max_iter})")
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=cov_type,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )
    gmm.fit(X_train)
    log.info(f"✓ GMM 学習完了 (収束: {gmm.converged_}, 反復回数: {gmm.n_iter_})")
    return gmm


def fit_class_conditional_gmm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components_neg: int = 1,
    n_components_pos: int = 2,
    cov_type: str = "diag",
    reg_covar: float = 1e-4,
    prior_mode: str = "balanced",
    logit_clip: float = 50.0,
    random_state: int = 42,
    max_iter: int = 100,
    n_init: int = 1,
) -> ClassConditionalGMM:
    """
    クラス条件付き GMM (gmm_0: y==0, gmm_1: y==1) を Train データのみで独立に学習します。
    """
    log.info(
        f"クラス条件付き GMM 学習開始 (k_neg={n_components_neg}, k_pos={n_components_pos}, "
        f"共分散: {cov_type}, reg_covar={reg_covar}, prior={prior_mode})"
    )
    cc_gmm = ClassConditionalGMM(
        n_components_neg=n_components_neg,
        n_components_pos=n_components_pos,
        covariance_type=cov_type,
        reg_covar=reg_covar,
        prior_mode=prior_mode,
        logit_clip=logit_clip,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )
    cc_gmm.fit(X_train, y_train)
    log.info(f"✓ クラス条件付き GMM 学習完了 (収束: {cc_gmm.converged_}, pi_0={cc_gmm.pi_0_:.3f}, pi_1={cc_gmm.pi_1_:.3f})")
    return cc_gmm

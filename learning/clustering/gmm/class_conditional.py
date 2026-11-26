"""
クラス条件付き GMM (Class-Conditional GMM / 教師ありクラスタリング) モジュール
(learning/clustering/gmm/class_conditional.py)

各クラス (y=0: リラックス/非集中, y=1: 集中) ごとに独立した GaussianMixture を
Train データのみで学習し、対数尤度と事前確率に基づくベイズの定理（対数オッズ + シグモイド）
により、ラベル反転やアンダーフローのない事後確率 y_soft = P(y=1 | x) を算出します。
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class ClassConditionalGMM:
    """
    2クラス (y ∈ {0, 1}) 向けのクラス条件付き GMM。
    sklearn の predict_proba(X) -> shape (N, 2) [P(y=0|x), P(y=1|x)] および
    predict(X) -> {0, 1} 互換のインタフェースを提供します。
    """

    def __init__(
        self,
        n_components_neg: int = 1,
        n_components_pos: int = 2,
        covariance_type: str = "diag",
        reg_covar: float = 1e-4,
        prior_mode: str = "balanced",
        logit_clip: float = 50.0,
        random_state: int = 42,
        max_iter: int = 100,
        n_init: int = 1,
    ):
        self.n_components_neg = int(n_components_neg)
        self.n_components_pos = int(n_components_pos)
        self.covariance_type = str(covariance_type)
        self.reg_covar = float(reg_covar)
        self.prior_mode = str(prior_mode).strip().lower()
        self.logit_clip = float(logit_clip)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)

        self.gmm_0_: GaussianMixture | None = None
        self.gmm_1_: GaussianMixture | None = None
        self.pi_0_: float = 0.5
        self.pi_1_: float = 0.5
        self.converged_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassConditionalGMM":
        """Train データ (X, y) のみを使用してクラス 0 / クラス 1 の GMM を独立に学習します。"""
        X_arr = np.asarray(X, dtype=np.float64)
        y_bin = (np.asarray(y, dtype=np.float64).ravel() >= 0.5).astype(int)

        X_0 = X_arr[y_bin == 0]
        X_1 = X_arr[y_bin == 1]
        if len(X_0) == 0 or len(X_1) == 0:
            raise ValueError(
                f"ClassConditionalGMM の学習には両クラスのサンプルが必要です "
                f"(n_neg={len(X_0)}, n_pos={len(X_1)})。"
            )

        k0 = max(1, min(self.n_components_neg, len(X_0)))
        k1 = max(1, min(self.n_components_pos, len(X_1)))

        self.gmm_0_ = GaussianMixture(
            n_components=k0,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
        ).fit(X_0)

        self.gmm_1_ = GaussianMixture(
            n_components=k1,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
        ).fit(X_1)

        if self.prior_mode == "empirical":
            pi1 = float(np.clip(len(X_1) / (len(X_0) + len(X_1)), 1e-6, 1.0 - 1e-6))
            self.pi_1_ = pi1
            self.pi_0_ = 1.0 - pi1
        else:
            # デフォルト: 均衡事前確率 (balanced: 0.5 / 0.5)
            self.pi_0_ = 0.5
            self.pi_1_ = 0.5

        self.converged_ = bool(self.gmm_0_.converged_ and self.gmm_1_.converged_)
        return self

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        """対数オッズ logit(x) = (log p(x|y=1) + log pi_1) - (log p(x|y=0) + log pi_0) を返します。"""
        if self.gmm_0_ is None or self.gmm_1_ is None:
            raise RuntimeError("ClassConditionalGMM は未学習です。先に fit(X, y) を呼び出してください。")

        X_arr = np.asarray(X, dtype=np.float64)
        log_p0 = self.gmm_0_.score_samples(X_arr)
        log_p1 = self.gmm_1_.score_samples(X_arr)

        logit = (log_p1 + np.log(self.pi_1_)) - (log_p0 + np.log(self.pi_0_))
        return np.clip(logit, -self.logit_clip, self.logit_clip)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """事後確率 [P(y=0|x), P(y=1|x)] (shape: [N, 2]) を返します。"""
        logit = self.predict_logit(X)
        p1 = 1.0 / (1.0 + np.exp(-logit))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1]).astype(np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """事後確率 P(y=1|x) >= 0.5 に基づく離散クラス予測 {0, 1} を返します。"""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

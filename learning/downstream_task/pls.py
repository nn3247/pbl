"""
Dual-Stream PLS (Partial Least Squares) 次元削減モジュール
(learning/downstream_task/pls.py)

視覚ストリーム (32d) と生体ストリーム (32d) を個別に教師ラベル y に対して射影し、
指定次元（例: 3d + 3d = 6d）に圧縮します。
"""

from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression


class DualStreamPLS:
    """視覚・生体ストリームを個別に次元削減する PLS クラス"""

    def __init__(self, n_vis: int = 3, n_bio: int = 3, scale: bool = True):
        self.n_vis = n_vis
        self.n_bio = n_bio
        self.scale = scale
        self.pls_vis = PLSRegression(n_components=n_vis, scale=scale)
        self.pls_bio = PLSRegression(n_components=n_bio, scale=scale)
        self.vis_signs_: np.ndarray = np.ones(n_vis, dtype=np.float32)
        self.bio_signs_: np.ndarray = np.ones(n_bio, dtype=np.float32)

    @staticmethod
    def _compute_component_signs(pls_model: PLSRegression, n_comps: int, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train データのみに基づき、各 PLS 成分の符号を y (および残差負荷量 y_loadings_) に対して一貫した向きに揃える"""
        signs = np.ones(n_comps, dtype=np.float32)
        y_load = getattr(pls_model, "y_loadings_", None)
        if y_load is not None:
            flat = np.asarray(y_load, dtype=np.float64).ravel()
            for c in range(min(n_comps, len(flat))):
                if flat[c] < 0:
                    signs[c] = -1.0
            return signs

        z_tr, _ = pls_model.transform(X, np.zeros((len(X), 1)))
        y_c = np.asarray(y, dtype=np.float64).ravel() - float(np.mean(y))
        for c in range(min(n_comps, z_tr.shape[1])):
            zc = z_tr[:, c] - float(np.mean(z_tr[:, c]))
            if float(np.dot(zc, y_c)) < 0:
                signs[c] = -1.0
        return signs

    def fit(self, X_vis: np.ndarray, X_bio: np.ndarray, y: np.ndarray):
        """Train データのみを用いて視覚・生体の PLS 射影軸を学習し、符号反転を防ぐ向き統一を行う"""
        self.pls_vis.fit(X_vis, y)
        self.pls_bio.fit(X_bio, y)
        self.vis_signs_ = self._compute_component_signs(self.pls_vis, self.n_vis, X_vis, y)
        self.bio_signs_ = self._compute_component_signs(self.pls_bio, self.n_bio, X_bio, y)
        return self

    def transform(self, X_vis: np.ndarray, X_bio: np.ndarray) -> np.ndarray:
        """学習済み射影軸（符号整列済み）に従って潜在特徴量 (z_vis, z_bio) を抽出し水平結合"""
        z_v, _ = self.pls_vis.transform(X_vis, np.zeros((len(X_vis), 1)))
        z_b, _ = self.pls_bio.transform(X_bio, np.zeros((len(X_bio), 1)))
        z_v = z_v * self.vis_signs_[:z_v.shape[1]]
        z_b = z_b * self.bio_signs_[:z_b.shape[1]]
        return np.concatenate([z_v, z_b], axis=1).astype(np.float32)

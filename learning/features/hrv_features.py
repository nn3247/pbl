"""
因果ウィンドウ (過去 45秒) に基づく HRV 4大指標 (mean_pp, rmssd, sdnn, sd2_sd1) 抽出モジュール
(learning/features/hrv_features.py)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

HRV_FEATURE_NAMES: List[str] = [
    "hrv_mean_pp",
    "hrv_rmssd",
    "hrv_sdnn",
    "hrv_sd2_sd1",
]


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def _align_hrv_by_keys(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_cols: List[str],
) -> Optional[np.ndarray]:
    """(username, start_time, end_time) をキーにしてキャッシュ済み HRV 特徴量を target_df 順に整列"""
    key_cols = [c for c in ["username", "start_time", "end_time"] if c in target_df.columns and c in source_df.columns]
    if not key_cols or not all(c in source_df.columns for c in feature_cols):
        return None

    t_keys = target_df[key_cols].copy()
    s_df = source_df[key_cols + feature_cols].copy()
    for c in key_cols:
        if c == "username":
            t_keys[c] = t_keys[c].astype(str)
            s_df[c] = s_df[c].astype(str)
        else:
            t_keys[c] = np.round(pd.to_numeric(t_keys[c], errors="coerce"), 4)
            s_df[c] = np.round(pd.to_numeric(s_df[c], errors="coerce"), 4)

    s_df = s_df.drop_duplicates(subset=key_cols, keep="first")
    merged = t_keys.merge(s_df, on=key_cols, how="left", sort=False)
    if len(merged) != len(target_df):
        return None
    return merged[feature_cols].to_numpy(dtype=np.float32)


def compute_causal_hrv_features(
    meta_df: pd.DataFrame,
    preprocess_features_dir: Optional[Path] = None,
    window_sec: float = 45.0,
    fs: float = 50.0,
    min_peaks: int = 10,
    cache_path: Optional[Path] = None,
    log=None,
) -> Tuple[np.ndarray, List[str]]:
    """
    各ウィンドウの end_time を終端とする過去 `window_sec` 秒 ([end_time - window_sec, end_time]) の
    リークフリー因果コンテキストから HRV 4大指標 (mean_pp, rmssd, sdnn, sd2_sd1) を算出します。

    Parameters
    ----------
    meta_df : pd.DataFrame
        `username` (または `subject_id`) と `end_time` を含むメタデータ
    preprocess_features_dir : Optional[Path]
        前処理済み被験者CSV (`bvp_selected`, `ppg_peaks`, `time`) のディレクトリ
    window_sec : float
        因果コンテキスト長 (秒, デフォルト 45.0)
    fs : float
        BVP 信号のサンプリング周波数 (Hz, デフォルト 50.0)
    min_peaks : int
        HRV 算出に必要な最小ピーク数 (未満は NaN)
    cache_path : Optional[Path]
        算出結果のキャッシュ CSV パス
    log : Optional[Logger]
        ロガー

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (shape=(len(meta_df), 4) の float32 配列, HRV_FEATURE_NAMES)
    """
    project_root = _ensure_project_root_on_path()
    if preprocess_features_dir is None:
        preprocess_features_dir = project_root / "preprocess" / "data" / "features"

    n_samples = len(meta_df)
    if n_samples == 0:
        return np.zeros((0, len(HRV_FEATURE_NAMES)), dtype=np.float32), list(HRV_FEATURE_NAMES)

    # キャッシュが存在し、キー整合が取れれば即座に返却
    if cache_path is not None and cache_path.exists():
        try:
            cached_df = pd.read_csv(cache_path)
            aligned = _align_hrv_by_keys(meta_df, cached_df, HRV_FEATURE_NAMES)
            if aligned is not None:
                if log is not None:
                    log.info(f"✓ キャッシュから HRV 4指標 ({len(HRV_FEATURE_NAMES)}次元) を読み込みました: {cache_path.name}")
                return aligned, list(HRV_FEATURE_NAMES)
        except Exception as e:
            if log is not None:
                log.warning(f"⚠️ HRV キャッシュの読み込みに失敗したため再計算します: {e}")

    from preprocess.features.rppg.hrv_extraction import (
        refine_peaks_parabolic,
        correct_ibi_outliers,
        compute_hrv_4metrics,
    )

    subject_col = "username" if "username" in meta_df.columns else "subject_id"
    if subject_col not in meta_df.columns or "end_time" not in meta_df.columns:
        raise ValueError("HRV 特徴量の算出には meta_df に 'username' (または 'subject_id') と 'end_time' が必要です。")

    out = np.full((n_samples, len(HRV_FEATURE_NAMES)), np.nan, dtype=np.float32)
    subjects = meta_df[subject_col].astype(str).to_numpy()
    end_times = pd.to_numeric(meta_df["end_time"], errors="coerce").to_numpy(dtype=np.float64)

    unique_users = np.unique(subjects)
    missing_users = []

    for user in unique_users:
        csv_path = preprocess_features_dir / f"{user}.csv"
        if not csv_path.exists():
            missing_users.append(user)
            continue

        df_sub = pd.read_csv(csv_path, usecols=["time", "bvp_selected", "ppg_peaks"])
        time_arr = df_sub["time"].to_numpy(dtype=np.float64)
        bvp_arr = df_sub["bvp_selected"].to_numpy(dtype=np.float64)
        peaks_raw = np.where(df_sub["ppg_peaks"].to_numpy() > 0)[0]

        if len(peaks_raw) < 4 or len(time_arr) < 4:
            continue

        dt = float(np.median(np.diff(time_arr))) if len(time_arr) > 1 else (1.0 / fs)
        fs_sub = (1.0 / dt) if dt > 0 else float(fs)

        peaks_ref_samples, _ = refine_peaks_parabolic(peaks_raw, bvp_arr, fs=fs_sub)
        peaks_clean_samples = correct_ibi_outliers(peaks_ref_samples, fs=fs_sub, method="Kubios")
        if len(peaks_clean_samples) < 4:
            continue

        t0 = float(time_arr[0])
        peaks_clean_sec = np.sort(t0 + peaks_clean_samples.astype(np.float64) / fs_sub)

        user_indices = np.where(subjects == user)[0]
        for idx in user_indices:
            w_end = end_times[idx]
            if np.isnan(w_end):
                continue
            w_start = max(t0, w_end - float(window_sec))
            lo = int(np.searchsorted(peaks_clean_sec, w_start, side="left"))
            hi = int(np.searchsorted(peaks_clean_sec, w_end, side="right"))
            w_peaks = peaks_clean_sec[lo:hi]
            if len(w_peaks) < min_peaks:
                continue

            metrics = compute_hrv_4metrics(w_peaks - w_start, fs=fs_sub)
            out[idx, 0] = metrics["mean_pp"]
            out[idx, 1] = metrics["rmssd"]
            out[idx, 2] = metrics["sdnn"]
            out[idx, 3] = metrics["sd2_sd1"]

    if missing_users and log is not None:
        log.warning(f"⚠️ HRV 抽出用の前処理 CSV が見つからない被験者: {missing_users}")

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            key_cols = [c for c in ["username", "start_time", "end_time"] if c in meta_df.columns]
            cache_df = meta_df[key_cols].copy() if key_cols else pd.DataFrame()
            for col_i, col_name in enumerate(HRV_FEATURE_NAMES):
                cache_df[col_name] = out[:, col_i]
            cache_df.to_csv(cache_path, index=False)
            if log is not None:
                log.info(f"✓ HRV 4指標キャッシュを保存しました: {cache_path}")
        except Exception as e:
            if log is not None:
                log.warning(f"⚠️ HRV キャッシュの保存をスキップしました: {e}")

    if log is not None:
        valid_ratio = float(np.mean(~np.isnan(out[:, 0]))) * 100.0
        log.info(
            f"✓ 過去 {window_sec:.0f}秒 因果ウィンドウから HRV 4指標 ({HRV_FEATURE_NAMES}) を算出しました "
            f"(有効率: {valid_ratio:.1f}%)"
        )

    return out, list(HRV_FEATURE_NAMES)

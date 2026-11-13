<!-- BEGIN:nextjs-agent-rules -->
## This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

##  AGENTS.md - Data Collect (データ収集システム)

本ドキュメントは、AIエージェント（Cursor, Claude Code, Antigravity, GitHub Copilot 等）が本プロジェクト（`data_collect`）のコードを理解・編集・運用する際の実装ガイドラインおよび開発規約です。

---

### 1. プロジェクト概要

オンライン授業・オンデマンド授業における学習者の集中度推定AIの構築を目的とした、**リアルタイム顔ランドマーク・生体特徴量計測および学習データ収集用Webアプリケーション**です。

#### 最重要プライバシー方針
- **映像データ・顔画像の非保持:** Webカメラ映像および顔画像の生データはサーバーへ一切送信・保存しません。
- **クライアント側での抽出:** ブラウザ内（MediaPipe Tasks-Vision）で抽出された20点の顔ランドマーク座標、および頬部のRGB値（rPPG用途）のみをFirebase Storageに安全にアップロードします。
- **匿名性の担保:** Firebase匿名認証およびランダム発行UID・識別用文字列でのみ管理します。

---

### 2. 技術スタック

- **フレームワーク:** Next.js 16 (App Router)
- **言語 / ランタイム:** TypeScript 5, React 19, Node.js v22以上
- **スタイリング:** Tailwind CSS v4 (`@tailwindcss/postcss`)
- **コンピュータビジョン / 特徴量抽出:** `@mediapipe/tasks-vision` (FaceLandmarker)
- **認証・ストレージ:** Firebase v12 (Firebase Auth 匿名認証, Firebase Storage)
- **QRコード認識:** `html5-qrcode`
- **ドキュメント表示:** `react-markdown`

---

### 3. 設計思想とコーディング規約

#### 3.1 プライバシー・セキュリティ最優先
- カメラの映像ストリーム（`<video>`, `<canvas>` の生ピクセル）を外部APIやサーバーへ送信するコードを絶対に記述しないこと。
- 送信対象は `CapturedFrame[]` に格納される特徴量データ（ランドマーク座標およびRGB統計値）のみに限定する。
- APIルート（`app/api/content`）へのアクセスは実験パスワード（`x-experiment-password` ヘッダー）または適切な検証を実施すること。

#### 3.2 計測精度とフレームレート管理
- 計測は `TARGET_FPS = 25` (約40ms間隔) を維持する。
- ランドマーク座標は `RawLandmarkCoord`（x, y, z）形式で保持し、抜粋20点（EAR用16点、頭部姿勢用2点、視線・虹彩用2点）を対象とする。
- キャリブレーション（5秒間の正面注視）期間中の平均座標は `CalibrationStats.meanLandmarks` として記録する。

#### 3.3 型安全性 (TypeScript)
- `any` や型アサーションの乱用を避け、`app/types.ts` に定義された `CapturedFrame`, `LandmarkPoints`, `RppgMetrics`, `MeasurementRecord` などの厳密な型を適用・保守すること。
- 新しい特徴量を追加する際は、必ず `app/types.ts` を同期更新すること。

#### 3.4 コンポーネントと責務分離
- **UIコンポーネント (`ConcentrationApp.tsx` 等):** 表示とユーザー操作イベントに専念する。
- **カスタムフック (`useMeasurement.ts` 等):** 計測ループ、ステート遷移、タイマー処理、Firebase連携ロジックをカプセル化する。
- **純粋関数 (`lib/landmarks.ts`, `lib/rppg.ts`):** 座標計算や色空間処理は副作用を持たない関数として実装・維持する。

#### 3.5 Next.js & React 19 の留意点
- クライアント側でDOMやフックを扱うファイル先頭には `"use client";` を明記すること。
- Next.js 16 では破壊的変更が含まれる場合があるため、最新のAPI規約に従うこと。

---


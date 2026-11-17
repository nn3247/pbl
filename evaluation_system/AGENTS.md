<!-- BEGIN:nextjs-agent-rules -->

## This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

## AGENTS.md - Evaluation System (評価用データ収集システム)

本ドキュメントは、AIエージェント（Cursor, Claude Code, Antigravity, GitHub Copilot 等）が本プロジェクト（`evaluation_system`）のコードを理解・編集・運用する際の実装ガイドラインおよび開発規約です。

---

### 1. プロジェクト概要

オンライン授業・オンデマンド授業における学習者の集中度推定AIモデルの評価を目的とした、**リアルタイム顔ランドマーク・生体特徴量計測および評価データ収集用Webアプリケーション**です。

#### 最重要プライバシー方針
- **映像データ・顔画像の非保持:** Webカメラ映像および顔画像の生データはサーバーへ一切送信・保存しません。
- **クライアント側での抽出:** ブラウザ内（MediaPipe Tasks-Vision）で抽出された20点の顔ランドマーク座標・顔メッシュ重心座標（`landmarkCentroid`）、および頬部12点ROIのRGB値（rPPG用途）のみをFirebase Storageに安全にアップロードします。
- **匿名性の担保:** Firebase匿名認証によるランダム発行UIDおよびセッションごとに生成される10桁の被験者ID（`subjectId`）でのみ管理し、個人情報との紐付けは行いません。

---

### 2. 技術スタック

- **フレームワーク:** Next.js 16 (App Router)
- **言語 / ランタイム:** TypeScript 5, React 19, Node.js v22以上
- **スタイリング:** Tailwind CSS v4 (`@tailwindcss/postcss`)
- **コンピュータビジョン / 特徴量抽出:** `@mediapipe/tasks-vision` (FaceLandmarker)
- **認証・ストレージ:** Firebase v13 (Firebase Auth 匿名認証, Firebase Storage)
- **QRコード認識:** `html5-qrcode`
- **ドキュメント表示:** `react-markdown`

---

### 3. 設計思想とコーディング規約

#### 3.1 プライバシー・セキュリティ最優先
- カメラの映像ストリーム（`<video>`, `<canvas>` の生ピクセル）を外部APIやサーバーへ送信するコードを絶対に記述しないこと。
- 送信対象は `MeasurementRecord`（`CapturedFrame[]` 等）に格納される特徴量データ（ランドマーク座標・重心・RGB統計値・実験メタデータ）のみに限定する。
- APIルート（`app/api/content`）へのアクセスは実験パスワード（`x-experiment-password` ヘッダー）または適切な検証を実施すること。

#### 3.2 計測精度とフレームレート管理
- 計測は `TARGET_FPS = 25`（約40ms間隔）を維持する。
- ランドマーク座標は `RawLandmarkCoord`（x, y, z）形式で保持し、抜粋20点（EAR用16点、頭部姿勢用2点、視線・虹彩用2点）および468点メッシュの重心（`landmarkCentroid`）を対象とする。
- rPPG計測は左右各6点（計12点）のROI個別RGBおよび左右・全体平均RGBを `RppgMetrics` として記録する。
- キャリブレーション（`CALIBRATION_DURATION = 5` 秒間の正面注視）期間中の平均座標は `CalibrationStats.meanLandmarks` として記録する。
- タスクモード（`video_mode`: `0`=Normal / `1`=Custom）および視聴動画に応じたラベル（`label`: `1`=cnn.mp4 / `0`=relax.mp4 / `2`=ts2vec.mp4）を正確に付与すること。

#### 3.3 型安全性 (TypeScript)
- `any` や型アサーションの乱用を避け、`app/types.ts` に定義された `CapturedFrame`, `LandmarkPoints`, `RppgMetrics`, `CalibrationStats`, `MeasurementRecord` などの厳密な型を適用・保守すること。
- 新しい特徴量やメタデータ（`experienced`, `subjectId` 等）を追加・変更する際は、必ず `app/types.ts` を同期更新すること。

#### 3.4 コンポーネントと責務分離
- **UIコンポーネント (`ConcentrationApp.tsx`, `TaskScreen.tsx`, `QrScanner.tsx`):** 事前同意フロー、モード選択、画面表示、ユーザー操作イベントに専念する。
- **カスタムフック (`useMeasurement.ts`, `useWebcam.ts`, `useFaceLandmarker.ts`):** 計測ループ、フェーズ遷移、キャリブレーション、タイマー処理、保存・遷移トリガーをカプセル化する。
- **純粋関数・ユーティリティ (`lib/landmarks.ts`, `lib/rppg.ts`, `lib/survey.ts`, `lib/firebase.ts`):** 座標計算・色空間処理・アンケートURL生成・Storage送信/ローカル保存処理を責務ごとに分離して実装・維持する。

#### 3.5 Next.js & React 19 の留意点
- クライアント側でDOMやフックを扱うファイル先頭には `"use client";` を明記すること。
- Next.js 16 では破壊的変更が含まれる場合があるため、`node_modules/next/dist/docs/` 等を参照し最新のAPI規約に従うこと。

---

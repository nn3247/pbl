// ============================================================
// lib/firebase.ts
// Firebase初期化、認証、およびgzip圧縮データ送信ユーティリティ
// ============================================================

import { initializeApp, getApps } from "firebase/app";
import { getStorage, ref, uploadBytes, FirebaseStorage } from "firebase/storage";
import { getAuth } from "firebase/auth";
import type { CalibrationStats, CapturedFrame } from "../types";

/** Firebase StorageおよびAuthのインスタンスを返す */
const firebaseConfig = {
  apiKey:             process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain:         process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId:          process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket:      process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId:  process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId:              process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];

export function getFirebaseStorage(): FirebaseStorage | null {
  try {
    return getStorage(app);
  } catch (e) {
    console.error("Firebase Storage初期化エラー:", e);
    return null;
  }
}

export function getFirebaseAuth() {
  try {
    return getAuth(app);
  } catch (e) {
    console.error("Firebase Auth初期化エラー:", e);
    return null;
  }
}

/**
 * 💡 指定された長さの完全ランダムな英数字文字列を生成する
 * 対象文字: アルファベット大文字、小文字、数字 (A-Z, a-z, 0-9)
 */
export function generateRandomString(length: number = 10): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  
  // ブラウザのより安全な暗号ランダムAPIを使用（SSRエラー防止のためwindowチェック入り）
  if (typeof window !== "undefined" && window.crypto && window.crypto.getRandomValues) {
    const array = new Uint32Array(length);
    window.crypto.getRandomValues(array);
    for (let i = 0; i < length; i++) {
      result += chars[array[i] % chars.length];
    }
  } else {
    // フォールバック（予備のランダム処理）
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
  }
  return result;
}

/** 被験者ID（10桁の英数字）を生成する */
export const generateSubjectId = (length: number = 10): string => generateRandomString(length);

/**
 * JSON文字列をgzip圧縮してUint8Arrayに変換する
 */
async function gzipJson(data: object): Promise<Uint8Array> {
  const json   = JSON.stringify(data);
  const stream = new Blob([json]).stream().pipeThrough(new CompressionStream("gzip"));
  const buffer = await new Response(stream).arrayBuffer();
  return new Uint8Array(buffer);
}




/**
 * 計測データをgzip圧縮してFirebase Storageにアップロードする
 * @returns 割り当てられた被験者ID (10桁文字列)
 */
export async function uploadMeasurement(
  storage: FirebaseStorage,
  frames: CapturedFrame[],
  calibration: CalibrationStats | undefined,
  uid: string,
  videoMode?: 0 | 1,
  subjectId?: string,
  experienced?: 0 | 1
): Promise<string> {
  const finalSubjectId = subjectId ? subjectId.trim() : generateSubjectId(10);
  const clientCreatedAtMs = Date.now();
  const fileData = {
    subjectId: finalSubjectId,
    clientCreatedAtMs,
    experienced: experienced ?? 0,
    video_mode: videoMode,
    calibration,
    data: frames,
  };

  const compressed = await gzipJson(fileData);
  
  // ── 💡 ファイル名を10桁の被験者IDに統一 ──────────────────────────────
  const storageRef = ref(storage, `user_signals/${uid}/pbl-b18-${finalSubjectId}.json.gz`);
  // ─────────────────────────────────────────────────────────────────────────

  await uploadBytes(storageRef, compressed, {
    contentType:     "application/json",
    contentEncoding: "gzip",
  });

  return finalSubjectId;
}

/**
 * 計測データをJSONファイルとしてローカルにダウンロードする
 * @returns 割り当てられた被験者ID (10桁文字列)
 */
export function downloadMeasurementJson(
  frames: CapturedFrame[],
  calibration?: CalibrationStats,
  videoMode?: 0 | 1,
  subjectId?: string,
  experienced?: 0 | 1
): string {
  const finalSubjectId = subjectId ? subjectId.trim() : generateSubjectId(10);
  const clientCreatedAtMs = Date.now();
  const fileData = {
    subjectId:  finalSubjectId,
    clientCreatedAtMs,
    experienced: experienced ?? 0,
    video_mode: videoMode,
    timestamp:  new Date().toISOString(),
    calibration,
    data:       frames,
  };
  const blob = new Blob([JSON.stringify(fileData, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);

  const a    = document.createElement("a");
  a.href     = url;
  a.download = `pbl-b18-${finalSubjectId}.json`;

  a.click();
  URL.revokeObjectURL(url);
  return finalSubjectId;
}
// ============================================================
// app/api/content/route.ts
// パスワード保護されたマークダウンファイルを返すAPIルート（informed_consent追加版）
// ============================================================

import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";
import crypto from "crypto";

// 簡易インメモリ・レートリミッター（パスワード総当たり攻撃対策：認証失敗のみカウント）
interface RateLimitRecord {
  count: number;
  resetTime: number;
}
const rateLimitMap = new Map<string, RateLimitRecord>();
const RATE_LIMIT_WINDOW_MS = 60 * 1000; // 1分間
const MAX_FAILED_ATTEMPTS_PER_WINDOW = 15; // 1分間に認証失敗 最大15回

const ALLOWED_FILES = ["terms", "privacy", "introduction", "informed_consent"] as const;
type AllowedFile = (typeof ALLOWED_FILES)[number];

function cleanupExpiredRecords(now: number): void {
  if (rateLimitMap.size <= 5000) return;
  for (const [key, record] of rateLimitMap.entries()) {
    if (now > record.resetTime) {
      rateLimitMap.delete(key);
    }
  }
}

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  const record = rateLimitMap.get(ip);
  if (!record || now > record.resetTime) {
    return false;
  }
  return record.count >= MAX_FAILED_ATTEMPTS_PER_WINDOW;
}

function recordFailedAttempt(ip: string): void {
  const now = Date.now();
  cleanupExpiredRecords(now);

  const record = rateLimitMap.get(ip);
  if (!record || now > record.resetTime) {
    rateLimitMap.set(ip, { count: 1, resetTime: now + RATE_LIMIT_WINDOW_MS });
    return;
  }
  record.count += 1;
}

export async function GET(req: NextRequest) {
  // 🔒 0. レート制限のチェック（過去の認証失敗回数が上限に達している場合のみブロック）
  const ip =
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    req.headers.get("x-real-ip") ||
    "unknown";

  if (isRateLimited(ip)) {
    return NextResponse.json(
      { error: "Too many failed attempts. Please try again later." },
      { status: 429, headers: { "Retry-After": "60" } }
    );
  }

  // 🔒 1. パスワードの取得と入力文字数上限チェック
  const clientPassword = req.headers.get("x-experiment-password") || "";
  const correctPassword = process.env.EXPERIMENT_PASSWORD || "";

  // 64文字以上の入力は不正なリクエストとして即時400エラー
  if (clientPassword.length > 64) {
    recordFailedAttempt(ip);
    return NextResponse.json(
      { error: "Bad Request: Password exceeds maximum allowed length" },
      { status: 400 }
    );
  }

  // 環境変数未設定の場合は401
  if (!correctPassword) {
    return NextResponse.json(
      { error: "Unauthorized: Invalid password" },
      { status: 401 }
    );
  }

  // タイミング攻撃を防ぐ定数時間比較（バイト長の一致を確認して例外を防止）
  const a = Buffer.from(clientPassword);
  const b = Buffer.from(correctPassword);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
    recordFailedAttempt(ip);
    return NextResponse.json(
      { error: "Unauthorized: Invalid password" },
      { status: 401 }
    );
  }

  // 2. ファイル名のバリデーション（"all" 指定時は4ファイルを1リクエストでまとめて返却）
  const { searchParams } = new URL(req.url);
  const file = searchParams.get("file");

  if (file === "all") {
    try {
      const entries = await Promise.all(
        ALLOWED_FILES.map(async (name) => {
          const filePath = path.join(process.cwd(), "content", `${name}.md`);
          const content = await fs.readFile(filePath, "utf-8");
          return [name, content] as const;
        })
      );
      return NextResponse.json({ contents: Object.fromEntries(entries) });
    } catch {
      return NextResponse.json({ error: "file not found" }, { status: 404 });
    }
  }

  if (!file || !ALLOWED_FILES.includes(file as AllowedFile)) {
    return NextResponse.json({ error: "invalid file" }, { status: 400 });
  }

  // 3. 単一ファイルの読み込みと返却
  const filePath = path.join(process.cwd(), "content", `${file}.md`);

  try {
    const content = await fs.readFile(filePath, "utf-8");
    return NextResponse.json({ content });
  } catch {
    return NextResponse.json({ error: "file not found" }, { status: 404 });
  }
}
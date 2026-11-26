// ============================================================
// ConcentrationApp.tsx
// UIのみを担当するメインコンポーネント（インフォームドコンセント・モード選択追加版）
// ============================================================

"use client";

import React, { useRef, useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import TaskScreen from "./components/TaskScreen";

const mdComponents: Components = {
  h1: ({ children }) => <h1 className="text-2xl font-bold mt-4 mb-2">{children}</h1>,
  h2: ({ children }) => <h2 className="text-xl font-bold mt-3 mb-1">{children}</h2>,
  h3: ({ children }) => <h3 className="text-base font-bold mt-2 mb-1">{children}</h3>,
  p:  ({ children }) => <p  className="text-sm leading-relaxed mb-2">{children}</p>,
  li: ({ children }) => <li className="text-sm ml-4 list-disc">{children}</li>,
};

interface ConcentrationAppProps {
  experimentPassword: string;
}

export default function ConcentrationApp({ experimentPassword }: ConcentrationAppProps) {
  const consentScrollRef = useRef<HTMLDivElement>(null);
  const introScrollRef  = useRef<HTMLDivElement>(null);

  // ── 画面遷移の状態 ────────────────────────────────────────
  const [consentAgreed,        setConsentAgreed]        = useState(false);
  const [agreed,               setAgreed]               = useState(false);
  const [introRead,            setIntroRead]            = useState(false);
  const [experienced,          setExperienced]          = useState<0 | 1 | null>(null);
  const [termsChecked,         setTermsChecked]         = useState(false);
  const [privacyChecked,       setPrivacyChecked]       = useState(false);
  const [consentScrolledToEnd, setConsentScrolledToEnd] = useState(false);
  const [introScrolledToEnd,   setIntroScrolledToEnd]   = useState(false);

  // ── タスクモード選択 ─────────────────────
  const [selectedMode,         setSelectedMode]         = useState<"normal" | "custom" | null>(null);

  // ── コンテンツ取得 ────────────────────────────────────────
  const [consentContent, setConsentContent] = useState("");
  const [termsContent,   setTermsContent]   = useState("");
  const [privacyContent, setPrivacyContent] = useState("");
  const [introContent,   setIntroContent]   = useState("");

  // ── 各種Markdownの一括fetch ──────────────────────────────────
  useEffect(() => {
    if (!experimentPassword) return;

    let cancelled = false;
    fetch("/api/content?file=all", {
      headers: {
        "x-experiment-password": experimentPassword,
      },
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: { contents?: Record<string, string> }) => {
        if (cancelled) return;
        const contents = d.contents ?? {};
        setConsentContent(contents.informed_consent ?? "取得できませんでした");
        setTermsContent(contents.terms ?? "取得できませんでした");
        setPrivacyContent(contents.privacy ?? "取得できませんでした");
        setIntroContent(contents.introduction ?? "取得できませんでした");
      })
      .catch(() => {
        if (cancelled) return;
        setConsentContent("取得できませんでした");
        setTermsContent("取得できませんでした");
        setPrivacyContent("取得できませんでした");
        setIntroContent("取得できませんでした");
      });

    return () => {
      cancelled = true;
    };
  }, [experimentPassword]);

  // ── スクロール位置のリセットと末尾検知 ──────────────────────
  useEffect(() => {
    if (consentScrollRef.current) {
      consentScrollRef.current.scrollTop = 0;
    }
  }, [consentContent]);

  useEffect(() => {
    if (introScrollRef.current) {
      introScrollRef.current.scrollTop = 0;
    }
  }, [introContent]);

  const handleConsentScroll = () => {
    const el = consentScrollRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
    if (atBottom) setConsentScrolledToEnd(true);
  };

  const handleIntroScroll = () => {
    const el = introScrollRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
    if (atBottom) setIntroScrolledToEnd(true);
  };

  // ── 1. 説明と同意書 (Informed Consent) 画面 ──────────────────
  if (!consentAgreed) {
    return (
      <div key="screen-consent" className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
        <h1 className="text-2xl font-bold mb-6">インフォームドコンセント</h1>
        <div className="w-full max-w-2xl bg-gray-800 rounded-lg p-4 flex flex-col">
          <div
            key="scroll-consent"
            ref={consentScrollRef}
            onScroll={handleConsentScroll}
            className="h-96 overflow-y-auto bg-gray-900 rounded p-4 text-gray-300"
          >
            {consentContent ? (
              <ReactMarkdown components={mdComponents}>{consentContent}</ReactMarkdown>
            ) : (
              <p className="text-gray-400 text-center py-8">同意書を読み込み中...</p>
            )}
          </div>
          {!consentScrolledToEnd ? (
            <p className="text-center text-sm text-gray-400 mt-3 animate-pulse">
              ↓ 最後までスクロールしてください
            </p>
          ) : (
            <p className="text-center text-sm text-green-400 mt-3 font-medium">
              ✓ 最後まで確認しました
            </p>
          )}
          <button
            onClick={() => setConsentAgreed(true)}
            disabled={!consentScrolledToEnd}
            className="mt-4 w-full py-3 bg-blue-600 hover:bg-blue-500 rounded-lg font-bold text-lg disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            内容を理解し、同意する
          </button>
        </div>
      </div>
    );
  }

  // ── 2. 利用規約とプライバシーポリシー同意画面 ──────────────────
  if (!agreed) {
    return (
      <div key="screen-terms" className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
        <h1 className="text-2xl font-bold mb-6">Data Collect</h1>
        <div className="w-full max-w-2xl space-y-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-lg font-bold mb-3">利用規約</h2>
            <div className="h-48 overflow-y-auto bg-gray-900 rounded p-3 text-gray-300">
              <ReactMarkdown components={mdComponents}>{termsContent}</ReactMarkdown>
            </div>
            <label className="flex items-center gap-2 mt-3 cursor-pointer">
              <input type="checkbox" checked={termsChecked}
                onChange={(e) => setTermsChecked(e.target.checked)}
                className="w-4 h-4 accent-orange-500" />
              <span className="text-sm">利用規約に同意する</span>
            </label>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-lg font-bold mb-3">プライバシーポリシー</h2>
            <div className="h-48 overflow-y-auto bg-gray-900 rounded p-3 text-gray-300">
              <ReactMarkdown components={mdComponents}>{privacyContent}</ReactMarkdown>
            </div>
            <label className="flex items-center gap-2 mt-3 cursor-pointer">
              <input type="checkbox" checked={privacyChecked}
                onChange={(e) => setPrivacyChecked(e.target.checked)}
                className="w-4 h-4 accent-orange-500" />
              <span className="text-sm">プライバシーポリシーに同意する</span>
            </label>
          </div>
          <button
            onClick={() => setAgreed(true)}
            disabled={!termsChecked || !privacyChecked}
            className="w-full py-3 bg-orange-600 hover:bg-orange-500 rounded-lg font-bold text-lg disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            ☑ 同意して開始する
          </button>
        </div>
      </div>
    );
  }

  // ── 3. 導入（はじめに）画面 ──────────────────────────────
  if (!introRead) {
    return (
      <div key="screen-intro" className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
        <h1 className="text-2xl font-bold mb-6">はじめに</h1>
        <div className="w-full max-w-2xl bg-gray-800 rounded-lg p-4 flex flex-col">
          <div
            key="scroll-intro"
            ref={introScrollRef}
            onScroll={handleIntroScroll}
            className="h-96 overflow-y-auto bg-gray-900 rounded p-4 text-gray-300"
          >
            {introContent ? (
              <ReactMarkdown components={mdComponents}>{introContent}</ReactMarkdown>
            ) : (
              <p className="text-gray-400 text-center py-8">説明文を読み込み中...</p>
            )}
          </div>
          {!introScrolledToEnd ? (
            <p className="text-center text-sm text-gray-400 mt-3 animate-pulse">
              ↓ 最後までスクロールしてください
            </p>
          ) : (
            <p className="text-center text-sm text-green-400 mt-3 font-medium">
              ✓ 最後まで確認しました
            </p>
          )}
          <button
            onClick={() => setIntroRead(true)}
            disabled={!introScrolledToEnd}
            className="mt-4 w-full py-3 bg-orange-600 hover:bg-orange-500 rounded-lg font-bold text-lg disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            全て読みました
          </button>
        </div>
      </div>
    );
  }

  // ── 4. 学習データ収集への協力経験確認画面 ────────────────────
  if (experienced === null) {
    return (
      <div key="screen-experience-check" className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
        <div className="w-full max-w-lg bg-gray-800 rounded-2xl p-8 shadow-2xl border border-gray-700 text-center">
          <h2 className="text-2xl font-bold mb-3">事前データ収集への協力確認</h2>
          <p className="text-gray-400 text-sm mb-8 leading-relaxed">
            以前実施された「学習データ収集」に協力したことがありますか？<br />
            当てはまる方を選択してください。
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <button
              onClick={() => setExperienced(1)}
              className="py-5 px-6 bg-gradient-to-br from-emerald-600 to-teal-700 hover:from-emerald-500 hover:to-teal-600 rounded-xl font-bold text-xl shadow-lg hover:shadow-emerald-500/20 active:scale-95 transition-all flex flex-col items-center justify-center gap-2 group cursor-pointer"
            >
              <span>協力した</span>
              <span className="text-xs font-normal text-emerald-100 opacity-80 group-hover:opacity-100">
                データ収集に参加経験あり
              </span>
            </button>

            <button
              onClick={() => setExperienced(0)}
              className="py-5 px-6 bg-gradient-to-br from-slate-600 to-gray-700 hover:from-slate-500 hover:to-gray-600 rounded-xl font-bold text-xl shadow-lg hover:shadow-gray-500/20 active:scale-95 transition-all flex flex-col items-center justify-center gap-2 group cursor-pointer"
            >
              <span>協力していない</span>
              <span className="text-xs font-normal text-gray-200 opacity-80 group-hover:opacity-100">
                今回が初めての参加
              </span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── 5. タスク選択画面 ────────────────────
  if (!selectedMode) {
    return (
      <div key="screen-select-mode" className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
        <div className="w-full max-w-lg bg-gray-800 rounded-2xl p-8 shadow-2xl border border-gray-700 text-center">
          <h2 className="text-2xl font-bold mb-3">タスクを選択してください</h2>
          <p className="text-gray-400 text-sm mb-8 leading-relaxed">
            実施する実験の種類を選択してください。<br />
            選択したタスクの画面へ遷移します。
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <button
              onClick={() => setSelectedMode("normal")}
              className="py-5 px-6 bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 rounded-xl font-bold text-xl shadow-lg hover:shadow-blue-500/20 active:scale-95 transition-all flex flex-col items-center justify-center gap-2 group cursor-pointer"
            >
              <span>Normal</span>
              <span className="text-xs font-normal text-green-100 opacity-80 group-hover:opacity-100">
                こちらを選択して下さい。
              </span>
            </button>

            <button
              onClick={() => setSelectedMode("custom")}
              className="py-5 px-6 bg-gradient-to-br from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 rounded-xl font-bold text-xl shadow-lg hover:shadow-purple-500/20 active:scale-95 transition-all flex flex-col items-center justify-center gap-2 group cursor-pointer"
            >
              <span>Custom</span>
              <span className="text-xs font-normal text-purple-100 opacity-80 group-hover:opacity-100">
                実施者が指示した場合にのみ選択してください。
              </span>
            </button>
          </div>

          <div className="mt-6 pt-4 border-t border-gray-700/60">
            <button
              onClick={() => setExperienced(null)}
              className="text-xs text-gray-400 hover:text-gray-200 transition"
            >
              ← 協力経験の選択に戻る
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── 6. メイン計測画面 ───────────────
  const isNormal = selectedMode === "normal";
  return (
    <TaskScreen
      key={`task-${selectedMode}`}
      mode={selectedMode}
      videoMode={isNormal ? 0 : 1}
      experienced={experienced}
      measureVideoSrc={isNormal ? "/cnn.mp4" : "/ts2vec.mp4"}
      endVideoSrc={isNormal ? "/relax.mp4" : undefined}
      onBack={() => setSelectedMode(null)}
    />
  );
}

// ============================================================
// lib/survey.ts
// Google フォーム（事後アンケート）連携および自動遷移ユーティリティ
// ============================================================

/**
 * 被験者IDを埋め込んだ事前入力用GoogleフォームURLを生成する
 * 
 * @param subjectId 被験者ID (10桁の英数字文字列)
 * @returns 事前入力用GoogleフォームURL (設定がない場合は null)
 */
export function buildSurveyFormUrl(subjectId: string): string | null {
  const baseUrl = process.env.NEXT_PUBLIC_SURVEY_FORM_BASE_URL;
  const entryId = process.env.NEXT_PUBLIC_SURVEY_FORM_SUBJECT_ENTRY_ID;

  if (!baseUrl) {
    console.warn("⚠️ [Survey] アンケートURL (NEXT_PUBLIC_SURVEY_FORM_BASE_URL) が設定されていません");
    return null;
  }

  const cleanSubjectId = subjectId ? subjectId.trim() : "";
  if (!cleanSubjectId || !entryId) {
    return baseUrl;
  }

  // URLに既にクエリパラメータが含まれているか判定
  const separator = baseUrl.includes("?") ? "&" : "?";
  const encodedEntry = encodeURIComponent(entryId.trim());
  const encodedValue = encodeURIComponent(cleanSubjectId);

  return `${baseUrl}${separator}${encodedEntry}=${encodedValue}`;
}

/**
 * 指定ミリ秒待機後にGoogleフォームへ同一タブでリダイレクトする
 *
 * @param subjectId 被験者ID
 * @param delayMs リダイレクトまでの待機時間 (ミリ秒)
 * @returns キャンセル用タイマーID (リダイレクトをスキップした場合は null)
 */
export function redirectToSurvey(
  subjectId: string,
  delayMs = 1500
): ReturnType<typeof setTimeout> | null {
  const targetUrl = buildSurveyFormUrl(subjectId);
  if (!targetUrl) {
    console.warn("⚠️ [Survey] 遷移先URLが取得できなかったためリダイレクトをスキップします");
    return null;
  }

  if (typeof window === "undefined") return null;

  return setTimeout(() => {
    window.location.href = targetUrl;
  }, delayMs);
}

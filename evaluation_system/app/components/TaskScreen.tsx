// ============================================================
// components/TaskScreen.tsx
// 計測を行うメイン画面コンポーネント
// ============================================================

"use client";

import React, { useRef, useEffect } from "react";
import { useFaceLandmarker } from "../hooks/useFaceLandmarker";
import { useWebcam }         from "../hooks/useWebcam";
import { useMeasurement }    from "../hooks/useMeasurement";

export interface TaskScreenProps {
  mode: "normal" | "custom";
  videoMode: 0 | 1;
  experienced?: 0 | 1;
  measureVideoSrc?: string;
  endVideoSrc?: string;
  onBack?: () => void;
}

export default function TaskScreen({
  mode,
  videoMode,
  experienced = 0,
  measureVideoSrc = "/class.mp4",
  endVideoSrc = "/campfire.mp4",
  onBack,
}: TaskScreenProps) {
  const canvasRef       = useRef<HTMLCanvasElement>(null);
  const measureVideoRef = useRef<HTMLVideoElement>(null);
  const endVideoRef     = useRef<HTMLVideoElement>(null);
  const containerRef    = useRef<HTMLDivElement>(null);

  // ── モデル初期化・カメラ起動・計測ループ ──────────────────
  const { faceLandmarkerRef, isReady, error: modelError } = useFaceLandmarker();
  const { videoRef, isRunning, startCamera, error: cameraError } = useWebcam();
  const {
    phase,
    isMeasuring,
    isFaceDetected,
    calibrationTime,
    status,
    startLoop,
    startMeasurement,
    handleVideoEnd,
    handleEndVideoEnd,
    handleTimeUpdate,
  } = useMeasurement({
    videoRef,
    canvasRef,
    measureVideoRef,
    endVideoRef,
    faceLandmarkerRef,
    videoMode,
    mode,
    experienced,
  });

  const isWatchingVideo = phase === "watching_video";
  const isEndingVideo   = phase === "ending_video";
  const isSessionActive = phase !== "idle";

  const displayStatus = modelError ?? cameraError ?? status;
  const handleStartCamera = () => startCamera(startLoop);

  const handleStartMeasurement = () => {
    const mVideo = measureVideoRef.current;
    const eVideo = endVideoRef.current;
    if (mVideo) {
      mVideo.muted = false;
      mVideo.volume = 1.0;
      mVideo.play().then(() => {
        mVideo.pause();
        mVideo.currentTime = 0;
      }).catch((err) => console.log(err));
    }
    if (eVideo && endVideoSrc) {
      eVideo.muted = false;
      eVideo.volume = 1.0;
      eVideo.play().then(() => {
        eVideo.pause();
        eVideo.currentTime = 0;
      }).catch((err) => console.log(err));
    }
    startMeasurement();
  };

  useEffect(() => {
    if (isEndingVideo && endVideoRef.current && endVideoSrc) {
      const video = endVideoRef.current;
      video.muted = false;
      video.currentTime = 0;
      video.play().catch((err) => console.error(err));
    }
  }, [isEndingVideo, phase, endVideoSrc]);

  const isModelLoading = !isReady;

  return (
    <div className="flex flex-col items-center p-5 bg-gray-900 min-h-screen text-white">
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-2xl font-bold">Data Collect</h1>
        <span
          className={`px-3 py-1 text-xs font-semibold rounded-full uppercase tracking-wider ${
            mode === "normal"
              ? "bg-blue-600/30 text-blue-400 border border-blue-500/40"
              : "bg-purple-600/30 text-purple-400 border border-purple-500/40"
          }`}
        >
          {mode === "normal" ? "Normal" : "Custom"}
        </span>
      </div>

      <p className="mb-4 text-gray-300">{displayStatus}</p>

      <div className="flex items-center gap-4 w-full max-w-5xl">
        <div ref={containerRef} className="relative flex-1 bg-black rounded-lg overflow-hidden border-2 border-gray-700 aspect-video">
          <video ref={videoRef} autoPlay playsInline muted className="absolute top-0 left-0 w-full h-full object-cover z-0" />
          <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none z-10" />
          <video
            ref={measureVideoRef}
            src={measureVideoSrc}
            playsInline
            preload="auto"
            onEnded={handleVideoEnd}
            onTimeUpdate={handleTimeUpdate}
            className={`absolute top-0 left-0 w-full h-full object-cover z-20 ${
              isWatchingVideo ? "opacity-100" : "opacity-0 pointer-events-none"
            }`}
          />
          {endVideoSrc && (
            <video
              ref={endVideoRef}
              src={endVideoSrc}
              playsInline
              preload="auto"
              onEnded={handleEndVideoEnd}
              className={`absolute top-0 left-0 w-full h-full object-cover z-30 ${
                isEndingVideo ? "opacity-100" : "opacity-0 pointer-events-none"
              }`}
            />
          )}

          {phase === "calibrating" && (
            <div className="absolute top-0 left-0 w-full h-full z-30 pointer-events-none flex flex-col items-center justify-center gap-4 bg-black/40">
              <p className="text-2xl font-bold text-white">画面を見て目を開けてください</p>
              <div className="w-48 h-3 bg-gray-600 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-400 rounded-full transition-all duration-100"
                  style={{ width: `${((5 - calibrationTime) / 5) * 100}%` }}
                />
              </div>
              <p className="text-lg font-mono text-green-300">{Math.ceil(calibrationTime)} 秒</p>
            </div>
          )}

          {isMeasuring && !isFaceDetected && (
            <div className="absolute top-0 left-0 w-full h-full z-40 pointer-events-none flex items-center justify-center">
              <div className="bg-red-600/90 text-white px-8 py-5 rounded-xl text-2xl font-bold animate-pulse shadow-lg text-center">
                ⚠️ 警告<br />
                <span className="text-lg font-normal">顔が認識できません。カメラの正面を見てください。</span>
              </div>
            </div>
          )}

          {/* 💡 データ保存中およびアンケート遷移時のローディング表示 */}
          {(status.includes("アンケートへ移動します") || status.includes("データ保存・送信中")) && (
            <div className="absolute top-0 left-0 w-full h-full z-50 flex flex-col items-center justify-center gap-4 bg-black/80 backdrop-blur-sm">
              <div className="w-12 h-12 border-4 border-blue-400 border-t-transparent rounded-full animate-spin" />
              <p className="text-xl font-bold text-white text-center px-6">{status}</p>
              <p className="text-sm text-gray-300">アンケートページへ自動的に切り替わります...</p>
            </div>
          )}
        </div>
      </div>

      <div className="mt-6 flex items-center space-x-4">
        <button
          onClick={handleStartCamera}
          disabled={isRunning || isModelLoading}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-500 rounded disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          {isModelLoading ? "AIモデル読み込み中..." : "カメラ起動"}
        </button>
        <button
          onClick={handleStartMeasurement}
          disabled={!isRunning || isSessionActive}
          className="px-6 py-2 bg-orange-600 hover:bg-orange-500 rounded disabled:opacity-50 transition"
        >
          計測開始
        </button>
        {onBack && !isSessionActive && !isRunning && (
          <button
            onClick={onBack}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm text-gray-300 transition"
          >
            ← 選択に戻る
          </button>
        )}
      </div>
    </div>
  );
}


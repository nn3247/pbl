"use client";

import React, { useEffect, useRef, useState } from "react";
import { Html5Qrcode } from "html5-qrcode";

interface QrScannerProps {
  onScanSuccess: (decodedText: string) => void;
  onError?: (error: string) => void;
}

export default function QrScanner({ onScanSuccess, onError }: QrScannerProps) {
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(true);
  const scannerRef = useRef<Html5Qrcode | null>(null);

  useEffect(() => {
    let isCancelled = false;
    let isStopping = false;
    let localScanner: Html5Qrcode | null = null;
    let startPromise: Promise<null> | null = null;

    // 安全にカメラとスキャナを停止・解放する関数
    const stopAndReleaseCamera = async (scanner: Html5Qrcode | null) => {
      if (!scanner || isStopping) return;
      isStopping = true;

      try {
        // 起動処理中であれば、起動が完了または失敗するまで待機してから停止する
        if (startPromise) {
          try {
            await startPromise;
          } catch {
            // 起動時のエラーは無視
          }
        }

        // スキャンが進行中であれば停止
        if (scanner.isScanning) {
          try {
            await scanner.stop();
          } catch (e: unknown) {
            // removeChildエラーや既に停止している場合のエラーを安全にハンドリング
            const msg = e instanceof Error ? e.message : String(e);
            if (!msg.includes("removeChild") && !msg.includes("not running")) {
              console.warn("Scanner stop warning:", e);
            }
          }
        }

        // カメラのMediaStreamトラックを明示的に停止
        const container = document.getElementById("qr-reader");
        if (container) {
          const videoElements = container.getElementsByTagName("video");
          for (let i = 0; i < videoElements.length; i++) {
            const v = videoElements[i];
            if (v.srcObject && "getTracks" in (v.srcObject as MediaStream)) {
              (v.srcObject as MediaStream).getTracks().forEach((track) => {
                try {
                  track.stop();
                } catch {
                  // 無視
                }
              });
              v.srcObject = null;
            }
          }

          // DOM内にコンテナが存在し、かつスキャン停止している場合のみクリア
          if (document.body.contains(container) && !scanner.isScanning) {
            try {
              scanner.clear();
            } catch {
              // clearのエラーは安全に握りつぶす
            }
          }
        }
      } catch (e) {
        console.warn("Camera release warning:", e);
      } finally {
        isStopping = false;
      }
    };

    const startScanner = async () => {
      setIsStarting(true);
      setCameraError(null);

      // DOMコンテナが存在するか確認
      const container = document.getElementById("qr-reader");
      if (!container || isCancelled) return;

      try {
        const scanner = new Html5Qrcode("qr-reader");
        localScanner = scanner;
        scannerRef.current = scanner;

        const config = {
          fps: 10,
          qrbox: { width: 220, height: 220 },
        };

        const handleSuccess = async (decodedText: string) => {
          if (isCancelled) return;
          isCancelled = true; // 重複スキャン防止

          await stopAndReleaseCamera(localScanner);
          onScanSuccess(decodedText);
        };

        const handleError = () => {
          // 各フレームごとのQRコード未検出は正常動作のため無視
        };

        // 内カメラ（facingMode: "user"）を直接指定して起動
        // （Html5Qrcode.getCameras() による全カメラ列挙を避け、フロントカメラを確実に要求・起動する）
        try {
          const p = scanner.start(
            { facingMode: "user" },
            config,
            handleSuccess,
            handleError
          );
          startPromise = p;
          await p;
        } catch {
          if (isCancelled) {
            await stopAndReleaseCamera(scanner);
            return;
          }
          // facingMode: "user" で起動できない場合のフォールバック
          const fallbackCameras = await Html5Qrcode.getCameras().catch(() => []);
          if (isCancelled) {
            await stopAndReleaseCamera(scanner);
            return;
          }
          if (!fallbackCameras || fallbackCameras.length === 0) {
            throw new Error("No cameras found");
          }
          const p = scanner.start(
            fallbackCameras[0].id,
            config,
            handleSuccess,
            handleError
          );
          startPromise = p;
          await p;
        }

        if (isCancelled) {
          await stopAndReleaseCamera(scanner);
        } else {
          setIsStarting(false);
        }
      } catch (err: unknown) {
        if (!isCancelled) {
          setIsStarting(false);
          const msg =
            "カメラの起動に失敗しました。カメラの利用権限を許可するか、下のボタンからパスワード入力を行ってください。";
          setCameraError(msg);
          onError?.(msg);
          console.warn("Scanner start error:", err);
        }
      }
    };

    startScanner();

    return () => {
      isCancelled = true;
      if (localScanner) {
        stopAndReleaseCamera(localScanner);
        scannerRef.current = null;
      }
    };
  }, [onScanSuccess, onError]);

  return (
    <div style={{ width: "100%", display: "flex", flexDirection: "column", alignItems: "center" }}>
      <div
        id="qr-reader"
        style={{
          width: "100%",
          maxWidth: "320px",
          minHeight: "260px",
          borderRadius: "8px",
          overflow: "hidden",
          backgroundColor: "#000",
          position: "relative",
        }}
      />

      {isStarting && !cameraError && (
        <p style={{ color: "#aaa", fontSize: "13px", marginTop: "12px" }}>
          カメラを起動しています...
        </p>
      )}

      {cameraError && (
        <p style={{ color: "#ff6b6b", fontSize: "13px", marginTop: "12px", textAlign: "center" }}>
          {cameraError}
        </p>
      )}
    </div>
  );
}

#!/usr/bin/env python3
"""
監視カメラ映像の顔モザイク処理（YouTube配信専用版 - GStreamer HWエンコード）

NVIDIA Jetson向けに最適化された顔モザイク処理実装（YouTube配信専用版）。
- YOLOv8による高精度人物検出
- TensorRT推論エンジン（自動変換・FP16精度）
- 通常のRTSPデコード（GStreamer不使用）
- ハードウェアエンコード（GStreamer + nvv4l2h264enc）
- プレビューウィンドウなし（配信専用）

技術ブログ用のリファレンス実装です。

使用方法:
    python face-mosaic-yolo-jetson-gst.py <rtsp_url> <stream_key> [options]

例:
    python face-mosaic-yolo-jetson-gst.py "rtsp://admin:password@192.168.1.100:554/stream" "xxxx-xxxx-xxxx-xxxx"
    python face-mosaic-yolo-jetson-gst.py "rtsp://camera/stream" "your-stream-key" --model yolov8s.pt --confidence 0.6

機能:
    - 初回実行時に自動的にTensorRTエンジン（.engine）を生成
    - 2回目以降は高速なTensorRTエンジンを使用
    - 通常のcv2.VideoCapture()でRTSPストリームをデコード
    - GStreamer (nvv4l2h264enc) によるハードウェアエンコード（CPU負荷削減）
    - プレビューウィンドウなし（リソース節約、YouTube配信に最適）
"""

import cv2
import numpy as np
import subprocess
import sys
import argparse
import threading
from time import perf_counter, sleep, time
from collections import deque

try:
    from ultralytics import YOLO
except ImportError:
    print("エラー: ultralyticsパッケージがインストールされていません")
    print("以下のコマンドでインストールしてください:")
    print("  pip install ultralytics")
    sys.exit(1)

# GStreamer版では log_ffmpeg_output は不要


class ThreadedVideoCapture:
    """
    RTSPストリームの読み込みを別スレッドで行い、
    常に最新のフレームのみを保持するクラス
    """
    def __init__(self, src, max_queue_size=1):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max_queue_size)
        
        self.q = deque(maxlen=max_queue_size)
        self.status = "stopped"
        self.thread = threading.Thread(target=self._update, daemon=True)

    def _update(self):
        print("[ThreadedVideoCapture] 読み取りスレッドを開始")
        while self.status == "running":
            ret, frame = self.cap.read()
            if not ret:
                print("[ThreadedVideoCapture] フレーム取得失敗。再接続試行...")
                self.cap.release()
                sleep(1)
                
                # 保存したURLで再接続
                self.cap = cv2.VideoCapture(self.src)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not self.cap.isOpened():
                    print("[ThreadedVideoCapture] 再接続失敗。1秒後にリトライ...")
                    sleep(1)
                
                continue
            
            self.q.append(frame)
        
        print("[ThreadedVideoCapture] 読み取りスレッドを停止")
        self.cap.release()

    def start(self):
        if self.status == "stopped":
            self.status = "running"
            self.thread.start()
        return self

    def read(self):
        try:
            return self.q.pop()
        except IndexError:
            return None

    def stop(self):
        self.status = "stopped"
        if self.thread.is_alive():
            self.thread.join(timeout=2)

def apply_mosaic(image, x, y, w, h, ratio=0.05):
    """
    指定された領域にモザイク処理を適用
    """
    # 境界チェック
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return image
    
    face_img = image[y:y+h, x:x+w]
    
    if face_img.size == 0:
        return image
    
    # 縮小してから拡大
    small = cv2.resize(face_img, None, fx=ratio, fy=ratio, 
                       interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h), 
                        interpolation=cv2.INTER_NEAREST)
    
    image[y:y+h, x:x+w] = mosaic
    
    return image

def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description='監視カメラ映像の顔モザイク処理（YouTube配信専用版 - GStreamer HWエンコード）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  %(prog)s "rtsp://admin:password@192.168.1.100:554/stream" "xxxx-xxxx-xxxx-xxxx"
  %(prog)s "rtsp://camera/stream" "your-stream-key" --model yolov8s.pt --confidence 0.6
  %(prog)s "rtsp://camera/stream" "your-stream-key" --width 1920 --height 1080 --fps 30

配信先: rtmp://a.rtmp.youtube.com/live2 (YouTube Live固定)
エンコーダー: GStreamer (nvv4l2h264enc)
"""
    )
    
    parser.add_argument('rtsp_url', 
                        help='監視カメラのRTSPストリームURL')
    parser.add_argument('stream_key',
                        help='YouTubeライブストリーミングキー')
    parser.add_argument('--width', '-W',
                        type=int,
                        default=1280,
                        help='出力映像の幅 (デフォルト: 1280)')
    parser.add_argument('--height', '-H',
                        type=int,
                        default=720,
                        help='出力映像の高さ (デフォルト: 720)')
    parser.add_argument('--fps', '-f',
                        type=int,
                        default=30,
                        help='フレームレート (デフォルト: 30)')
    # ビットレートを引数に追加（GStreamer設定用）
    parser.add_argument('--bitrate', '-b',
                        type=int,
                        default=2500,
                        help='配信ビットレート (kbps) (デフォルト: 2500)')
    parser.add_argument('--model', '-m',
                        default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                        help='YOLOv8モデル (デフォルト: yolov8n.pt)')
    parser.add_argument('--confidence', '-c',
                        type=float,
                        default=0.5,
                        help='検出信頼度閾値 0.0-1.0 (デフォルト: 0.5)')
    parser.add_argument('--head-ratio', '-r',
                        type=float,
                        default=0.25,
                        help='頭部領域の割合 0.1-0.5 (デフォルト: 0.25)')
    parser.add_argument('--no-tensorrt',
                        action='store_true',
                        help='TensorRT変換をスキップしてPyTorchモデルを使用')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    youtube_url = f"rtmp://a.rtmp.youtube.com/live2/{args.stream_key}"
    
    print("=" * 70)
    print("監視カメラ映像の顔モザイク処理（YouTube配信専用版 - GStreamer HWエンコード）")
    print("=" * 70)
    print(f"入力: {args.rtsp_url}")
    print(f"出力: rtmp://a.rtmp.youtube.com/live2/****")
    print(f"解像度: {args.width}x{args.height} @ {args.fps}fps")
    print(f"ビットレート: {args.bitrate} kbps")
    print(f"モデル: {args.model}")
    print(f"検出パラメータ: confidence={args.confidence}, head_ratio={args.head_ratio}")
    print("=" * 70)
    
    # --- YOLOv8モデルの読み込み（TensorRT最適化対応） ---
    # (変更なし)
    import os
    import torch
    
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("警告: CUDAが利用できません")
        print("TensorRT変換をスキップし、PyTorchモデルを使用します")
    
    device_name = torch.cuda.get_device_name(0).replace(' ', '_') if cuda_available else 'cpu'
    base_name = args.model.replace('.pt', '')
    engine_file = f"{base_name}_{device_name}.engine"
    
    model = None
    model_type = None
    
    if os.path.exists(engine_file):
        print(f"TensorRTエンジンファイル（{engine_file}）が見つかりました")
        print("TensorRTエンジンを読み込んでいます...")
        try:
            model = YOLO(engine_file, task='detect')
            model_type = 'TensorRT'
            print("TensorRTエンジンの読み込みが完了しました")
        except Exception as e:
            print(f"警告: TensorRTエンジンの読み込みに失敗しました: {e}")
            print("PyTorchモデルを使用します")
    
    if model is None:
        print(f"PyTorchモデル（{args.model}）を読み込んでいます...")
        try:
            model = YOLO(args.model)
            print("PyTorchモデルの読み込みが完了しました")
            
            if not cuda_available:
                print("\nCUDAが利用できないため、TensorRT変換をスキップします")
                model_type = 'PyTorch (CPU)'
            elif args.no_tensorrt:
                print("\n--no-tensorrtフラグが指定されたため、TensorRT変換をスキップします")
                model_type = 'PyTorch'
            else:
                print("\nTensorRTエンジンへの変換を試みています...")
                print("（初回のみ時間がかかります。数分お待ちください）")
                try:
                    model.export(format='engine', half=True, device=0)
                    default_engine = args.model.replace('.pt', '.engine')
                    
                    if os.path.exists(default_engine) and default_engine != engine_file:
                        print(f"\nエンジンファイルを {default_engine} から {engine_file} にリネームしています...")
                        os.rename(default_engine, engine_file)
                    
                    print(f"\n変換が完了しました。TensorRTエンジン（{engine_file}）を読み込んでいます...")
                    model = YOLO(engine_file, task='detect')
                    model_type = 'TensorRT'
                    print("TensorRTエンジンでの実行準備が完了しました")
                    
                except Exception as e:
                    print(f"\n警告: TensorRTへの変換に失敗しました: {e}")
                    print("PyTorchモデルをそのまま使用します")
                    model_type = 'PyTorch'
        except Exception as e:
            print(f"エラー: モデルの読み込みに失敗しました: {e}")
            sys.exit(1)
    
    print(f"使用するモデル形式: {model_type}")
    print("=" * 70)
    
    # RTSPストリームを開く（スレッド化）
    print("RTSPストリームに接続しています（スレッド読み取り）...")
    cap = ThreadedVideoCapture(args.rtsp_url).start()
    sleep(2)
    print("接続成功")
    
    # --- 修正箇所: FFmpeg から GStreamer (cv2.VideoWriter) へ ---
    
    print("ハードウェアエンコーダー（nvv4l2h264enc）を使用します")
    
    # GStreamerパイプラインの構築
    # appsrc (OpenCVからBGR) -> videoconvert (BGR->I420/NV12)
    # -> nvv4l2h264enc (HWエンコード) -> h264parse (H.264解析)
    # -> flvmux (FLVコンテナ化) -> rtmpsink (YouTubeへ送信)
    
    bitrate_bps = args.bitrate * 1000 # kbps を bps に変換
    
    gst_pipeline = (
        f"appsrc ! "
        f"video/x-raw,format=BGR ! "
        f"nvvideoconvert ! "
        f"video/x-raw(memory:NVMM),format=NV12 ! " # NV12フォーマットを使用
        f"nvv4l2h264enc "
        f"bitrate={bitrate_bps} "
        f"preset-level=4 " # 4 = UltraFastPreset (低遅延)
        f"insert-sps-pps=true ! " # SPS/PPSをストリームに含める
        f"h264parse ! "
        f"flvmux streamable=true ! " # RTMP用のFLVコンテナ
        f"rtmpsink location='{youtube_url}'"
    )

    print("\nGStreamerパイプラインを起動しています...")
    print(f"パイプライン: appsrc ! ... ! rtmpsink location=...")

    # GStreamerバックエンドを指定してVideoWriterを作成
    out = cv2.VideoWriter(
        gst_pipeline,
        cv2.CAP_GSTREAMER,
        0, # CAP_GSTREAMERの場合、fourccは0でOK
        args.fps,
        (args.width, args.height)
    )

    if not out.isOpened():
        print("\n" + "="*50)
        print("エラー: GStreamer (cv2.VideoWriter) の起動に失敗しました")
        print("="*50)
        print("以下の点を確認してください:")
        print(" 1. GStreamerの依存関係がインストールされているか")
        print(" 　（例: gstreamer1.0-plugins-good, gstreamer1.0-plugins-bad, gstreamer1.0-plugins-ugly）")
        print(" 2. JetsonのマルチメディアAPIがインストールされているか")
        print(" 　（例: gstreamer1.0-libav, gstreamer1.0-tools, libgstreamer-plugins-base1.0-dev）")
        print(" 3. OpenCVがGStreamerサポート付きでビルドされているか")
        print(" 4. 'nvv4l2h264enc' が利用可能か（ターミナルで `gst-inspect-1.0 nvv4l2h264enc` を実行）")
        cap.stop()
        sys.exit(1)

    # (FFmpegサブプロセスとログスレッドは不要)
    
    print("\nYouTube Liveへのストリーミングを開始しました")
    print("YouTube Studio (https://studio.youtube.com) で配信状況を確認してください")
    print("※配信が開始されるまで数秒〜数十秒かかる場合があります\n")
    
    # --- 修正ここまで ---

    frame_count = 0
    total_detections = 0
    start_time = time()
    skipped_frames = 0
    
    try:
        print("処理を開始します（Ctrl+Cで終了）")
        print("GStreamerパイプラインで配信中...\n")
        
        while True:
            frame = cap.read()
            
            if frame is None:
                sleep(0.01)
                continue
            
            # フレームを指定された解像度にリサイズ
            frame = cv2.resize(frame, (args.width, args.height))
            
            # YOLOv8で人物検出
            results = model(
                frame, 
                classes=[0],
                conf=args.confidence,
                verbose=False
            )
            
            detected_heads = []
            
            # 検出結果から頭部領域を計算
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    person_w = x2 - x1
                    person_h = y2 - y1
                    
                    # 小さすぎる検出や不自然なアスペクト比を除外
                    if person_w < 30 or person_h < 50:
                        continue
                    aspect_ratio = person_w / person_h if person_h > 0 else 0
                    if aspect_ratio < 0.2 or aspect_ratio > 3.0:
                        continue
                    
                    # 頭部領域を推定
                    head_h = int(person_h * args.head_ratio)
                    head_x = max(0, x1 - int(person_w * 0.1))
                    head_y = max(0, y1 - int(head_h * 0.1))
                    head_w = min(args.width - head_x, person_w + int(person_w * 0.2))
                    head_h = min(args.height - head_y, head_h + int(head_h * 0.2))
                    
                    detected_heads.append((head_x, head_y, head_w, head_h))
            
            # モザイクを適用
            for (x, y, w, h) in detected_heads:
                frame = apply_mosaic(frame, x, y, w, h, ratio=0.05)
            
            total_detections += len(detected_heads)
            
            # --- 修正箇所: FFmpegへの書き込み -> GStreamer (VideoWriter) への書き込み ---
            try:
                out.write(frame)
            except Exception as e:
                # GStreamerパイプラインが切断された場合など
                print(f"\n警告: GStreamerへのフレーム送信中にエラーが発生しました: {e}")
                print("パイプラインが切断された（ストリームキーが不正、ネットワーク切断など）可能性があります。")
                break
            # --- 修正ここまで ---
            
            frame_count += 1
            # 100フレームごとに進捗状況を表示
            if frame_count % 100 == 0:
                elapsed_time = time() - start_time
                actual_fps = frame_count / elapsed_time
                avg_detections = total_detections / frame_count
                
                print(f"処理済み: {frame_count}フレーム | "
                      f"検出数: {len(detected_heads)} | "
                      f"平均検出: {avg_detections:.2f} | "
                      f"実FPS: {actual_fps:.1f} (目標: {args.fps})")
                      
    except KeyboardInterrupt:
        print("\n\nキーボード割り込みを検出しました。終了します...")
    
    finally:
        # クリーンアップ
        print("リソースを解放しています...")
        cap.stop()
        
        # --- 修正箇所: FFmpegの終了処理 -> VideoWriterの解放 ---
        if out and out.isOpened():
            print("GStreamerパイプラインを解放しています...")
            out.release()
        # --- 修正ここまで ---
        
        print("完了しました")

if __name__ == "__main__":
    main()
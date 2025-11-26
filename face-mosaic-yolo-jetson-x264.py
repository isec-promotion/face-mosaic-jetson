#!/usr/bin/env python3
"""
監視カメラ映像の顔モザイク処理（YouTube配信専用版 - プレビューなし）

NVIDIA Jetson向けに最適化された顔モザイク処理実装（YouTube配信専用版）。
- YOLOv8による高精度人物検出
- TensorRT推論エンジン（自動変換・FP16精度）
- 通常のRTSPデコード（GStreamer不使用）
- ソフトウェアエンコード（libx264）
- プレビューウィンドウなし（配信専用）
- フレームレート補間機能（ソースFPSが低い場合に自動的に目標FPSに水増し）

技術ブログ用のリファレンス実装です。

使用方法:
    python face-mosaic-yolo-jetson-x264.py <rtsp_url> <stream_key> [options]

例:
    python face-mosaic-yolo-jetson-x264.py "rtsp://admin:password@192.168.1.100:554/stream" "xxxx-xxxx-xxxx-xxxx"
    python face-mosaic-yolo-jetson-x264.py "rtsp://camera/stream" "your-stream-key" --model yolov8s.pt --confidence 0.6
    python face-mosaic-yolo-jetson-x264.py "rtsp://camera/stream" "your-stream-key" --interpolate-fps 30 --interpolation-method linear

機能:
    - 初回実行時に自動的にTensorRTエンジン（.engine）を生成
    - 2回目以降は高速なTensorRTエンジンを使用
    - 通常のcv2.VideoCapture()でRTSPストリームをデコード（GStreamer不要）
    - ソフトウェアエンコード（libx264）
    - プレビューウィンドウなし（リソース節約、YouTube配信に最適）
    - フレームレート補間：ソースFPSが目標FPSより低い場合、自動的に補間して目標FPSに水増し
      * duplicate: フレーム複製方式（高速、品質は低め）
      * linear: 線形補間方式（中程度の速度、品質は高め）
"""

import cv2
import numpy as np
import subprocess
import sys
import argparse
import threading
from time import perf_counter, sleep, time
from collections import deque
import math

try:
    from ultralytics import YOLO
except ImportError:
    print("エラー: ultralyticsパッケージがインストールされていません")
    sys.exit(1)

def log_ffmpeg_output(process):
    while True:
        line = process.stderr.readline()
        if not line:
            break
        line = line.decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        print(f"[FFmpeg] {line}")

class ThreadedVideoCapture:
    """
    RTSPストリームの読み込みを別スレッドで行い、
    常に最新のフレームのみを保持するクラス
    """
    def __init__(self, src, max_queue_size=1):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max_queue_size)
        
        # --- 【追加】FPSを取得して保持 ---
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # -----------------------------

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
    画像の指定領域にモザイクを適用する
    
    Args:
        image: 入力画像（BGR形式）
        x: モザイク領域の左上X座標
        y: モザイク領域の左上Y座標
        w: モザイク領域の幅
        h: モザイク領域の高さ
        ratio: モザイクの縮小率（デフォルト: 0.05）
    
    Returns:
        モザイク適用後の画像
    """
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0: return image
    face_img = image[y:y+h, x:x+w]
    if face_img.size == 0: return image
    
    small = cv2.resize(face_img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = mosaic
    return image

class FrameInterpolator:
    """
    フレームレート補間クラス
    ソースFPSが目標FPSより低い場合、フレームを補間して目標FPSに水増しする
    """
    def __init__(self, source_fps, target_fps, method='linear'):
        """
        フレーム補間器を初期化
        
        Args:
            source_fps: ソース映像のFPS
            target_fps: 目標FPS
            method: 補間方法 ('duplicate': フレーム複製, 'linear': 線形補間)
        """
        self.sourceFps = source_fps
        self.targetFps = target_fps
        self.method = method
        
        # 補間が必要かどうかを判定
        self.needsInterpolation = source_fps > 0 and source_fps < target_fps
        
        if self.needsInterpolation:
            # 1つのソースフレームから生成するフレーム数
            self.framesPerSource = target_fps / source_fps
            # 前のフレームを保持（線形補間用）
            self.prevFrame = None
            # 補間フレームのカウンタ
            self.interpolationCounter = 0.0
            
            print(f"[FrameInterpolator] 補間を有効化: {source_fps}fps -> {target_fps}fps")
            print(f"[FrameInterpolator] 補間方法: {method}")
            print(f"[FrameInterpolator] 1ソースフレームあたり {self.framesPerSource:.2f} フレームを生成")
        else:
            print(f"[FrameInterpolator] 補間不要: ソースFPS({source_fps}) >= 目標FPS({target_fps})")
    
    def interpolate(self, currentFrame):
        """
        現在のフレームから補間フレームを生成
        
        Args:
            currentFrame: 現在のフレーム（BGR形式、numpy配列）
        
        Returns:
            補間されたフレームのリスト（空の場合は補間不要）
        """
        if not self.needsInterpolation or currentFrame is None:
            return [currentFrame] if currentFrame is not None else []
        
        interpolatedFrames = []
        
        if self.method == 'duplicate':
            # フレーム複製方式：各フレームを複数回送信
            numFrames = int(round(self.framesPerSource))
            for _ in range(numFrames):
                interpolatedFrames.append(currentFrame.copy())
        
        elif self.method == 'linear':
            # 線形補間方式：前のフレームと現在のフレームをブレンド
            if self.prevFrame is None:
                # 最初のフレームはそのまま複数回送信（前のフレームがないため）
                numFrames = int(round(self.framesPerSource))
                for _ in range(numFrames):
                    interpolatedFrames.append(currentFrame.copy())
                # 前のフレームとして保存
                self.prevFrame = currentFrame.copy()
            else:
                # 前のフレームと現在のフレームの間を補間
                # 例: 10fps -> 30fps の場合、1ソースフレームあたり3フレーム生成
                # 最初のフレーム: 前のフレーム100%、現在のフレーム0%（前回の最後のフレームと同じ）
                # 中間のフレーム: 前のフレームと現在のフレームをブレンド
                # 最後のフレーム: 前のフレーム0%、現在のフレーム100%
                numFrames = int(round(self.framesPerSource))
                
                for i in range(numFrames):
                    if i == 0:
                        # 最初のフレームは前のフレーム（前回の最後のフレームと同じ）
                        interpolatedFrames.append(self.prevFrame.copy())
                    elif i == numFrames - 1:
                        # 最後のフレームは現在のフレーム
                        interpolatedFrames.append(currentFrame.copy())
                    else:
                        # 中間フレーム: 補間比率を計算（0.0 = 前のフレーム、1.0 = 現在のフレーム）
                        alpha = i / (numFrames - 1)
                        
                        # フレームをブレンド
                        blended = cv2.addWeighted(
                            self.prevFrame, 
                            1.0 - alpha, 
                            currentFrame, 
                            alpha, 
                            0
                        )
                        interpolatedFrames.append(blended)
                
                # 前のフレームを更新
                self.prevFrame = currentFrame.copy()
        
        return interpolatedFrames

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='監視カメラ映像の顔モザイク処理（Auto FPS版）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('rtsp_url', help='監視カメラのRTSPストリームURL')
    parser.add_argument('stream_key', help='YouTubeライブストリーミングキー')
    parser.add_argument('--width', '-W', type=int, default=1280, help='出力映像の幅')
    parser.add_argument('--height', '-H', type=int, default=720, help='出力映像の高さ')
    parser.add_argument('--fps', '-f', type=int, default=30, help='FPS自動取得失敗時のデフォルトFPS')
    parser.add_argument('--model', '-m', default='yolov8n.pt', choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'])
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='検出信頼度閾値')
    parser.add_argument('--head-ratio', '-r', type=float, default=0.25, help='頭部領域の割合')
    parser.add_argument('--no-tensorrt', action='store_true', help='TensorRT変換をスキップ')
    parser.add_argument('--interpolate-fps', type=int, default=30, help='フレーム補間の目標FPS（ソースFPSがこの値より低い場合に補間）')
    parser.add_argument('--interpolation-method', choices=['duplicate', 'linear'], default='linear', help='フレーム補間方法（duplicate: 複製, linear: 線形補間）')
    return parser.parse_args()

def main():
    args = parse_arguments()
    youtube_url = f"rtmp://a.rtmp.youtube.com/live2/{args.stream_key}"
    
    print("=" * 70)
    print("監視カメラ映像の顔モザイク処理（YouTube配信専用版 - Auto FPS）")
    print("=" * 70)
    
    # モデル読み込み処理（TensorRT関連）は省略せずそのまま使用
    import os
    import torch
    
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("警告: CUDAが利用できません。TensorRT変換をスキップします。")
    
    device_name = torch.cuda.get_device_name(0).replace(' ', '_') if cuda_available else 'cpu'
    base_name = args.model.replace('.pt', '')
    engine_file = f"{base_name}_{device_name}.engine"
    
    model = None
    model_type = None
    
    # 1. TensorRTエンジンの確認
    if os.path.exists(engine_file):
        print(f"TensorRTエンジン（{engine_file}）を読み込んでいます...")
        try:
            model = YOLO(engine_file, task='detect')
            model_type = 'TensorRT'
        except Exception:
            pass
    
    # 2. PyTorchモデルの読み込みと変換
    if model is None:
        print(f"PyTorchモデル（{args.model}）を読み込んでいます...")
        try:
            model = YOLO(args.model)
            if not cuda_available or args.no_tensorrt:
                model_type = 'PyTorch'
            else:
                print("\nTensorRTエンジンへの変換を試みています...")
                try:
                    model.export(format='engine', half=True, device=0)
                    default_engine = args.model.replace('.pt', '.engine')
                    if os.path.exists(default_engine) and default_engine != engine_file:
                        os.rename(default_engine, engine_file)
                    model = YOLO(engine_file, task='detect')
                    model_type = 'TensorRT'
                except Exception as e:
                    print(f"変換失敗: {e} - PyTorchモデルを使用します")
                    model_type = 'PyTorch'
        except Exception as e:
            print(f"エラー: {e}")
            sys.exit(1)
            
    print(f"使用モデル: {model_type}")
    print("=" * 70)
    
    # --- 【修正】ビデオキャプチャとFPS判定ロジック ---
    print("RTSPストリームに接続しています...")
    
    # まずインスタンスを作成（まだスレッドは開始しない）
    cap = ThreadedVideoCapture(args.rtsp_url)
    
    # FPS情報の取得
    source_fps = cap.fps
    print(f"検出されたソースFPS: {source_fps}")
    
    # フレーム補間の判定
    interpolateTargetFps = args.interpolate_fps
    useInterpolation = source_fps > 0 and source_fps < interpolateTargetFps
    
    if useInterpolation:
        # フレーム補間を使用する場合、目標FPSは補間目標FPS
        target_fps = interpolateTargetFps
        print(f"-> 適用FPS: {target_fps} (フレーム補間: {source_fps}fps -> {target_fps}fps)")
    elif source_fps > 0 and source_fps < 120:
        # 補間不要で、ソースFPSが有効な場合
        target_fps = source_fps
        # 整数に近い場合は丸める (29.97 -> 30, 14.9 -> 15)
        if abs(target_fps - round(target_fps)) < 0.1:
            target_fps = round(target_fps)
        print(f"-> 適用FPS: {target_fps} (ソース同期)")
    else:
        # ソースFPSが取得できない、または異常な値の場合
        target_fps = args.fps
        print(f"-> 適用FPS: {target_fps} (デフォルト値)")
    
    # フレーム補間器の初期化
    frameInterpolator = None
    if useInterpolation:
        frameInterpolator = FrameInterpolator(
            source_fps=source_fps,
            target_fps=target_fps,
            method=args.interpolation_method
        )
    
    # スレッド開始
    cap.start()
    sleep(2) # バッファ充填待ち
    # -----------------------------------------------
    
    print(f"入力: {args.rtsp_url}")
    print(f"解像度: {args.width}x{args.height} @ {target_fps}fps")

    # FFmpeg設定
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{args.width}x{args.height}',
        '-r', str(target_fps),     # 【修正】固定args.fpsではなく判定したtarget_fpsを使用
        '-i', '-',
        '-f', 'lavfi',
        '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
        '-fflags', '+genpts',
        '-vsync', 'cfr',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-b:v', '2500k',
        '-maxrate', '2500k',
        '-bufsize', '5000k',
        '-bf', '0',
        '-sc_threshold', '0',
        '-g', str(int(target_fps) * 2), # 【修正】GOP長もFPSに合わせて調整（2秒間隔）
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-ar', '44100',
        '-flvflags', 'no_duration_filesize',
        '-f', 'flv',
        youtube_url,
    ]
    
    try:
        print("\nFFmpegを起動しています...")
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        log_thread = threading.Thread(target=log_ffmpeg_output, args=(ffmpeg_process,), daemon=True)
        log_thread.start()
        sleep(2)
        
        if ffmpeg_process.poll() is not None:
            print("\n警告: FFmpegプロセスが終了しました。")
        else:
            print("\nYouTube Liveへのストリーミングを開始しました")
        
    except FileNotFoundError:
        print("エラー: FFmpegが見つかりません")
        sys.exit(1)
    
    frame_count = 0
    total_detections = 0
    start_time = time()
    
    try:
        print("処理を開始します（Ctrl+Cで終了）\n")
        
        # フレーム送信間隔の計算（目標FPSに合わせる）
        frameInterval = 1.0 / target_fps if target_fps > 0 else 0.033
        lastFrameTime = perf_counter()
        
        while True:
            frame = cap.read()
            
            if frame is None:
                sleep(0.01)
                continue
            
            frame = cv2.resize(frame, (args.width, args.height))
            
            # YOLO検出
            results = model(frame, classes=[0], conf=args.confidence, verbose=False)
            
            detected_heads = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    person_w, person_h = x2 - x1, y2 - y1
                    
                    if person_w < 30 or person_h < 50: continue
                    aspect = person_w / person_h if person_h > 0 else 0
                    if aspect < 0.2 or aspect > 3.0: continue
                    
                    head_h = int(person_h * args.head_ratio)
                    head_x = max(0, x1 - int(person_w * 0.1))
                    head_y = max(0, y1 - int(head_h * 0.1))
                    head_w = min(args.width - head_x, person_w + int(person_w * 0.2))
                    head_h = min(args.height - head_y, head_h + int(head_h * 0.2))
                    
                    detected_heads.append((head_x, head_y, head_w, head_h))
            
            # モザイク
            for (x, y, w, h) in detected_heads:
                frame = apply_mosaic(frame, x, y, w, h)
            
            total_detections += len(detected_heads)
            
            # フレーム補間の適用
            if frameInterpolator is not None:
                # 補間フレームを生成
                interpolatedFrames = frameInterpolator.interpolate(frame)
                
                # 補間フレームを順次送信
                for interpFrame in interpolatedFrames:
                    try:
                        ffmpeg_process.stdin.write(interpFrame.tobytes())
                        frame_count += 1
                        
                        # 目標FPSに合わせて送信間隔を調整
                        currentTime = perf_counter()
                        elapsed = currentTime - lastFrameTime
                        if elapsed < frameInterval:
                            sleep(frameInterval - elapsed)
                        lastFrameTime = perf_counter()
                    except (BrokenPipeError, IOError):
                        print("警告: FFmpegパイプエラー")
                        raise
            else:
                # 補間なし：フレームをそのまま送信
                try:
                    ffmpeg_process.stdin.write(frame.tobytes())
                    frame_count += 1
                    
                    # 目標FPSに合わせて送信間隔を調整
                    currentTime = perf_counter()
                    elapsed = currentTime - lastFrameTime
                    if elapsed < frameInterval:
                        sleep(frameInterval - elapsed)
                    lastFrameTime = perf_counter()
                except (BrokenPipeError, IOError):
                    print("警告: FFmpegパイプエラー")
                    break
            
            if frame_count % 100 == 0:
                elapsed = time() - start_time
                actual_fps = frame_count / elapsed
                fps_diff = actual_fps - target_fps
                print(f"FPS: {actual_fps:.1f} (目標: {target_fps}, 差: {fps_diff:+.1f}) | 検出数: {len(detected_heads)}")
                
    except KeyboardInterrupt:
        print("\n終了します...")
    finally:
        cap.stop()
        if ffmpeg_process:
            try:
                ffmpeg_process.stdin.close()
                ffmpeg_process.terminate()
                ffmpeg_process.wait(timeout=3)
            except:
                pass
        print("完了")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
監視カメラ映像の顔モザイク処理（YouTube配信専用版 - プレビューなし）

NVIDIA Jetson向けに最適化された顔モザイク処理実装（YouTube配信専用版）。
- YOLOv8による高精度人物検出
- TensorRT推論エンジン（自動変換・FP16精度）
- 通常のRTSPデコード（GStreamer不使用）
- ハードウェアエンコード（NVENC）
- プレビューウィンドウなし（配信専用）

技術ブログ用のリファレンス実装です。

使用方法:
    python face-mosaic-yolo-jetson-ffmpeg.py <rtsp_url> <stream_key> [options]

例:
    python face-mosaic-yolo-jetson-ffmpeg.py "rtsp://admin:password@192.168.1.100:554/stream" "xxxx-xxxx-xxxx-xxxx"
    python face-mosaic-yolo-jetson-ffmpeg.py "rtsp://camera/stream" "your-stream-key" --model yolov8s.pt --confidence 0.6

機能:
    - 初回実行時に自動的にTensorRTエンジン（.engine）を生成
    - 2回目以降は高速なTensorRTエンジンを使用
    - 通常のcv2.VideoCapture()でRTSPストリームをデコード（GStreamer不要）
    - NVENCによるハードウェアエンコード（CPU負荷削減）
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

def log_ffmpeg_output(process):
    while True:
        line = process.stderr.readline()
        if not line:
            break
        line = line.decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        # ★ 全部表示（最初はこれが確実）
        print(f"[FFmpeg] {line}")

        # もし絞るなら下記のように増やす
        # if any(k in line.lower() for k in ['error','warning','failed','connection',
        #                                    'unrecognized option','unknown encoder',
        #                                    'invalid argument','option not found']):
        #     print(f"[FFmpeg] {line}")


class ThreadedVideoCapture:
    """
    RTSPストリームの読み込みを別スレッドで行い、
    常に最新のフレームのみを保持するクラス
    """
    def __init__(self, src, max_queue_size=1):
        self.src = src  # <-- 修正: RTSPのURLを保存する
        self.cap = cv2.VideoCapture(self.src) # <-- 修正: self.src を使用
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max_queue_size)
        
        # dequeをサイズ1で作成し、常に最新のフレームのみ保持
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
                
                # --- 修正箇所 (再接続ロジック) ---
                # self.cap = cv2.VideoCapture(self.cap.getBackendName()) # 削除
                self.cap = cv2.VideoCapture(self.src) # 変更: 保存したURLで再接続
                # --- 修正ここまで ---
                
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # 変更: 再接続が成功したか確認する
                if not self.cap.isOpened():
                    print("[ThreadedVideoCapture] 再接続失敗。1秒後にリトライ...")
                    sleep(1)
                
                continue
            
            # dequeにフレームを追加（古いフレームは自動的に破棄）
            self.q.append(frame)
        
        print("[ThreadedVideoCapture] 読み取りスレッドを停止")
        self.cap.release()

    def start(self):
        if self.status == "stopped":
            self.status = "running"
            self.thread.start()
        return self

    def read(self):
        # キューにフレームがあれば、それを返す
        try:
            return self.q.pop()
        except IndexError:
            # キューが空の場合
            return None

    def stop(self):
        self.status = "stopped"
        if self.thread.is_alive():
            self.thread.join(timeout=2)

def apply_mosaic(image, x, y, w, h, ratio=0.05):
    """
    指定された領域にモザイク処理を適用
    
    Args:
        image: 入力画像
        x, y: モザイク領域の左上座標
        w, h: モザイク領域の幅と高さ
        ratio: モザイクの粗さ（小さいほど粗い、推奨: 0.05-0.1）
    
    Returns:
        モザイク処理後の画像
    """
    # 境界チェック
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return image
    
    # モザイク処理する領域を抽出
    face_img = image[y:y+h, x:x+w]
    
    if face_img.size == 0:
        return image
    
    # 縮小してから拡大することでモザイク効果を作成
    small = cv2.resize(face_img, None, fx=ratio, fy=ratio, 
                      interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h), 
                       interpolation=cv2.INTER_NEAREST)
    
    # 元の画像にモザイクを適用
    image[y:y+h, x:x+w] = mosaic
    
    return image

def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description='監視カメラ映像の顔モザイク処理（YouTube配信専用版 - プレビューなし）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  %(prog)s "rtsp://admin:password@192.168.1.100:554/stream" "xxxx-xxxx-xxxx-xxxx"
  %(prog)s "rtsp://camera/stream" "your-stream-key" --model yolov8s.pt --confidence 0.6
  %(prog)s "rtsp://camera/stream" "your-stream-key" --width 1920 --height 1080 --fps 30

配信先: rtmp://a.rtmp.youtube.com/live2 (YouTube Live固定)

モデルの選択:
  yolov8n.pt: Nano (最速、メモリ少)
  yolov8s.pt: Small (バランス)
  yolov8m.pt: Medium (高精度)
  yolov8l.pt: Large (最高精度)
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
    # コマンドライン引数を解析
    args = parse_arguments()
    
    # YouTube RTMPストリームURL
    youtube_url = f"rtmp://a.rtmp.youtube.com/live2/{args.stream_key}"
    
    print("=" * 70)
    print("監視カメラ映像の顔モザイク処理（YouTube配信専用版 - プレビューなし）")
    print("=" * 70)
    print(f"入力: {args.rtsp_url}")
    print(f"出力: rtmp://a.rtmp.youtube.com/live2/****")
    print(f"解像度: {args.width}x{args.height} @ {args.fps}fps")
    print(f"モデル: {args.model}")
    print(f"検出パラメータ: confidence={args.confidence}, head_ratio={args.head_ratio}")
    print("=" * 70)
    
    # YOLOv8モデルの読み込み（TensorRT最適化対応）
    import os
    import torch
    
    # CUDA利用可否の確認
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("警告: CUDAが利用できません")
        print(f"torch.cuda.is_available(): {cuda_available}")
        print("TensorRT変換をスキップし、PyTorchモデルを使用します")
        print("\nCUDA対応PyTorchのインストール方法:")
        print("  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("=" * 70)
    
    # TensorRTエンジンファイル名を生成（環境依存）
    # 異なるGPU/TensorRTバージョン間では互換性がないため、デバイス名を含める
    device_name = torch.cuda.get_device_name(0).replace(' ', '_') if cuda_available else 'cpu'
    base_name = args.model.replace('.pt', '')
    engine_file = f"{base_name}_{device_name}.engine"
    
    model = None
    model_type = None
    
    # 1. TensorRTエンジンファイルが存在するかチェック
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
    
    # 2. TensorRTエンジンが読み込めなかった場合、PyTorchモデルから変換を試みる
    if model is None:
        print(f"PyTorchモデル（{args.model}）を読み込んでいます...")
        try:
            model = YOLO(args.model)
            print("PyTorchモデルの読み込みが完了しました")
            
            # CUDAが利用できない、または--no-tensorrtフラグがある場合はTensorRT変換をスキップ
            if not cuda_available:
                print("\nCUDAが利用できないため、TensorRT変換をスキップします")
                print("PyTorchモデル（CPU）をそのまま使用します")
                model_type = 'PyTorch (CPU)'
            elif args.no_tensorrt:
                print("\n--no-tensorrtフラグが指定されたため、TensorRT変換をスキップします")
                print("PyTorchモデルをそのまま使用します")
                model_type = 'PyTorch'
            else:
                # TensorRTへの変換を試みる
                print("\nTensorRTエンジンへの変換を試みています...")
                print("（初回のみ時間がかかります。数分お待ちください）")
                print("※すぐに開始したい場合は Ctrl+C で中断し、--no-tensorrt オプションを使用してください")
                try:
                    # TensorRTへエクスポート
                    # half=True でFP16精度（Jetsonで高速）
                    model.export(format='engine', half=True, device=0)
                    
                    # エクスポート時に生成されるデフォルトのファイル名
                    default_engine = args.model.replace('.pt', '.engine')
                    
                    # デバイス名を含むファイル名にリネーム
                    if os.path.exists(default_engine) and default_engine != engine_file:
                        print(f"\nエンジンファイルを {default_engine} から {engine_file} にリネームしています...")
                        os.rename(default_engine, engine_file)
                    
                    # 変換されたエンジンファイルを再読み込み
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
            print("初回実行時はモデルのダウンロードに時間がかかる場合があります")
            sys.exit(1)
    
    print(f"使用するモデル形式: {model_type}")
    print("=" * 70)
    
    # 通常のcv2.VideoCaptureでRTSPストリームを開く（スレッド化）
    print("RTSPストリームに接続しています（スレッド読み取り）...")
    
    cap = ThreadedVideoCapture(args.rtsp_url).start()
    
    # 最初のフレームが読み込めるまで少し待つ
    sleep(2)
    
    print("接続成功")
    
    # ハードウェアエンコーダー（h264_v4l2m2m）を使用
    print("ハードウェアエンコーダー（h264_v4l2m2m）を使用します（CFR/低遅延）")
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{args.width}x{args.height}',
        '-r', str(args.fps),
        '-re',
        '-i', '-',
        '-f', 'lavfi',
        '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
        # タイムスタンプ/CFR
        '-fflags', '+genpts',
        '-vsync', 'cfr',
        
        # --- 修正箇所 (Jetson向けNVENC設定 Ver.3) ---
        '-c:v', 'libx264',
        '-preset', 'veryfast', # CPU負荷を最小限に
        '-tune', 'zerolatency', # 低遅延
        # --- 修正ここまで ---
        
        '-b:v', '2500k',
        '-maxrate', '2500k', # -b:v と同じ値にすることでCBRに近づける
        '-bufsize', '5000k', # maxrate の2倍程度
        '-bf', '0',
        '-sc_threshold', '0',
        '-g', str(args.fps * 2),
        '-pix_fmt', 'yuv420p',
        # 音声
        '-c:a', 'aac',
        '-b:a', '128k',
        '-ar', '44100',
        # RTMP FLV
        '-flvflags', 'no_duration_filesize',
        '-f', 'flv',
        youtube_url,
    ]
    
    try:
        print("\nFFmpegを起動しています...")
        print("FFmpegコマンド:")
        print(" ".join(ffmpeg_cmd[:15]) + " ... " + ffmpeg_cmd[-1])
        
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        # FFmpegのログを別スレッドで監視
        log_thread = threading.Thread(target=log_ffmpeg_output, args=(ffmpeg_process,), daemon=True)
        log_thread.start()
        
        # FFmpegが起動するまで少し待つ
        sleep(2)
        
        # プロセスが生きているか確認
        if ffmpeg_process.poll() is not None:
            print("\n警告: FFmpegプロセスが予期せず終了しました")
            print("ストリームキーが正しいか確認してください")
        else:
            print("\nYouTube Liveへのストリーミングを開始しました")
            print("YouTube Studio (https://studio.youtube.com) で配信状況を確認してください")
            print("※配信が開始されるまで数秒〜数十秒かかる場合があります\n")
        
    except FileNotFoundError:
        print("エラー: FFmpegが見つかりません")
        print("FFmpegをインストールしてPATHに追加してください")
        cap.release()
        sys.exit(1)
    
    frame_count = 0
    total_detections = 0
    start_time = time()
    skipped_frames = 0
    
    try:
        print("処理を開始します（Ctrl+Cで終了）")
        # FFmpeg側でタイミングを制御するモードであることを表示
        print("FFmpeg -reモード: FFmpeg側でタイミングを制御します\n")
        
        while True:
            frame = cap.read()
            
            if frame is None:
                # キューにフレームが溜まるまで少し待つ
                sleep(0.01)
                continue
            
            # フレームを指定された解像度にリサイズ
            frame = cv2.resize(frame, (args.width, args.height))
            
            # YOLOv8で人物検出（クラス0は人物）
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
                    # バウンディングボックスの座標を取得
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 人物の幅と高さを計算
                    person_w = x2 - x1
                    person_h = y2 - y1
                    
                    # 小さすぎる検出を除外
                    if person_w < 30 or person_h < 50:
                        continue
                    
                    # 不自然なアスペクト比の検出を除外
                    aspect_ratio = person_w / person_h if person_h > 0 else 0
                    if aspect_ratio < 0.2 or aspect_ratio > 3.0:
                        continue
                    
                    # 頭部領域を推定して計算（上下左右に少しマージンを追加）
                    head_h = int(person_h * args.head_ratio)
                    head_x = max(0, x1 - int(person_w * 0.1))
                    head_y = max(0, y1 - int(head_h * 0.1))
                    head_w = min(args.width - head_x, person_w + int(person_w * 0.2))
                    head_h = min(args.height - head_y, head_h + int(head_h * 0.2))
                    
                    detected_heads.append((head_x, head_y, head_w, head_h))
            
            # 検出したすべての頭部領域にモザイクを適用
            for (x, y, w, h) in detected_heads:
                frame = apply_mosaic(frame, x, y, w, h, ratio=0.05)
            
            # モザイク処理された人数をカウント
            total_detections += len(detected_heads)
            
            # 処理済みのフレームをFFmpegの標準入力に書き込む
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("警告: FFmpegプロセスが終了しました。ストリームキーが正しいか確認してください。")
                break
            except Exception as e:
                print(f"警告: フレームの送信中にエラーが発生しました: {e}")
                break
            
            frame_count += 1
            # 100フレームごとに進捗状況を表示
            if frame_count % 100 == 0:
                elapsed_time = time() - start_time
                actual_fps = frame_count / elapsed_time
                avg_detections = total_detections / frame_count
                target_fps = args.fps
                
                # 処理速度が目標に追いついているかどうかの差分を表示
                fps_diff = actual_fps - target_fps
                
                print(f"処理済み: {frame_count}フレーム | "
                      f"検出数: {len(detected_heads)} | "
                      f"平均検出: {avg_detections:.2f} | "
                      f"実FPS: {actual_fps:.1f} (目標: {target_fps}, 差: {fps_diff:+.1f})")
                
    except KeyboardInterrupt:
        print("\n\nキーボード割り込みを検出しました。終了します...")
    
    finally:
        # クリーンアップ
        print("リソースを解放しています...")
        cap.stop()
        
        if ffmpeg_process:
            try:
                ffmpeg_process.stdin.close()
            except:
                pass
            
            try:
                ffmpeg_process.terminate()
                try:
                    ffmpeg_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print("FFmpegプロセスを強制終了しています...")
                    ffmpeg_process.kill()
                    ffmpeg_process.wait()
            except:
                pass
        
        print("完了しました")

if __name__ == "__main__":
    main()

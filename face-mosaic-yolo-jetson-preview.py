#!/usr/bin/env python3
"""
監視カメラ映像の顔モザイク処理（Jetson最適化版 - GStreamer不使用）

NVIDIA Jetson向けに最適化された顔モザイク処理実装（GStreamer不使用版）。
- YOLOv8による高精度人物検出
- TensorRT推論エンジン（自動変換・FP16精度）
- 通常のRTSPデコード（GStreamer不使用）
- ハードウェアエンコード（NVENC）

技術ブログ用のリファレンス実装です。

使用方法:
    python face-mosaic-yolo-jetson-preview.py <rtsp_url> [options]

例:
    python face-mosaic-yolo-jetson-preview.py "rtsp://admin:password@192.168.1.100:554/stream"
    python face-mosaic-yolo-jetson-preview.py "rtsp://camera/stream" --output udp://127.0.0.1:9000
    python face-mosaic-yolo-jetson-preview.py "rtsp://camera/stream" --model yolov8s.pt --confidence 0.6

機能:
    - 初回実行時に自動的にTensorRTエンジン（.engine）を生成
    - 2回目以降は高速なTensorRTエンジンを使用
    - 通常のcv2.VideoCapture()でRTSPストリームをデコード（GStreamer不要）
    - NVENCによるハードウェアエンコード（CPU負荷削減）
"""

import cv2
import numpy as np
import subprocess
import sys
import argparse

try:
    from ultralytics import YOLO
except ImportError:
    print("エラー: ultralyticsパッケージがインストールされていません")
    print("以下のコマンドでインストールしてください:")
    print("  pip install ultralytics")
    sys.exit(1)

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
        description='監視カメラ映像の顔モザイク処理（Jetson最適化版 - GStreamer不使用）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  %(prog)s "rtsp://admin:password@192.168.1.100:554/stream"
  %(prog)s "rtsp://camera/stream" --output udp://127.0.0.1:9000
  %(prog)s "rtsp://camera/stream" --model yolov8s.pt --confidence 0.6
  %(prog)s "rtsp://camera/stream" --width 1920 --height 1080 --fps 30

モデルの選択:
  yolov8n.pt: Nano (最速、メモリ少)
  yolov8s.pt: Small (バランス)
  yolov8m.pt: Medium (高精度)
  yolov8l.pt: Large (最高精度)
        """
    )
    
    parser.add_argument('rtsp_url', 
                       help='監視カメラのRTSPストリームURL')
    parser.add_argument('--output', '-o',
                       default='udp://127.0.0.1:8080',
                       help='出力ストリームURL (デフォルト: udp://127.0.0.1:8080)')
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
                       default=25,
                       help='フレームレート (デフォルト: 25)')
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
    parser.add_argument('--no-preview',
                       action='store_true',
                       help='プレビューウィンドウを表示しない')
    parser.add_argument('--no-tensorrt',
                       action='store_true',
                       help='TensorRT変換をスキップしてPyTorchモデルを使用')
    
    return parser.parse_args()

def main():
    # コマンドライン引数を解析
    args = parse_arguments()
    
    print("=" * 70)
    print("監視カメラ映像の顔モザイク処理（Jetson最適化版 - GStreamer不使用）")
    print("=" * 70)
    print(f"入力: {args.rtsp_url}")
    print(f"出力: {args.output}")
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
    
    # 通常のcv2.VideoCaptureでRTSPストリームを開く（GStreamer不使用）
    print("RTSPストリームに接続しています（通常デコード）...")
    
    cap = cv2.VideoCapture(args.rtsp_url)
    
    if not cap.isOpened():
        print("エラー: RTSPストリームを開けませんでした")
        print("URLが正しいか、カメラが動作しているか確認してください")
        sys.exit(1)
    
    print("接続成功")
    
    # 環境に応じたFFmpegエンコーダーの選択
    import platform
    is_jetson = os.path.exists('/etc/nv_tegra_release') or 'tegra' in platform.platform().lower()
    
    if is_jetson:
        # Jetson環境: libx264（ソフトウェアエンコード）を使用
        # Jetsonのハードウェアエンコーダー（omx/nvv4l2）はGStreamerで使用
        print("Jetson環境を検出しました。libx264エンコーダーを使用します")
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{args.width}x{args.height}',
            '-r', str(args.fps),
            '-i', '-',
            '-f', 'mpegts',
            '-codec:v', 'libx264',  # ソフトウェアエンコーダー
            '-preset', 'ultrafast',  # 最速プリセット
            '-tune', 'zerolatency',  # 低遅延チューニング
            '-b:v', '2000k',
            '-maxrate', '2500k',
            '-bufsize', '4000k',
            '-g', str(args.fps * 2),
            '-bf', '0',
            args.output,
        ]
    else:
        # 通常のPC環境: NVENCハードウェアエンコーダーを使用
        print("PC環境を検出しました。NVENCエンコーダーを使用します")
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{args.width}x{args.height}',
            '-r', str(args.fps),
            '-i', '-',
            '-f', 'mpegts',
            '-codec:v', 'h264_nvenc',  # NVENCハードウェアエンコーダ
            '-preset', 'p4',
            '-tune', 'll',
            '-b:v', '2000k',
            '-maxrate', '2500k',
            '-bufsize', '4000k',
            '-g', str(args.fps * 2),
            '-bf', '0',
            args.output,
        ]
    
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("ストリーミングを開始しました")
        print(f"\nVLCで視聴: vlc {args.output}\n")
        
    except FileNotFoundError:
        print("エラー: FFmpegが見つかりません")
        print("FFmpegをインストールしてPATHに追加してください")
        cap.release()
        sys.exit(1)
    
    frame_count = 0
    total_detections = 0
    
    try:
        print("処理を開始します（'q'キーで終了）\n")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("警告: フレームの取得に失敗しました。再接続を試みます...")
                cap.release()
                cap = cv2.VideoCapture(args.rtsp_url)
                continue
            
            # フレームをリサイズ
            frame = cv2.resize(frame, (args.width, args.height))
            
            # YOLOv8で人物検出
            results = model(
                frame, 
                classes=[0],  # 人物クラス
                conf=args.confidence,
                verbose=False
            )
            
            detected_heads = []
            
            # 検出結果を処理
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # バウンディングボックスの座標を取得
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # 整数に変換
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 人物の幅と高さ
                    person_w = x2 - x1
                    person_h = y2 - y1
                    
                    # サイズフィルタリング
                    if person_w < 30 or person_h < 50:
                        continue
                    
                    # アスペクト比チェック
                    aspect_ratio = person_w / person_h if person_h > 0 else 0
                    if aspect_ratio < 0.2 or aspect_ratio > 3.0:
                        continue
                    
                    # 頭部領域を計算
                    head_h = int(person_h * args.head_ratio)
                    head_y = y1
                    head_x = x1
                    head_w = person_w
                    
                    # 頭部領域を少し拡大
                    margin_w = int(head_w * 0.1)
                    margin_h = int(head_h * 0.1)
                    
                    head_x = max(0, head_x - margin_w)
                    head_y = max(0, head_y - margin_h)
                    head_w = min(args.width - head_x, head_w + margin_w * 2)
                    head_h = min(args.height - head_y, head_h + margin_h * 2)
                    
                    detected_heads.append((head_x, head_y, head_w, head_h, confidence))
            
            # モザイク処理
            for (x, y, w, h, conf) in detected_heads:
                frame = apply_mosaic(frame, x, y, w, h, ratio=0.05)
                total_detections += 1
            
            # FFmpegに送信
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("警告: FFmpegプロセスが終了しました")
                break
            
            frame_count += 1
            if frame_count % 100 == 0:
                avg_detections = total_detections / frame_count
                print(f"処理済み: {frame_count}フレーム | "
                      f"検出数: {len(detected_heads)} | "
                      f"平均: {avg_detections:.2f}")
            
            # プレビュー表示
            if not args.no_preview:
                cv2.putText(frame, f'Detected: {len(detected_heads)} people', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.imshow('YOLOv8 Face Mosaic (Press Q to quit)', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n終了します...")
                    break
                
    except KeyboardInterrupt:
        print("\n\nキーボード割り込みを検出しました。終了します...")
    
    finally:
        # クリーンアップ
        print("リソースを解放しています...")
        cap.release()
        cv2.destroyAllWindows()
        
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

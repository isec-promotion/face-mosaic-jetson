# Face Mosaic for Jetson - 監視カメラ映像の顔モザイク処理

NVIDIA Jetson 向けに最適化された監視カメラ映像の顔モザイク処理システムです。RTSP 経由で監視カメラの映像を取得し、YOLOv8 と TensorRT を使用して人物検出を行い、顔部分にモザイク処理を適用した上で YouTube Live に配信します。

## 概要

このプロジェクトは、プライバシー保護が必要な監視カメラ映像を公開配信する際に使用できる顔モザイク処理システムです。以下の特徴があります：

- **高精度な人物検出**: YOLOv8 による高精度な人物検出
- **TensorRT 最適化**: Jetson 向けに最適化された TensorRT 推論エンジン（自動変換・FP16 精度）
- **リアルタイム処理**: ハードウェアエンコーダーによる低遅延処理
- **YouTube Live 配信**: RTMP 経由でのライブストリーミング対応

## プログラム一覧

### 1. face-mosaic-yolo-jetson-preview.py

**ローカルプレビュー専用版**

YouTube 配信は行わず、Jetson 上でモザイク処理の結果を確認するためのプログラムです。

**特徴:**

- プレビューウィンドウで処理結果を確認可能
- FFmpeg によるローカルストリーミング（UDP 出力）
- 開発・テスト用途に最適

**使用方法:**

```bash
python face-mosaic-yolo-jetson-preview.py "rtsp://camera_url" [オプション]
```

**オプション:**

- `--output, -o`: 出力ストリーム URL（デフォルト: udp://127.0.0.1:8080）
- `--width, -W`: 出力映像の幅（デフォルト: 1280）
- `--height, -H`: 出力映像の高さ（デフォルト: 720）
- `--fps, -f`: フレームレート（デフォルト: 25）
- `--model, -m`: YOLOv8 モデル（yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt）
- `--confidence, -c`: 検出信頼度閾値（0.0-1.0、デフォルト: 0.5）
- `--no-preview`: プレビューウィンドウを表示しない
- `--no-tensorrt`: TensorRT 変換をスキップ

**例:**

```bash
# 基本的な使用方法
python face-mosaic-yolo-jetson-preview.py "rtsp://admin:password@192.168.1.100:554/stream"

# 高解像度・高精度モデルで実行
python face-mosaic-yolo-jetson-preview.py "rtsp://camera/stream" --width 1920 --height 1080 --model yolov8s.pt
```

### 2. face-mosaic-yolo-jetson-ffmpeg.py

**YouTube 配信版（FFmpeg + Jetson ハードウェアエンコーダー）**

FFmpeg を使用して YouTube Live に配信します。h264_nvmpi（Jetson ハードウェアエンコーダー）を使用します。

**特徴:**

- YouTube Live 配信対応
- FFmpeg + Jetson ハードウェアエンコーダー（h264_nvmpi）
- 低 CPU 負荷
- プレビューウィンドウなし（配信専用）
- スレッド化された RTSP 読み込み

**使用方法:**

```bash
python face-mosaic-yolo-jetson-ffmpeg.py "rtsp://camera_url" "youtube_stream_key" [オプション]
```

**オプション:**

- `--width, -W`: 出力映像の幅（デフォルト: 1280）
- `--height, -H`: 出力映像の高さ（デフォルト: 720）
- `--fps, -f`: フレームレート（デフォルト: 30）
- `--model, -m`: YOLOv8 モデル
- `--confidence, -c`: 検出信頼度閾値（デフォルト: 0.5）
- `--no-tensorrt`: TensorRT 変換をスキップ

**例:**

```bash
# YouTube Liveに配信
python face-mosaic-yolo-jetson-ffmpeg.py "rtsp://admin:password@192.168.1.100:554/stream" "xxxx-xxxx-xxxx-xxxx"

# カスタム設定で配信
python face-mosaic-yolo-jetson-ffmpeg.py "rtsp://camera/stream" "your-stream-key" --width 1920 --height 1080 --fps 30 --model yolov8s.pt
```

### 3. face-mosaic-yolo-jetson-gst.py

**YouTube 配信版（GStreamer + ハードウェアエンコーダー）**

GStreamer と Jetson のハードウェアエンコーダー（nvv4l2h264enc）を使用して YouTube Live に配信します。

**特徴:**

- Jetson ハードウェアエンコーダー使用（nvv4l2h264enc）
- GStreamer パイプライン
- 低 CPU 負荷
- YouTube Live 配信対応

**使用方法:**

```bash
python face-mosaic-yolo-jetson-gst.py "rtsp://camera_url" "youtube_stream_key" [オプション]
```

**オプション:**

- `--width, -W`: 出力映像の幅（デフォルト: 1280）
- `--height, -H`: 出力映像の高さ（デフォルト: 720）
- `--fps, -f`: フレームレート（デフォルト: 30）
- `--bitrate, -b`: 配信ビットレート（kbps、デフォルト: 2500）
- `--model, -m`: YOLOv8 モデル
- `--confidence, -c`: 検出信頼度閾値（デフォルト: 0.5）
- `--no-tensorrt`: TensorRT 変換をスキップ

**例:**

```bash
# ハードウェアエンコーダーでYouTube Liveに配信
python face-mosaic-yolo-jetson-gst.py "rtsp://camera/stream" "your-stream-key"

# カスタムビットレートで配信
python face-mosaic-yolo-jetson-gst.py "rtsp://camera/stream" "your-stream-key" --bitrate 3000 --fps 30
```

### 4. face-mosaic-yolo-jetson-x264.py

**YouTube 配信版（FFmpeg + ソフトウェアエンコーダー）**

FFmpeg と libx264（ソフトウェアエンコーダー）を使用して YouTube Live に配信します。Jetson のハードウェアエンコーダーが利用できない環境や、互換性を重視する場合に使用します。

**使用方法:**

```bash
python face-mosaic-yolo-jetson-x264.py "rtsp://camera_url" "youtube_stream_key" [オプション]
```

## 必要な環境

### ハードウェア

- NVIDIA Jetson（Nano、Xavier NX、AGX Xavier、Orin など）
- 監視カメラ（RTSP 対応）

### ソフトウェア

- Python 3.6 以上
- CUDA Toolkit
- TensorRT
- OpenCV（GStreamer サポート付き、gst 版を使用する場合）
- FFmpeg（ffmpeg/x264 版を使用する場合）
- GStreamer（gst 版を使用する場合）

## インストール

### 1. 依存パッケージのインストール

```bash
# Python依存パッケージ
pip install ultralytics opencv-python numpy

# CUDA対応PyTorchのインストール（必要に応じて）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. FFmpeg のインストール（ffmpeg/x264 版を使用する場合）

```bash
sudo apt update
sudo apt install ffmpeg
```

### 3. GStreamer のインストール（gst 版を使用する場合）

```bash
sudo apt update
sudo apt install \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev

# nvv4l2h264encが利用可能か確認
gst-inspect-1.0 nvv4l2h264enc
```

## 使用方法

### 初回実行時の注意

初回実行時には以下の処理が自動的に行われます：

1. **YOLOv8 モデルのダウンロード**: 指定したモデル（yolov8n.pt 等）が自動的にダウンロードされます
2. **TensorRT エンジンの生成**: PyTorch モデルから TensorRT エンジン（.engine）が生成されます（数分かかる場合があります）

2 回目以降の実行では、生成された TensorRT エンジンが使用されるため、起動が高速になります。

### YouTube Live ストリーミングキーの取得

1. YouTube Studio にアクセス: https://studio.youtube.com
2. 左メニューから「配信」を選択
3. 「ストリームキー」をコピー

### 基本的な使用フロー

1. **プレビュー版でテスト**:

```bash
python face-mosaic-yolo-jetson-preview.py "rtsp://camera_url"
```

2. **モザイク処理を確認後、配信開始**:

```bash
# FFmpeg版
python face-mosaic-yolo-jetson-ffmpeg.py "rtsp://camera_url" "your-stream-key"

# GStreamer版（推奨：ハードウェアエンコーダー使用）
python face-mosaic-yolo-jetson-gst.py "rtsp://camera_url" "your-stream-key"
```

3. **YouTube Studio で配信状況を確認**: https://studio.youtube.com

## モデルの選択

YOLOv8 には複数のモデルサイズがあり、精度と速度のトレードオフがあります：

| モデル     | サイズ | 速度 | 精度     | 推奨用途             |
| ---------- | ------ | ---- | -------- | -------------------- |
| yolov8n.pt | Nano   | 最速 | 標準     | リアルタイム処理優先 |
| yolov8s.pt | Small  | 高速 | 良好     | バランス重視         |
| yolov8m.pt | Medium | 中速 | 高精度   | 精度重視             |
| yolov8l.pt | Large  | 低速 | 最高精度 | オフライン処理       |

**推奨**: Jetson Nano では`yolov8n.pt`、Jetson Xavier 以上では`yolov8s.pt`を推奨します。

## トラブルシューティング

### CUDA が利用できない

```bash
# PyTorchのCUDA対応版をインストール
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### FFmpeg が見つからない

```bash
# FFmpegをインストール
sudo apt install ffmpeg

# PATHを確認
which ffmpeg
```

### GStreamer エンコーダーが起動しない

```bash
# GStreamerプラグインを確認
gst-inspect-1.0 nvv4l2h264enc

# OpenCVがGStreamerサポート付きでビルドされているか確認
python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
```

### RTSP ストリームに接続できない

- カメラの IP アドレス、ポート番号、認証情報を確認
- ファイアウォール設定を確認
- カメラの管理画面で RTSP が有効になっているか確認

### YouTube 配信が開始されない

- ストリームキーが正しいか確認
- YouTube Studio で配信状態を確認
- ネットワーク接続を確認
- 配信開始まで数秒〜数十秒かかる場合があります

## パフォーマンスチューニング

### フレームレートの調整

```bash
# 低フレームレート（CPU負荷軽減）
python face-mosaic-yolo-jetson-*.py ... --fps 15

# 高フレームレート（滑らかな映像）
python face-mosaic-yolo-jetson-*.py ... --fps 30
```

### 解像度の調整

```bash
# HD解像度（バランス）
python face-mosaic-yolo-jetson-*.py ... --width 1280 --height 720

# Full HD（高画質、高負荷）
python face-mosaic-yolo-jetson-*.py ... --width 1920 --height 1080

# SD解像度（低負荷）
python face-mosaic-yolo-jetson-*.py ... --width 640 --height 480
```

### 検出パラメータの調整

```bash
# 検出感度を上げる（より多くの人物を検出）
python face-mosaic-yolo-jetson-*.py ... --confidence 0.3

# 検出感度を下げる（誤検出を減らす）
python face-mosaic-yolo-jetson-*.py ... --confidence 0.7
```

## ライセンス

このプロジェクトは MIT ライセンスのもとで公開されています。

## 謝辞

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [OpenCV](https://opencv.org/)
- [FFmpeg](https://ffmpeg.org/)
- [GStreamer](https://gstreamer.freedesktop.org/)

## 注意事項

- プライバシー保護のため、必ず適切にモザイク処理が施されていることを確認してから配信してください
- 監視カメラの使用および配信には、適用される法律や規制を遵守してください
- このソフトウェアの使用によって生じるいかなる損害についても、作者は責任を負いません

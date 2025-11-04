# Collision Warning App (YOLOv8 + OpenCV)

## 1) Cài đặt nhanh (Windows/macOS/Linux)
```bash
# 1. Cài Python 3.10+ và VS Code (+Extensions: Python, Pylance)
# 2. Mở VS Code -> Terminal mới
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Ghi chú: Nếu có GPU NVIDIA, cài thêm PyTorch CUDA theo hướng dẫn của PyTorch rồi mới `pip install ultralytics` (tự động tận dụng GPU).

## 2) Chạy ứng dụng
```bash
# Camera mặc định
python main.py --source 0

# Hoặc chạy với video có sẵn
python main.py --source path/to/video.mp4

# Tùy chọn
python main.py --conf 0.35 --ttc_warn 2.0 --ttc_clear 3.0 --show_ttc 1
```

## 3) Cấu trúc
```
collision_warning_app/
├─ main.py                # App chính
├─ requirements.txt
└─ README.md
```

## 4) Lỗi thường gặp
- `ModuleNotFoundError`: Chưa kích hoạt venv hoặc chưa cài `pip install -r requirements.txt`.
- Video đen/không hiển thị: Chọn đúng `--source` (0,1,2...) hoặc cấp quyền camera.
- Giật FPS: Giảm độ phân giải camera, tăng `--conf`, tắt hiển thị TTC.

## 5) Mẹo hiệu năng
- Dùng YOLOv8n (mặc định) cho CPU; có GPU thì vẫn ok.
- Giảm kích thước khung hình (ví dụ 640x480).
- Bật GPU (cài PyTorch CUDA) nếu có NVIDIA.

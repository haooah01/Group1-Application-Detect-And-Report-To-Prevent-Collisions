import argparse
import time
import math
import sys
from collections import deque

import cv2
import numpy as np

# YOLOv8
try:
    from ultralytics import YOLO
except Exception as e:
    print("Lỗi import Ultralytics YOLO. Hãy chắc chắn đã `pip install ultralytics`.", file=sys.stderr)
    raise

# Âm thanh: ưu tiên pygame, fallback winsound (Windows), cuối cùng là chuông hệ thống
def init_audio():
    try:
        import pygame
        # Khởi tạo mixer 1 kênh (mono) để phù hợp với mảng 1D
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        return "pygame"
    except Exception:
        try:
            import winsound  # type: ignore
            return "winsound"
        except Exception:
            return "bell"

def play_beep(backend, level="warn"):
    if backend == "pygame":
        import pygame
        import numpy as np
        freq = 880 if level == "warn" else 660
        dur_ms = 120 if level == "warn" else 80
        sr = 44100
        t = np.linspace(0, dur_ms/1000.0, int(sr*dur_ms/1000.0), False)
        tone = (0.5*np.sin(2*np.pi*freq*t)).astype(np.float32)
        # chuyển sang int16 mono, phát qua buffer
        arr = (tone * 32767).astype(np.int16)
        snd = pygame.mixer.Sound(buffer=arr.tobytes())
        snd.play()
    elif backend == "winsound":
        import winsound
        freq = 880 if level == "warn" else 660
        dur_ms = 120 if level == "warn" else 80
        winsound.Beep(freq, dur_ms)
    else:
        print("\a", end="")

CLASSES_KEEP = {"car", "motorcycle", "bus", "truck", "person"}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help="0 (camera) hoặc đường dẫn video")
    ap.add_argument("--conf", type=float, default=0.35, help="ngưỡng độ tin cậy YOLO")
    ap.add_argument("--ttc_warn", type=float, default=2.0, help="ngưỡng TTC (giây) kích hoạt cảnh báo")
    ap.add_argument("--ttc_clear", type=float, default=3.0, help="ngưỡng TTC (giây) tắt cảnh báo (hysteresis)")
    ap.add_argument("--show_ttc", type=int, default=1, help="hiển thị TTC trên overlay (1/0)")
    ap.add_argument("--width", type=int, default=640, help="độ rộng frame (giảm để tăng FPS)")
    ap.add_argument("--height", type=int, default=480, help="độ cao frame")
    return ap.parse_args()

def open_source(src_str, w, h):
    try:
        src = int(src_str)
    except ValueError:
        src = src_str
    cap = cv2.VideoCapture(src)
    if w and h:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video {src_str}")
    return cap

def estimate_ttc(prev_h, curr_h, dt):
    # TTC xấp xỉ: TTC ≈ -h / (dh/dt). Khi h tăng (tiến gần), dh/dt > 0 => TTC = h / (dh/dt)
    if dt <= 1e-6:
        return None
    dh = curr_h - prev_h
    hdot = dh / dt
    if hdot <= 1e-3 or curr_h <= 1e-3:
        return None
    return max(0.0, float(curr_h / hdot))

def center_region_priority(xc, w_frame, margin_ratio=0.25):
    # Ưu tiên đối tượng gần tâm khung hình theo trục ngang
    left = w_frame * margin_ratio
    right = w_frame * (1 - margin_ratio)
    if xc < left:
        return 1  # mép trái
    elif xc > right:
        return 1  # mép phải
    else:
        return 0  # vùng trung tâm

def main():
    args = parse_args()
    audio_backend = init_audio()

    # Tải YOLOv8n (pretrained COCO)
    model = YOLO("yolov8n.pt")

    cap = open_source(args.source, args.width, args.height)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Lưu chiều cao bbox qua thời gian để tính TTC
    history = {}  # id_like -> deque[(t, h)]
    warn_active = False
    last_beep = 0.0

    fps_hist = deque(maxlen=30)
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_now = time.time()
        dt_loop = t_now - t_prev if t_prev else 0.0
        t_prev = t_now

        # YOLO inference
        results = model.predict(frame, verbose=False, conf=args.conf, imgsz=max(args.width, args.height))

        # Gom đối tượng quan trọng
        candidates = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, str(cls_id))
                if cls_name not in CLASSES_KEEP:
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                conf = float(box.conf[0].item())
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                priority = center_region_priority(xc, w_frame)
                candidates.append({
                    "bbox": (x1, y1, x2, y2),
                    "cls": cls_name,
                    "conf": conf,
                    "w": w, "h": h,
                    "xc": xc, "yc": yc,
                    "priority": priority
                })

        # Chọn đối tượng ưu tiên: trong vùng trung tâm, cao nhất (gần) trước
        candidates.sort(key=lambda c: (c["priority"], -c["h"]))

        # Tính TTC cho đối tượng ưu tiên nhất (nếu có)
        ttc = None
        danger_bbox = None
        if candidates:
            c0 = candidates[0]
            key = f"{c0['cls']}_center"  # id tạm theo lớp + trung tâm
            hist = history.setdefault(key, deque(maxlen=5))
            hist.append((t_now, c0["h"]))

            if len(hist) >= 2:
                (t_prev_h, h_prev), (t_curr_h, h_curr) = hist[-2], hist[-1]
                ttc = estimate_ttc(h_prev, h_curr, t_curr_h - t_prev_h)

            # Vẽ khung & nhãn
            (x1, y1, x2, y2) = c0["bbox"]
            color = (0, 255, 0)  # xanh: an toàn
            label = f"{c0['cls']} {c0['conf']:.2f}"
            if ttc is not None and ttc <= args.ttc_warn:
                color = (0, 0, 255)  # đỏ: nguy hiểm
                danger_bbox = (x1, y1, x2, y2)
            elif ttc is not None and ttc <= args.ttc_clear:
                color = (0, 255, 255)  # vàng: cảnh giác

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), max(0, int(y1)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Cảnh báo âm thanh theo hysteresis
        if ttc is not None:
            if not warn_active and ttc <= args.ttc_warn:
                warn_active = True
                if time.time() - last_beep > 0.2:
                    play_beep(audio_backend, "warn")
                    last_beep = time.time()
            elif warn_active and ttc > args.ttc_clear:
                warn_active = False
        else:
            # Không có TTC hợp lệ -> giữ trạng thái hiện tại, không beep thêm
            pass

        # Overlay TTC & trạng thái
        status = "SAFE"
        if warn_active:
            status = "DANGER"
        elif ttc is not None and ttc <= args.ttc_clear:
            status = "CAUTION"

        if args.show_ttc:
            ttc_str = f"TTC: {ttc:.2f}s" if ttc is not None else "TTC: --"
            cv2.putText(frame, f"{status} | {ttc_str}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if warn_active else ((0, 255, 255) if status=='CAUTION' else (0, 255, 0)), 2)

        # FPS
        fps_hist.append(1.0 / dt_loop if dt_loop > 1e-6 else 0.0)
        fps = sum(fps_hist) / max(1, len(fps_hist))
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Biểu tượng cảnh báo
        if warn_active and danger_bbox is not None:
            (x1,y1,x2,y2) = danger_bbox
            # Vẽ tam giác cảnh báo nhỏ phía trên trung tâm bbox
            cx = int((x1+x2)/2)
            top = int(y1) - 20
            pts = np.array([[cx, top], [cx-12, top+24], [cx+12, top+24]], np.int32)
            cv2.fillPoly(frame, [pts], (0,0,255))
            cv2.putText(frame, "!", (cx-4, top+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Collision Warning (YOLOv8 + TTC)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

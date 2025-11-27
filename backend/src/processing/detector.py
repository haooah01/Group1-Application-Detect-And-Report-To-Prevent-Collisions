from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        Khởi tạo và tải mô hình YOLOv8.
        """
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            # 5 lớp chính cần nhận diện theo tài liệu
            self.target_classes = ['car', 'motorcycle', 'bus', 'truck', 'person']
            print(f"Tải mô hình '{model_path}' thành công.")
        except Exception as e:
            print(f"Lỗi khi tải mô hình '{model_path}': {e}")
            raise

    def preprocess_frame(self, frame):
        """
        Tiền xử lý khung hình để phù hợp với YOLOv8:
        - Chuyển từ BGR sang RGB
        - Resize + letterbox giữ tỷ lệ
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame

    def detect_objects(self, frame):
        """
        Nhận diện đối tượng trong một khung hình.
        """
        rgb_frame = self.preprocess_frame(frame)
        results = self.model.predict(rgb_frame, conf=self.conf_threshold, verbose=False)
        return results
    
    def detect_and_track(self, frame):
        """
        Hàm tương thích với desktop_app.py.
        Dùng YOLOv8 để nhận diện và theo dõi đối tượng (track).
        """
        rgb_frame = self.preprocess_frame(frame)
        results = self.model.track(rgb_frame, persist=True, conf=self.conf_threshold, verbose=False)
        return results

    

    def filter_results(self, results):
        """
        Giữ lại các đối tượng thuộc lớp mục tiêu và có độ tin cậy đủ cao.
        """
        detections = []
        for box in results[0].boxes:
            cls_name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            if cls_name in self.target_classes and conf >= self.conf_threshold:
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                })
        return detections

    def draw_results(self, frame, results):
        """
        Vẽ khung bao và nhãn lên khung hình.
        """
        annotated_frame = results[0].plot()
        return annotated_frame

# --- Kiểm tra nhanh ---
if __name__ == "__main__":
    detector = ObjectDetector()
    print("Khởi tạo ObjectDetector thành công.")

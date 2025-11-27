import time

# === CÁC HẰNG SỐ HIỆU CHỈNH ===
# Chiều rộng trung bình (m) của các loại đối tượng (theo COCO)
KNOWN_WIDTHS = {
    1: 0.6,   # person
    2: 1.8,   # car
    3: 0.8,   # motorcycle
    5: 2.5,   # bus
    7: 2.3    # truck
}

# Tiêu cự camera (pixel) — cần hiệu chỉnh thực tế
FOCAL_LENGTH = 800  

# Ngưỡng khoảng cách cảnh báo tức thì (m)
CRITICAL_DISTANCES = {
    1: 1.5,   # person
    2: 1.8,   # car
    3: 2.5,   # motorcycle
    5: 2.0,   # bus
    7: 2.0    # truck
}

# Ngưỡng phát hiện tốc độ phóng to khung (pixel/frame)
EXPANSION_RATE_THRESHOLD = 10.0

# Hệ số làm mượt khoảng cách (EMA)
EMA_ALPHA = 0.4


class TTCCalculator:
    """
    Bộ xử lý ước lượng Time To Collision (TTC)
    - Làm mượt khoảng cách
    - Tính TTC dựa trên đạo hàm kích thước khung
    - Phát hiện cảnh báo tức thì khi đối tượng quá gần hoặc lao tới nhanh
    """

    def __init__(self, fps):
        self.history = {}          # track_id: dict
        self.smooth_dist = {}       # track_id: giá trị khoảng cách đã làm mượt
        self.time_per_frame = 1.0 / fps if fps > 0 else 0.033
        self.alert_active = False   # dùng cho hysteresis
        print(f"TTCCalculator khởi tạo với FPS={fps:.2f}")

    # --- ƯỚC LƯỢNG KHOẢNG CÁCH ---
    def _estimate_distance(self, class_id, pixel_size):
        """Ước lượng khoảng cách (m) dựa trên kích thước pixel của bounding box."""
        if pixel_size <= 0:
            return float('inf')
        known_width = KNOWN_WIDTHS.get(class_id, 1.8)
        return (known_width * FOCAL_LENGTH) / pixel_size

    # --- LÀM MƯỢT DỮ LIỆU ---
    def _smooth_value(self, track_id, new_value):
        prev = self.smooth_dist.get(track_id, new_value)
        smoothed = EMA_ALPHA * new_value + (1 - EMA_ALPHA) * prev
        self.smooth_dist[track_id] = smoothed
        return smoothed

    # --- TÍNH TOÁN TTC ---
    def calculate_ttc(self, track_id, box, class_id=None):
        """
        Tính toán khoảng cách, vận tốc, TTC, và cờ cảnh báo tức thì.
        """
        x1, y1, x2, y2 = box
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        pixel_size = (width + height) / 2.0

        # Ước tính khoảng cách
        raw_distance = self._estimate_distance(class_id, pixel_size)
        distance = self._smooth_value(track_id, raw_distance)
        current_time = time.time()

        # Khởi tạo lịch sử nếu chưa có
        if track_id not in self.history:
            self.history[track_id] = {
                "distances": [distance],
                "sizes": [pixel_size],
                "velocity": 0.0,
                "last_time": current_time
            }
            return distance, 0.0, float("inf"), False

        entry = self.history[track_id]
        distances = entry["distances"]
        sizes = entry["sizes"]

        # Cập nhật danh sách lịch sử
        distances.append(distance)
        sizes.append(pixel_size)
        if len(distances) > 6:
            distances.pop(0)
        if len(sizes) > 6:
            sizes.pop(0)

        # Tính vận tốc trung bình
        diffs = [distances[i] - distances[i - 1] for i in range(1, len(distances))]
        avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
        velocity = avg_diff / self.time_per_frame

        # Tính tốc độ phóng to bounding box
        size_diffs = [sizes[i] - sizes[i - 1] for i in range(1, len(sizes))]
        avg_size_rate = sum(size_diffs) / len(size_diffs) if size_diffs else 0.0

        # Tính TTC (Time to Collision)
        ttc = float("inf")
        if velocity < -0.05:  # đối tượng đang tiến lại gần
            ttc = distance / (-velocity)

        # --- Kiểm tra điều kiện cảnh báo tức thì ---
        immediate_alert = False

        # 1. Nếu khoảng cách < ngưỡng an toàn
        crit_dist = CRITICAL_DISTANCES.get(class_id, 1.8)
        if distance <= crit_dist:
            immediate_alert = True

        # 2. Nếu bounding box tăng nhanh (đối tượng lao tới)
        if avg_size_rate >= EXPANSION_RATE_THRESHOLD:
            immediate_alert = True

        # Cập nhật lịch sử
        self.history[track_id] = {
            "distances": distances,
            "sizes": sizes,
            "velocity": velocity,
            "last_time": current_time
        }

        return distance, velocity, ttc, immediate_alert

    # --- LOGIC HYSTERESIS ---
    def check_collision_warning(self, ttc):
        """
        Logic bật/tắt cảnh báo dựa trên TTC (hysteresis)
        """
        TTC_ON_THRESHOLD = 2.0
        TTC_OFF_THRESHOLD = 2.5

        if not self.alert_active and ttc <= TTC_ON_THRESHOLD:
            self.alert_active = True
        elif self.alert_active and ttc >= TTC_OFF_THRESHOLD:
            self.alert_active = False

        return self.alert_active

    # --- DỌN DẸP BỘ NHỚ ---
    def cleanup_history(self, current_track_ids):
        """Xóa track_id không còn xuất hiện để tránh tràn bộ nhớ."""
        current_ids = set(current_track_ids)
        self.history = {tid: h for tid, h in self.history.items() if tid in current_ids}
        self.smooth_dist = {tid: d for tid, d in self.smooth_dist.items() if tid in current_ids}

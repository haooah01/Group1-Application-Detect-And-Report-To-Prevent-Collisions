import tkinter as tk
from tkinter import filedialog
import cv2
import os
from PIL import Image, ImageTk
import pygame
import time
import math

# --- IMPORT CAC MODULE CUA CHUNG TA ---
from backend.src.processing.detector import ObjectDetector
from backend.src.processing.calculator import TTCCalculator
# Import ca 2 giao dien:
from ui.main_window_ui import MainWindowUI
from ui.settings_window_ui import SettingsWindowUI
from ui.about_window_ui import AboutWindowUI

# --- 1. CAU HINH CAC DUONG DAN ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_PATH = os.path.join(ROOT_DIR, "data", "videos", "test_video.mp4")
SOUND_PATH = os.path.join(ROOT_DIR, "assets", "sounds", "alert.mp3")

# --- 2. LOP UNG DUNG (BO DIEU KHIEN) ---

class CollisionApp:
    def __init__(self):
        # --- A. Khoi tao cua so chinh ---
        self.window = tk.Tk()
        self.window.title("Ung dung Canh bao Va cham")
        self.window.geometry("1280x720")

        # --- B. Khoi tao Cau hinh (Config) ---
        self.config = {
            "ttc_threshold": 3.0,
            "ai_model": "yolov8n.pt",
            "sound_enabled": True
        }

        # --- C. Tao Giao dien tu file UI ---
        self.ui = MainWindowUI(self.window)

        # --- D. Khoi tao cac bien logic ---
        self.cap = None
        self.calculator = None
        self.video_path = DEFAULT_VIDEO_PATH
        self.is_running = False
        self.alert_triggered = False
        self.settings_window = None  # De theo doi cua so Cai dat
        self.about_window = None

        # --- E. Khoi tao cac he thong phu ---
        self.detector = self._init_detector()
        self.sound_enabled = self._init_sound()

        if self.detector is None:
            self.ui.status_bar_label.config(text="LOI: Khong the tai mo hinh AI.")
            return

        # --- F. Ket noi Logic voi Giao dien ---
        self.ui.start_button.config(command=self.play_resume_video)
        self.ui.stop_button.config(command=self.pause_video)
        self.ui.settings_button.config(command=self.open_settings_window)
        self.ui.about_button.config(command=self.open_about_window)
        self.ui.exit_button.config(command=self.window.quit)

        # An hop canh bao luc ban dau
        self.ui.warning_frame.pack_forget()

        # --- G. Khoi tao cac bien theo doi FPS ---
        self.frame_count = 0
        self.start_time = time.time()

        # --- H. Bat dau ung dung ---
        print("Khoi chay ung dung.")
        self.window.mainloop()

    # --- CAC HAM KHOI TAO ---

    def _init_detector(self):
        """Tai mo hinh AI dua tren config."""
        try:
            print(f"Dang tai mo hinh AI: {self.config['ai_model']}...")
            detector = ObjectDetector(self.config['ai_model'])
            print("Tai mo hinh thanh cong.")
            return detector
        except Exception as e:
            print(f"[LOI] Khong the tai mo hinh YOLO. Loi: {e}")
            return None

    def _init_sound(self):
        """Khoi tao hoac tat am thanh dua tren config."""
        if not self.config["sound_enabled"]:
            print("Am thanh da bi tat trong cai dat.")
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            return False

        if not os.path.exists(SOUND_PATH):
            print(f"[CANH BAO] Khong tim thay file am thanh tai: {SOUND_PATH}")
            return False

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(SOUND_PATH)
            print(f"Tai am thanh canh bao ({SOUND_PATH}) thanh cong.")
            return True
        except Exception as e:
            print(f"[CANH BAO] Khong tai duoc file am thanh (pygame): {e}")
            return False

    def open_video_file(self):
        """Ham nay duoc goi boi cua so Cai dat."""
        self.pause_video()
        if self.cap:
            self.cap.release()
            self.cap = None
            print("Da giai phong video cu.")

        path = filedialog.askopenfilename(
            title="Chon file video",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        if not path:
            return

        self.video_path = path
        self.play_resume_video()

    def open_settings_window(self):
        """Mo cua so Cai dat."""
        if self.settings_window is not None and self.settings_window.window.winfo_exists():
            self.settings_window.window.focus()
            return
        self.settings_window = SettingsWindowUI(self.window, self, self.config)

    def save_settings(self, new_config):
        """Luu cau hinh moi."""
        print("Luu cai dat moi:", new_config)
        model_changed = self.config["ai_model"] != new_config["ai_model"]
        sound_changed = self.config["sound_enabled"] != new_config.get("sound_enabled", True)
        self.config.update(new_config)

        if model_changed:
            self.detector = self._init_detector()
            self.ui.status_bar_label.config(text=f"Da doi model thanh: {self.config['ai_model']}")

        if sound_changed:
            self.sound_enabled = self._init_sound()

        print("Da cap nhat cai dat.")

    def open_about_window(self):
        """Mo cua so Gioi thieu."""
        if self.about_window is not None and self.about_window.window.winfo_exists():
            self.about_window.window.focus()
            return
        self.about_window = AboutWindowUI(self.window)

    def play_resume_video(self):
        """Bat dau hoac tiep tuc phat video."""
        if self.cap is None:
            if not self.video_path or not os.path.exists(self.video_path):
                print(f"Loi: Khong tim thay file video {self.video_path}")
                return

            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print("Loi: Khong the mo file video.")
                self.cap = None
                return

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.calculator = TTCCalculator(fps)
            print(f"Da tai video: {self.video_path}")

        if not self.is_running:
            self.is_running = True
            print("Video dang phat...")
            self.update_frame()

    def pause_video(self):
        """Tam dung video."""
        self.is_running = False
        print("Video da tam dung.")

    def play_alert_sound(self):
        if self.sound_enabled:
            pygame.mixer.music.play()

    def get_vehicle_name(self, class_id):
        """Tra ve ten phuong tien theo class_id."""
        mapping = {
            1: "Nguoi di bo",
            2: "O to",
            3: "Xe may",
            5: "Xe buyt",
            7: "Xe tai"
        }
        return mapping.get(class_id, "Khong xac dinh")

    def update_frame(self):
        if not self.is_running or not self.cap:
            return

        success, frame = self.cap.read()
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.window.after(10, self.update_frame)
            return

        results = self.detector.detect_and_track(frame)
        annotated_frame = self.detector.draw_results(frame, results)

        current_track_ids = []
        overall_danger = False
        min_ttc = float("inf")
        min_distance = float("inf")
        min_class_id = None
        object_count = 0

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_data = results[0].boxes.data.cpu().numpy()
            current_track_ids = results[0].boxes.id.int().cpu().tolist()
            object_count = len(current_track_ids)

            frame_h, frame_w = frame.shape[:2]
            center_x = frame_w / 2

            for box_data in boxes_data:
                x1, y1, x2, y2, track_id, conf, class_id = box_data
                box = [x1, y1, x2, y2]
                track_id = int(track_id)
                class_id = int(class_id)

                cx = (x1 + x2) / 2
                roi_width = frame_w * 0.6
                if abs(cx - center_x) > roi_width / 2:
                    continue

                if class_id in [2, 3, 5, 7]:
                    distance, velocity, ttc, immediate_alert = self.calculator.calculate_ttc(track_id, box, class_id)
                    hysteresis_alert = self.calculator.check_collision_warning(ttc)
                    alert_state = hysteresis_alert or immediate_alert
                    if alert_state:
                        overall_danger = True

                    color = (0, 0, 255) if alert_state else (0, 255, 0)
                    info_text = f"D: {distance:.1f}m"

                    if ttc != float("inf") and ttc < min_ttc:
                        min_ttc = ttc
                        min_distance = distance
                        min_class_id = class_id

                        ttc_text = f"TTC: {ttc:.2f}s"
                        cv2.putText(annotated_frame, ttc_text, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(annotated_frame, info_text, (int(x1), int(y1) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- HIEN THI HUD TTC NGUY HIEM NHAT ---
        x0 = frame.shape[1] - 320
        y0 = 50
        bg_color = (80, 80, 80)
        icon = ""
        title_text = "NORMAL"
        ttc_text = "TTC: --"
        dist_text = "Khoang cach: --"
        vehicle_text = "Phuong tien: --"

        if min_ttc != float("inf"):
            if min_ttc > 3.0:
                bg_color = (0, 200, 0)
                icon = ""
                title_text = "AN TOAN"
            elif 2.0 < min_ttc <= 3.0:
                bg_color = (0, 255, 255)
                icon = "⚠"
                title_text = "CANH BAO"
            else:
                bg_color = (0, 0, 255)
                icon = ""
                title_text = "NGUY HIEM!"
            ttc_text = f"TTC: {min_ttc:.2f}s"
            dist_text = f"Khoang cach: {min_distance:.1f}m"
            vehicle_text = f"Phuong tien: {self.get_vehicle_name(min_class_id)}"

        # --- 3. Ve khung nen trong suot hon ---
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (x0 - 10, y0 - 40), (x0 + 290, y0 + 120), bg_color, -1)

        # alpha = 0.35 de tao do trong suot (co the tang/giam)
        alpha = 0.35
        annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)

        # Ve vien trang ben ngoai
        cv2.rectangle(annotated_frame, (x0 - 10, y0 - 40), (x0 + 290, y0 + 120), (255, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame, f"{icon}  {title_text}", (x0, y0), font, 1.0, (255, 255, 255), 3)
        cv2.putText(annotated_frame, ttc_text, (x0, y0 + 40), font, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, dist_text, (x0, y0 + 75), font, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, vehicle_text, (x0, y0 + 110), font, 0.8, (255, 255, 255), 2)

        # --- 3. Xu ly canh bao hinh anh & am thanh ---
        self.calculator.cleanup_history(current_track_ids)

        # Kiem tra xem co canh bao do hay khong (min_ttc <= 2.0)
        is_red_alert = (min_ttc != float("inf") and min_ttc <= 2.0)

        # Hien/An khung canh bao UI (giu nhu cu)
        if overall_danger and not self.alert_triggered:
            self.ui.warning_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
            self.alert_triggered = True
            self.play_alert_sound()
        elif not overall_danger and self.alert_triggered:
            self.ui.warning_frame.grid_forget()
            self.alert_triggered = False


        # --- CHI PHAT AM THANH KHI CANH BAO DO ---
        if is_red_alert:
            # Neu am thanh bat va chua phat thi phat
            if self.sound_enabled and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        else:
            # Neu khong con canh bao do thi dung am thanh
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()


        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            status_text = (f"FPS: {fps:.1f} | Objects: {object_count} | "
                           f"Model: {self.config['ai_model']} | "
                           f"Sound: {'ON' if self.sound_enabled else 'OFF'}")
            self.ui.status_bar_label.config(text=status_text)
            self.frame_count = 0
            self.start_time = time.time()

        label_w = self.ui.video_label.winfo_width()
        label_h = self.ui.video_label.winfo_height()
        if label_w > 1 and label_h > 1:
            frame_resized = cv2.resize(annotated_frame, (label_w, label_h))
        else:
            frame_resized = annotated_frame

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=pil_img)
        self.ui.video_label.configure(image=img_tk)
        self.ui.video_label.image = img_tk
        self.ui.status_bar_label.lift()


        self.window.after(1, self.update_frame)


if __name__ == "__main__":
    app = CollisionApp()

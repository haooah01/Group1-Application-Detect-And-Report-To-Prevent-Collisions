import tkinter as tk
from tkinter import ttk

class SettingsWindowUI:
    def __init__(self, parent, app, config):
        self.window = tk.Toplevel(parent)
        self.window.title("⚙ Cài đặt hệ thống")
        self.window.geometry("430x370")
        self.window.configure(bg="#2b2b2b")  # xám dịu hơn
        self.app = app
        self.config = config

        # --- Tiêu đề ---
        tk.Label(
            self.window,
            text="CÀI ĐẶT HỆ THỐNG",
            fg="#4dd0e1",  # xanh ngọc nhẹ
            bg="#2b2b2b",
            font=("Helvetica", 17, "bold")
        ).pack(pady=15)

        # --- Khung nội dung ---
        frame = tk.Frame(self.window, bg="#2b2b2b")
        frame.pack(fill="both", expand=True, padx=25)

        label_style = {"fg": "#f0f0f0", "bg": "#2b2b2b", "font": ("Helvetica", 12)}

        # --- Model AI ---
        tk.Label(frame, text="Mô hình AI:", **label_style).grid(
            row=0, column=0, sticky="w", pady=10
        )
        self.model_entry = ttk.Entry(frame, width=28)
        self.model_entry.insert(0, config["ai_model"])
        self.model_entry.grid(row=0, column=1, sticky="e")

        # --- Âm thanh ---
        self.sound_var = tk.BooleanVar(value=config["sound_enabled"])
        tk.Checkbutton(
            frame,
            text="Bật âm thanh cảnh báo",
            variable=self.sound_var,
            fg="#f0f0f0",
            bg="#2b2b2b",
            selectcolor="#4dd0e1",
            activebackground="#2b2b2b",
            font=("Helvetica", 12)
        ).grid(row=1, column=0, columnspan=2, pady=10, sticky="w")

        # --- Ngưỡng TTC ---
        tk.Label(
            frame, text="Ngưỡng TTC cảnh báo (giây):", **label_style
        ).grid(row=2, column=0, sticky="w", pady=10)

        self.ttc_scale = ttk.Scale(
            frame, from_=1.0, to=5.0, orient="horizontal", length=220
        )
        self.ttc_scale.set(config["ttc_threshold"])
        self.ttc_scale.grid(row=2, column=1, sticky="e")

        # --- Chọn video ---
        style = ttk.Style()
        style.configure(
            "Modern.TButton",
            font=("Helvetica", 11, "bold"),
            padding=6,
            background="#4dd0e1",
            foreground="#1e1e1e"
        )
        style.map("Modern.TButton",
                  background=[("active", "#64e5f3")])

        ttk.Button(
            frame,
            text="🎞  Chọn file video",
            style="Modern.TButton",
            command=app.open_video_file
        ).grid(row=3, column=0, columnspan=2, pady=18)

        # --- Khung nút hành động ---
        button_frame = tk.Frame(self.window, bg="#2b2b2b")
        button_frame.pack(pady=12)

        save_btn = ttk.Button(
            button_frame,
            text="💾  Lưu thay đổi",
            style="Modern.TButton",
            command=self.save_settings
        )
        close_btn = ttk.Button(
            button_frame,
            text="❌  Đóng",
            style="Modern.TButton",
            command=self.window.destroy
        )
        save_btn.pack(side="left", padx=12)
        close_btn.pack(side="right", padx=12)

        # --- Ghi chú nhỏ ---
        tk.Label(
            self.window,
            text="Thay đổi sẽ được áp dụng ngay khi lưu.",
            fg="#a0a0a0",
            bg="#2b2b2b",
            font=("Helvetica", 10, "italic")
        ).pack(pady=(0, 10))

    # --- Lưu cấu hình ---
    def save_settings(self):
        new_config = {
            "ai_model": self.model_entry.get(),
            "sound_enabled": self.sound_var.get(),
            "ttc_threshold": float(self.ttc_scale.get()),
        }
        self.app.save_settings(new_config)
        self.window.destroy()

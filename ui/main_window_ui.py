import tkinter as tk
from tkinter import ttk

class MainWindowUI:
    def __init__(self, root):
        root.configure(bg="#1e1e1e")
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # --- Header ---
        self.header = tk.Label(
            root,
            text="🚗  ỨNG DỤNG CẢNH BÁO VA CHẠM",
            bg="#1e1e1e",
            fg="#4dd0e1",  # xanh ngọc dịu hơn
            font=("Helvetica", 22, "bold"),
            pady=10
        )
        self.header.grid(row=0, column=0, sticky="ew")

        # --- Vùng video ---
        self.video_label = tk.Label(root, bg="black", relief="sunken")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 5))

        # --- Khung cảnh báo ---
        self.warning_frame = tk.Frame(root, bg="#ff3333", height=60)
        self.warning_label = tk.Label(
            self.warning_frame,
            text="🚨 CẢNH BÁO VA CHẠM! 🚨",
            bg="#ff3333",
            fg="white",
            font=("Helvetica", 16, "bold")
        )
        self.warning_label.pack(expand=True, fill="both")

        # --- Thanh điều khiển ---
        control_frame = tk.Frame(root, bg="#2c2c2c", height=55)
        control_frame.grid(row=2, column=0, sticky="ew", pady=(3, 0))

        style = ttk.Style()
        style.configure(
            "Modern.TButton",
            font=("Helvetica", 12, "bold"),
            padding=6,
            background="#4dd0e1",
            foreground="#1e1e1e"
        )
        style.map("Modern.TButton",
                  background=[("active", "#64e5f3")])

        # --- Cụm nút bên trái ---
        left_controls = tk.Frame(control_frame, bg="#2c2c2c")
        left_controls.pack(side="left", padx=10)

        self.start_button = ttk.Button(left_controls, text="▶ Bắt đầu", style="Modern.TButton")
        self.stop_button = ttk.Button(left_controls, text="⏸ Tạm dừng", style="Modern.TButton")
        self.settings_button = ttk.Button(left_controls, text="⚙ Cài đặt", style="Modern.TButton")
        self.about_button = ttk.Button(left_controls, text="ℹ Giới thiệu", style="Modern.TButton")
        self.exit_button = ttk.Button(left_controls, text="❌ Thoát", style="Modern.TButton")

        for btn in [self.start_button, self.stop_button, self.settings_button, self.about_button, self.exit_button]:
            btn.pack(side="left", padx=6, ipadx=3)

        # --- Khu vực thông tin bên phải (để không bị trống) ---
        right_info = tk.Frame(control_frame, bg="#2c2c2c")
        right_info.pack(side="right", padx=15)
        tk.Label(
            right_info,
            text="Nhóm 1 - ĐH GTVT TP.HCM",
            bg="#2c2c2c",
            fg="#aaaaaa",
            font=("Helvetica", 10, "italic")
        ).pack(anchor="e")

        tk.Label(
            right_info,
            text="Phiên bản 1.0",
            bg="#2c2c2c",
            fg="#777777",
            font=("Helvetica", 9)
        ).pack(anchor="e")

        # --- Thanh trạng thái ---
        status_frame = tk.Frame(root, bg="#282828", height=30)
        status_frame.grid(row=3, column=0, sticky="ew")

        self.status_bar_label = tk.Label(
            status_frame,
            text="Sẵn sàng...",
            bg="#282828",
            fg="#77e8ff",  # xanh nhạt nhẹ, dễ nhìn
            font=("Helvetica", 10, "bold"),
            anchor="w",
            padx=12
        )
        self.status_bar_label.pack(fill="both")

        # --- Cấu hình hàng ---
        root.grid_rowconfigure(0, weight=0)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=0)
        root.grid_rowconfigure(3, weight=0)

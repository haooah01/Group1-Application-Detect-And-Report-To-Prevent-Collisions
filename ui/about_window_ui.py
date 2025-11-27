import tkinter as tk
import webbrowser

class AboutWindowUI:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("ℹ Giới thiệu ứng dụng")
        self.window.geometry("520x560")
        self.window.configure(bg="#1e1e1e")

        # --- Hàm tạo link ---
        def create_link(parent, text, url):
            link = tk.Label(parent, text=text, fg="#00aaff", bg="#1e1e1e",
                            font=("Arial", 10, "underline"), cursor="hand2",
                            wraplength=460, justify="left")
            link.pack(anchor="w", padx=25, pady=(0, 10))
            link.bind("<Button-1>", lambda e: webbrowser.open_new(url))

        # --- Tiêu đề ---
        tk.Label(self.window, text="VỀ ỨNG DỤNG",
                 fg="#00aaff", bg="#1e1e1e",
                 font=("Helvetica", 18, "bold")).pack(pady=(15, 5))

        # --- Canvas + Scrollbar ---
        canvas_frame = tk.Frame(self.window, bg="#1e1e1e")
        canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        canvas = tk.Canvas(canvas_frame, bg="#1e1e1e", highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1e1e1e")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        frame = scrollable_frame  # alias

        # --- Nội dung (chỉ chữ trắng) ---
        def section(title, content):
            tk.Label(frame, text=title, fg="#00ffff", bg="#1e1e1e",
                     font=("Arial", 10, "bold")).pack(anchor="w", pady=(8, 2))
            tk.Label(frame, text=content, fg="white", bg="#1e1e1e",
                     font=("Arial", 10), justify="left", wraplength=460).pack(anchor="w", padx=25)

        section("Đề tài:",
                 "Xây dựng ứng dụng nhận diện và cảnh báo nhằm ngăn ngừa va chạm.")
        section("Mô tả tổng quan:",
                 "Ứng dụng sử dụng thị giác máy tính (Computer Vision) kết hợp mô hình YOLOv8 "
                 "để phát hiện các phương tiện giao thông và người đi bộ. Thuật toán tính toán TTC "
                 "(Time To Collision) dựa trên kích thước bounding box giúp xác định nguy cơ va chạm "
                 "và phát cảnh báo thời gian thực bằng hình ảnh và âm thanh.")
        section("Nhóm thực hiện:", "Nhóm 1 – Đại học Giao thông Vận tải TP.HCM")
        section("Thành viên:",
                 "- Nguyễn Hoàng Tùng (Nhóm trưởng)\n"
                 "- Trần Thế Hảo\n"
                 "- Nguyễn Văn Hà\n"
                 "- Cao Xuân Cường\n"
                 "- Phạm Xuân Quỳnh")
        section("Giảng viên hướng dẫn:", "Thầy Trần Anh Quân")
        section("Công nghệ sử dụng:",
                 "• Mô hình AI: YOLOv8 (Ultralytics)\n"
                 "• Ngôn ngữ: Python\n"
                 "• Thư viện: OpenCV, Tkinter, Pygame\n"
                 "• Giao diện: Tkinter + ttk Style\n"
                 "• Hệ điều hành: Windows / Linux")

        tk.Label(frame, text="Liên hệ:",
                 fg="#00ffff", bg="#1e1e1e", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
        create_link(frame, "Email: tungnh2222@ut.edu.vn", "mailto:tungnh2222@ut.edu.vn")

        tk.Label(frame, text="Link GitHub:",
                 fg="#00ffff", bg="#1e1e1e", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
        create_link(frame,
                    "https://github.com/haooah01/Group1-Application-Detect-And-Report-To-Prevent-Collisions.git",
                    "https://github.com/haooah01/Group1-Application-Detect-And-Report-To-Prevent-Collisions.git")

        section("Ngày hoàn thành:", "Tháng 12 năm 2025")
        section("Phiên bản:", "v1.0 – Ứng dụng nhận diện và cảnh báo va chạm")

        # --- Footer ---
        tk.Label(self.window,
                 text="© 2025 - Nhóm 1 | Đại học GTVT TP.HCM",
                 fg="#777777", bg="#1e1e1e",
                 font=("Arial", 9), pady=6).pack(side="bottom")

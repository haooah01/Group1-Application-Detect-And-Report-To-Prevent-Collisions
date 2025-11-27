from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import os # Cần thiết để xây dựng đường dẫn file

# Import lớp ObjectDetector từ file detector.py
# (Giả sử app.py và processing/ nằm cùng cấp trong thư mục src/)
try:
    from processing.detector import ObjectDetector
except ImportError:
    print("\n[LỖI] Không thể import ObjectDetector. Hãy đảm bảo file 'backend/src/processing/detector.py' tồn tại.\n")
    exit()

# === 1. KHỞI TẠO CÁC THÀNH PHẦN ===

# --- Khởi tạo Ứng dụng FastAPI ---
app = FastAPI(
    title="Collision Warning System API",
    description="API cho hệ thống cảnh báo va chạm sử dụng YOLOv8 và TTC.",
    version="0.1.0"
)

# --- Xác định đường dẫn ---
# Giả định rằng chúng ta chạy app.py từ thư mục gốc (collision_warning_system/)
# Bằng lệnh: python backend/src/app.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Trỏ về thư mục 'backend/'
ROOT_DIR = os.path.dirname(BASE_DIR) # Trỏ về thư mục 'collision_warning_system/'

VIDEO_PATH = os.path.join(ROOT_DIR, "data", "videos", "test_video.mp4")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolov8n.pt") # Đường dẫn tới model nếu bạn lưu riêng

# --- Tải Mô hình AI ---
# Tải mô hình 1 LẦN DUY NHẤT khi server khởi động
# Chúng ta sẽ dùng model yolov8n.pt mặc định mà thư viện tự tải
try:
    detector = ObjectDetector() # Dùng model mặc định 'yolov8n.pt'
except Exception as e:
    print(f"\n[LỖI] Không thể tải mô hình YOLO. Lỗi: {e}\n")
    exit()

# --- Cấu hình Giao diện (Frontend) ---
# Trỏ đến thư mục 'templates' của frontend
templates = Jinja2Templates(directory=os.path.join(ROOT_DIR, "frontend", "templates"))
# Phục vụ các file tĩnh (CSS, JS, Images) từ thư mục 'static' của frontend
app.mount("/static", 
          StaticFiles(directory=os.path.join(ROOT_DIR, "frontend", "static")), 
          name="static")


# === 2. ĐỊNH NGHĨA HÀM XỬ LÝ VIDEO ===

def video_stream_generator():
    """
    Hàm generator này sẽ đọc video, xử lý từng frame và 'yield' (trả về) 
    frame đó dưới dạng byte JPEG.
    """
    if not os.path.exists(VIDEO_PATH):
        print(f"[LỖI] Không tìm thấy video tại: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở file video: {VIDEO_PATH}")
        return

    print("\nBắt đầu xử lý và truyền video...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # Hết video, quay lại từ đầu
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. Đưa frame qua mô hình AI
        results = detector.detect_and_track(frame)
        
        # 2. Vẽ kết quả (bounding box, ID) lên frame
        annotated_frame = detector.draw_results(frame, results)

        # 3. Mã hóa frame thành JPEG
        # .jpg giúp nén ảnh, giảm băng thông truyền
        (flag, encoded_image) = cv2.imencode(".jpg", annotated_frame)
        if not flag:
            continue

        # 4. Yield frame dưới dạng byte cho response
        # Đây là định dạng chuẩn của multipart/x-mixed-replace
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

    cap.release()
    print("Dừng truyền video.")


# === 3. TẠO CÁC API ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Endpoint này phục vụ file index.html từ thư mục templates.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_stream")
def video_stream():
    """
    Endpoint này trả về luồng video đã được xử lý.
    Trình duyệt sẽ tự động nhận diện và hiển thị.
    """
    # Trả về một StreamingResponse, nội dung là hàm generator ở trên
    return StreamingResponse(
        video_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# --- Phần này để chạy server (giống như trước) ---
if __name__ == "__main__":
    print(f"Server sẽ chạy từ thư mục gốc: {ROOT_DIR}")
    print(f"Video sẽ được tải từ: {VIDEO_PATH}")
    print(f"Giao diện sẽ được tải từ: {os.path.join(ROOT_DIR, 'frontend', 'templates')}")
    print("\nKhởi chạy server Uvicorn tại http://127.0.0.1:8000")
    
    # Chạy uvicorn. 'app:app' nghĩa là "file 'app.py', đối tượng 'app'"
    # reload=True giúp server tự khởi động lại khi bạn sửa code
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
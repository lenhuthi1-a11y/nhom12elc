# recognize_faces.py

import cv2
import os
import pickle
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis  # ĐỔI: dùng InsightFace thay cho DeepFace

# ==== CẤU HÌNH ====
THU_MUC_EMB = "app/data/embeddings/du_lieu_khuon_mat.pkl"
KET_QUA_DIEM_DANH = "app/data/sessions/danh_sach_diem_danh.csv"
NGUONG_NHAN_DIEN = 0.4
CAM_INDEX = 0

# ==== TẢI DỮ LIỆU HUẤN LUYỆN ====
with open(THU_MUC_EMB, "rb") as f:
    du_lieu = pickle.load(f)

ma_sinh_vien = du_lieu["ma_sv"]
vector_khuon_mat = np.array(du_lieu["vector"])  # embeddings InsightFace (đã chuẩn hoá)

# Nếu có danh sách tên sinh viên (file CSV gốc)
CSV_PATH = "app/data/danhsach.csv"
ten_sinh_vien = {}
if os.path.exists(CSV_PATH):
    import csv
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ten_sinh_vien[row["ma_sv"]] = row["ho_ten"]

print(f"✅ Đã nạp {len(ma_sinh_vien)} khuôn mặt để nhận diện.")

# ==== HÀM TÍNH KHOẢNG CÁCH (Cosine) ====
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
    return float(np.dot(v1, v2) / denom)

# ==== KHỞI TẠO INSIGHTFACE ====
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # dùng CPU; có GPU NVIDIA thì đổi sang CUDAExecutionProvider
app.prepare(ctx_id=0, det_size=(640, 640))

# ==== MỞ CAMERA ====
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ Không mở được camera. Kiểm tra lại thiết bị.")

print("\n📷 Camera đang chạy... Nhấn Q để thoát.")
print("Khi hệ thống nhận diện được khuôn mặt, sẽ hiển thị TÊN sinh viên.\n")

# ==== LƯU KẾT QUẢ ====
da_diem_danh = {}  # tránh trùng lặp

# ==== VÒNG LẶP CHÍNH ====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # ĐỔI: dùng InsightFace để phát hiện & trích xuất embedding
        faces = app.get(frame)

        for face in faces:
            emb = getattr(face, "embedding", None)
            if emb is None:
                continue  # không có embedding thì bỏ qua

            # Tính độ tương đồng (cosine similarity) với tất cả vector đã huấn luyện
            similarities = [cosine_similarity(emb, emb_ref) for emb_ref in vector_khuon_mat]
            index_max = int(np.argmax(similarities)) if similarities else -1
            if index_max == -1:
                continue

            do_tuong_dong = similarities[index_max]

            # Xác định sinh viên hoặc Unknown
            if do_tuong_dong >= (1.0 - NGUONG_NHAN_DIEN):
                ma_sv = ma_sinh_vien[index_max]
                ho_ten = ten_sinh_vien.get(ma_sv, "Không rõ tên")
            else:
                ma_sv = "Unknown"
                ho_ten = "Unknown"

            # Vẽ khung quanh khuôn mặt (đổi bbox theo InsightFace)
            bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox.tolist()
            color = (0, 255, 0) if ma_sv != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, ho_ten, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Ghi nhận điểm danh nếu là sinh viên hợp lệ
            if ma_sv != "Unknown" and ma_sv not in da_diem_danh:
                now = datetime.now().strftime("%H:%M:%S")
                da_diem_danh[ma_sv] = {"ho_ten": ho_ten, "gio_vao": now}
                print(f"✅ {ho_ten} ({ma_sv}) | {now}")

    except Exception:
        pass  # giữ nguyên hành vi im lặng khi lỗi khung hình

    cv2.imshow("DIEM DANH - InsightFace", frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()

# ==== LƯU FILE CSV ====
os.makedirs(os.path.dirname(KET_QUA_DIEM_DANH), exist_ok=True)
with open(KET_QUA_DIEM_DANH, "w", encoding="utf-8") as f:
    f.write("ma_sv,ho_ten,gio_vao\n")
    for ma_sv, data in da_diem_danh.items():
        f.write(f"{ma_sv},{data['ho_ten']},{data['gio_vao']}\n")

print(f"\n🎓 Điểm danh hoàn tất! Kết quả lưu tại: {KET_QUA_DIEM_DANH}")

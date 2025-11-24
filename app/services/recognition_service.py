#recognition_service
import os
import pickle
import numpy as np
import csv
from insightface.app import FaceAnalysis
from django.conf import settings


class RecognitionService:

    def __init__(self):
        print("Đang tải mô hình InsightFace (ArcFace)...")
        # --- Lấy logic từ recognize_faces.py ---
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        print("Đang tải dữ liệu huấn luyện (embeddings)...")
        emb_path = os.path.join(settings.BASE_DIR, "app", "data", "embeddings", "du_lieu_khuon_mat.pkl")

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Không tìm thấy file embedding tại: {emb_path}")

        with open(emb_path, "rb") as f:
            du_lieu = pickle.load(f)

        self.ma_sinh_vien = du_lieu["ma_sv"]
        self.vector_khuon_mat = np.array(du_lieu["vector"])

        # --- Lấy logic từ recognize_faces.py ---
        self.ten_sinh_vien = {}
        csv_path = os.path.join(settings.BASE_DIR, "app", "data", "danhsach.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.ten_sinh_vien[row["ma_sv"]] = row["ho_ten"]

        print(f"✅ Đã nạp {len(self.ma_sinh_vien)} khuôn mặt. Dịch vụ sẵn sàng.")

    def cosine_similarity(self, vec1, vec2):
        # Embedding của insightface đã được chuẩn hóa L2,
        # nên cosine similarity chính là tích vô hướng (dot product).
        return float(np.dot(vec1, vec2))

    def recognize(self, frame_bgr, nguong_nhan_dien=0.4):
        """
        Nhận diện khuôn mặt từ một khung hình (ảnh BGR của OpenCV).
        Trả về một list các kết quả.
        """
        results = []
        try:
            # --- Lấy logic từ recognize_faces.py ---
            faces = self.app.get(frame_bgr)

            for face in faces:
                emb = getattr(face, "embedding", None)
                if emb is None:
                    continue

                similarities = [self.cosine_similarity(emb, emb_ref) for emb_ref in self.vector_khuon_mat]
                index_max = int(np.argmax(similarities)) if similarities else -1
                if index_max == -1:
                    continue

                do_tuong_dong = similarities[index_max]
                bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]

                # --- Lấy logic từ recognize_faces.py ---
                # Giữ nguyên logic ngưỡng của bạn
                if do_tuong_dong >= (1.0 - nguong_nhan_dien):
                    ma_sv = self.ma_sinh_vien[index_max]
                    ho_ten = self.ten_sinh_vien.get(ma_sv, "Không rõ tên")
                else:
                    ma_sv = "Unknown"
                    ho_ten = "Unknown"

                results.append({
                    "ma_sv": ma_sv,
                    "ho_ten": ho_ten,
                    "do_tuong_dong": do_tuong_dong,
                    "bbox": bbox
                })
        except Exception as e:
            print(f"Lỗi nhận diện: {e}")

        return results


# TẢI MODEL 1 LẦN DUY NHẤT:
# Tạo một instance toàn cục của service, nó sẽ tải model khi Django khởi động
recognition_service = RecognitionService()


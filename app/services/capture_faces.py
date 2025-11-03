#capture_faces.py -- tải ảnh lên
import cv2
import os
import csv
import shutil
from glob import glob
import hashlib
import time

CSV_PATH = "app/data/danhsach.csv"
RAW_DIR  = "app/data/faces_raw"
CAM_INDEX = 0                       # đổi nếu máy có nhiều camera
NUM_PHOTOS_DEFAULT = 20             # mặc định số ảnh cần chụp / SV
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INTERVAL_SEC = 0.5                  # ~0.5 giây / 1 tấm

# ---------- Tiện ích ----------
def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {CSV_PATH}")

def read_roster(csv_path):
    roster = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ma_sv" in row and "ho_ten" in row and row["ma_sv"]:
                roster.append({"ma_sv": row["ma_sv"].strip(), "ho_ten": row["ho_ten"].strip()})
    if not roster:
        raise ValueError("CSV rỗng hoặc sai cột. Cần có header: ma_sv,ho_ten")
    return roster

def next_filename(folder, index):
    return os.path.join(folder, f"{index:02d}.jpg")

def is_img(path):
    _, ext = os.path.splitext(path.lower())
    return ext in VALID_EXTS

def print_header_capture():
    print("📂 CHẾ ĐỘ: CHỤP ẢNH CHO SINH VIÊN")
    print("===============================================")

def print_roster(roster):
    print("📁 DANH SÁCH SINH VIÊN:")
    for i, r in enumerate(roster, start=1):
        print(f"  {i:02d}. {r['ho_ten']} ({r['ma_sv']})")

def choose_students(roster, purpose_hint="chụp"):
    print_header_capture()
    print_roster(roster)
    print(f"\n👉 Nhập số thứ tự sinh viên muốn {purpose_hint} lên (vd: 1,3,5-8) hoặc gõ 'all' để chọn tất cả.")
    choice = input("→ Lựa chọn: ").strip().lower()

    if choice == "all":
        selected = roster[:]
    else:
        def expand_tokens(tokens):
            idxs = set()
            for tok in tokens:
                tok = tok.strip()
                if not tok:
                    continue
                if "-" in tok:
                    try:
                        a, b = tok.split("-", 1)
                        a, b = int(a), int(b)
                        if a > b:
                            a, b = b, a
                        for k in range(a, b + 1):
                            idxs.add(k)
                    except Exception:
                        pass
                else:
                    if tok.isdigit():
                        idxs.add(int(tok))
            return sorted([i for i in idxs if 1 <= i <= len(roster)])

        indices = expand_tokens(choice.split(","))
        selected = [roster[i - 1] for i in indices]

    if not selected:
        print("❌ Lựa chọn không hợp lệ. Không có sinh viên nào được chọn.")
        return []

    print("\n✅ ĐÃ CHỌN:")
    for sv in selected:
        print(f"   - {sv['ho_ten']} ({sv['ma_sv']})")
    print("===============================================")
    return selected

def ask_int(prompt, default=None, min_val=1, max_val=9999):
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return default
        if s.isdigit():
            val = int(s)
            if val < (min_val or 1):
                print(f"⚠️  Giá trị tối thiểu là {min_val}.")
                continue
            if max_val and val > max_val:
                print(f"⚠️  Giá trị tối đa là {max_val}.")
                continue
            return val
        print("⚠️  Vui lòng nhập số hợp lệ.")

# ---------- Chế độ 1: Chụp camera (tự động 0.5s/tấm) ----------
def capture_mode(roster):
    selected = choose_students(roster, purpose_hint="chụp")
    if not selected:
        return

    num_photos = ask_int(
        prompt=f"\n📸 Nhập số tấm muốn chụp cho mỗi sinh viên (Enter = {NUM_PHOTOS_DEFAULT}): ",
        default=NUM_PHOTOS_DEFAULT,
        min_val=1,
        max_val=500
    )

    print(f"👉 Tốc độ chụp đặt trước: ~{INTERVAL_SEC:.1f}s / tấm")
    print("👉 Tiến hành chụp ảnh -> nhấn 1 để bắt đầu , nhấn 2 để thoát")
    go = input("→ Chọn (1/2): ").strip()
    if go != "1":
        print("🛑 Đã thoát chế độ chụp.")
        return

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("❌ Không mở được camera. Kiểm tra thiết bị / quyền truy cập.")

    for row in selected:
        ma_sv  = row["ma_sv"]
        ho_ten = row["ho_ten"]
        save_dir = os.path.join(RAW_DIR, ma_sv)
        os.makedirs(save_dir, exist_ok=True)

        # tiếp tục đánh số nếu đã có ảnh cũ
        existing = sorted(glob(os.path.join(save_dir, "*.jpg")))
        count = len(existing)
        target = num_photos
        print(f"\n➡️ Tự động chụp cho: {ho_ten} ({ma_sv}) | đã có {count}/{target}")

        last_shot = 0.0
        start_time = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ Không lấy được khung hình từ camera.")
                break

            now = time.time()
            # Điều kiện auto-chụp mỗi 0.5s
            if count < target and (now - last_shot) >= INTERVAL_SEC:
                count += 1
                cv2.imwrite(next_filename(save_dir, count), frame)
                last_shot = now
                print(f"✅ Đã chụp {count}/{target} ảnh → {save_dir}")

            # Overlay trạng thái lên khung hình
            elapsed = now - start_time
            remain = max(target - count, 0)
            est_left = remain * INTERVAL_SEC
            status = f"{ho_ten}  |  {count}/{target} tấm  |  ~{INTERVAL_SEC:.1f}s/tấm  |  còn ~{est_left:.1f}s"
            disp = frame.copy()
            cv2.rectangle(disp, (8, 8), (8 + 690, 46), (0, 0, 0), -1)
            cv2.putText(disp, status, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # gợi ý phím nóng
            cv2.putText(disp, "Q: bo qua SV | N: SV tiep | ESC: thoat", (16, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Dang chup (Auto 0.5s/tam)", disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):  # bỏ qua SV này
                print(f"⏭  Bỏ qua: {ho_ten} (đang {count}/{target})")
                break
            if key in (ord('n'), ord('N')):  # chuyển sang SV tiếp theo
                print(f"⏭  Chuyển tiếp: {ho_ten} (đang {count}/{target})")
                break
            if key == 27:  # ESC
                print("🛑 Thoát chụp.")
                cap.release()
                cv2.destroyAllWindows()
                return

            if count >= target:
                print(f"✔️ Hoàn tất: {ho_ten} ({count} ảnh) → {save_dir}")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎉 Đã chụp xong! Ảnh đã được lưu tại app/data/faces_raw/<ma_sv>/")

# ---------- Chế độ 1: Chụp camera ----------
def capture_mode(roster):
    # 1) Chọn sinh viên
    selected = choose_students(roster, purpose_hint="chụp")
    if not selected:
        return

    # 2) Hỏi số tấm muốn chụp / SV
    num_photos = ask_int(
        prompt=f"\n📸 Tiến hành lựa chọn số tấm muốn chụp cho mỗi sinh viên (Enter = {NUM_PHOTOS_DEFAULT}): ",
        default=NUM_PHOTOS_DEFAULT,
        min_val=1,
        max_val=500
    )

    # 3) Hỏi bắt đầu hay thoát
    print("👉 Tiến hành chụp ảnh -> nhấn 1 để bắt đầu , nhấn 2 để thoát")
    go = input("→ Chọn (1/2): ").strip()
    if go != "1":
        print("🛑 Đã thoát chế độ chụp.")
        return

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("❌ Không mở được camera. Kiểm tra thiết bị / quyền truy cập.")

    for row in selected:
        ma_sv  = row["ma_sv"]
        ho_ten = row["ho_ten"]
        save_dir = os.path.join(RAW_DIR, ma_sv)
        os.makedirs(save_dir, exist_ok=True)

        # tính số ảnh đã có (nếu chạy lại, tiếp tục đánh số)
        existing = sorted(glob(os.path.join(save_dir, "*.jpg")))
        count = len(existing)
        print(f"\n➡️ Đang chụp cho: {ho_ten} ({ma_sv}) | đã có {count}/{num_photos}")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ Không lấy được khung hình từ camera.")
                break

            disp = frame.copy()
            cv2.putText(
                disp,
                f"{ho_ten} ({count}/{num_photos}) - SPACE: chup | N: SV tiep | Q: bo qua | ESC: thoat",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 120, 120),
                2
            )
            cv2.imshow("Dang chup", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                if count < num_photos:
                    count += 1
                    cv2.imwrite(next_filename(save_dir, count), frame)
                    print(f"✅ Da chup {count}/{num_photos} anh")
                if count >= num_photos:
                    print(f"✔️ Hoàn tất: {ho_ten} ({count} ảnh)")
                    break
            elif key in (ord('n'), ord('N')):  # chuyển sang SV tiếp theo
                print(f"⏭  Chuyển tiếp: {ho_ten} (đang {count}/{num_photos})")
                break
            elif key in (ord('q'), ord('Q')):  # bỏ qua SV này
                print(f"⏭  Bỏ qua: {ho_ten} (đang {count}/{num_photos})")
                break
            elif key == 27:  # ESC: thoát toàn bộ chụp
                print("🛑 Thoát chụp.")
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎉 Đã chụp xong!")

# ---------- Chế độ 2: Tải ảnh có sẵn ----------
# --- Hàm tiện ích: băm MD5 để phát hiện ảnh trùng nội dung ---
def md5sum(path, chunk=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def import_mode(roster):
    print("\n📂 CHẾ ĐỘ: NHẬP ẢNH CÓ SẴN")
    print("===============================================")
    print("📁 DANH SÁCH SINH VIÊN:")
    for i, row in enumerate(roster, start=1):
        print(f"  {i:02d}. {row['ho_ten']} ({row['ma_sv']})")

    print("\n👉 Nhập số thứ tự sinh viên muốn tải lên (vd: 1,3,5) hoặc gõ 'all' để chọn tất cả.")
    choice = input("→ Lựa chọn: ").strip().lower()

    # --- Xác định sinh viên được chọn ---
    selected = []
    if choice == "all":
        selected = roster
    else:
        try:
            indices = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
            selected = [roster[i - 1] for i in indices if 1 <= i <= len(roster)]
        except Exception:
            print("❌ Lựa chọn không hợp lệ. Dừng nhập ảnh.")
            return

    if not selected:
        print("❌ Không có sinh viên nào được chọn.")
        return

    print("\n✅ ĐÃ CHỌN:")
    for sv in selected:
        print(f"   - {sv['ho_ten']} ({sv['ma_sv']})")
    print("===============================================")

    # --- Nhập ảnh cho từng sinh viên ---
    for row in selected:
        ma_sv  = row["ma_sv"]
        ho_ten = row["ho_ten"]

        print(f"\n📸 {ho_ten} ({ma_sv})")
        src_folder = input(f"👉 Nhập đường dẫn thư mục chứa ảnh của {ho_ten}: ").strip().strip('"')
        if not src_folder:
            print(f"⏭  Bỏ qua: {ho_ten}")
            continue
        if not os.path.isdir(src_folder):
            print(f"❌ Thư mục không tồn tại: {src_folder}")
            continue

        # Gom ảnh hợp lệ từ thư mục nguồn
        all_files = [os.path.join(src_folder, f) for f in os.listdir(src_folder)]
        all_imgs = [p for p in all_files if is_img(p)]
        if not all_imgs:
            print(f"⚠️ Không tìm thấy ảnh hợp lệ trong {src_folder}")
            continue

        # Thư mục đích
        dst_dir = os.path.join(RAW_DIR, ma_sv)
        os.makedirs(dst_dir, exist_ok=True)

        # Nếu thư mục sinh viên đã tồn tại và có ảnh → hỏi người dùng
        if os.listdir(dst_dir):
            print(f"⚠️ Thư mục '{dst_dir}' đã có ảnh sẵn.")
            ans = input("👉 Bạn có muốn xoá ảnh cũ và tải lại không? (y/n): ").strip().lower()
            if ans == "y":
                shutil.rmtree(dst_dir)
                os.makedirs(dst_dir, exist_ok=True)
                print("✅ Đã xoá ảnh cũ, sẵn sàng nhập mới.")
            else:
                print("⏭ Giữ ảnh cũ, bỏ qua sinh viên này.")
                continue

        # Sao chép ảnh hợp lệ sang thư mục đích
        copied = 0
        for idx, src in enumerate(all_imgs, start=1):
            ext = os.path.splitext(src)[1].lower()
            dst = os.path.join(dst_dir, f"{idx:02d}{ext}")
            shutil.copy2(src, dst)
            copied += 1

        print(f"✔️  Đã tải {copied} ảnh cho {ho_ten} → {dst_dir}")

    print("\n🎉 Hoàn tất nhập ảnh cho sinh viên đã chọn!")


if __name__ == "__main__":
    ensure_dirs()
    roster = read_roster(CSV_PATH)

    print("\n📸 CHẾ ĐỘ GHI DANH KHUÔN MẶT")
    print("1️⃣  Chụp ảnh qua camera")
    print("2️⃣  Tải ảnh có sẵn từ thư mục (đã chụp sẵn)")
    mode = input("→ Chọn chế độ (1 hoặc 2): ").strip()

    if mode == "1":
        capture_mode(roster)
    elif mode == "2":
        import_mode(roster)
    else:
        print("❌ Lựa chọn không hợp lệ.")
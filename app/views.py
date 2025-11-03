from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import json
import base64
import numpy as np
import cv2
from datetime import timedelta, datetime
import csv

from .services.recognition_service import recognition_service
from .models import AttendanceSession, AttendanceLog


def setup_session(request):
    if request.method == "POST":
        subject_name = request.POST.get('subject_name')
        subject_code = request.POST.get('subject_code')
        class_name = request.POST.get('class_name')
        duration_minutes = int(request.POST.get('duration', 60))

        # Lấy 'Giờ vào lớp' từ form
        class_start_time_str = request.POST.get('class_start_time')
        class_start_time = timezone.make_aware(
            datetime.strptime(class_start_time_str, '%Y-%m-%dT%H:%M')
        )

        # Giờ mở camera (session_start_time) là NGAY BÂY GIỜ
        session_start_time = timezone.now()
        # Giờ kết thúc = Giờ mở camera + thời lượng
        end_time = session_start_time + timedelta(minutes=duration_minutes)

        # Tạo phiên mới trong database
        session = AttendanceSession.objects.create(
            subject_name=subject_name,
            subject_code=subject_code,
            class_name=class_name,
            class_start_time=class_start_time,  # Lưu giờ vào lớp (để tính trễ)
            session_start_time=session_start_time,  # Lưu giờ mở camera (là bây giờ)
            end_time=end_time,  # Lưu giờ đóng camera
            is_active=True
        )
        return redirect('run_session', session_id=session.id)

    return render(request, "setup.html")


def run_session(request, session_id):
    """Hiển thị trang camera điểm danh (index.html) cho một phiên cụ thể."""
    session = get_object_or_404(AttendanceSession, id=session_id)
    logs = session.logs.all().order_by('-gio_vao')

    context = {
        "session": session,
        "logs": logs
    }
    # Lưu ý: tệp 'index.html' dùng template filter {{ log.gio_vao|date:... }}
    # nên nó đã tự động chuyển múi giờ đúng, không cần sửa ở đây.
    return render(request, "index.html", context)


@csrf_exempt
def recognize_api(request):
    """API endpoint để nhận ảnh, nhận diện và lưu vào DB."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
        image_data = data.get('image')
        session_id = data.get('session_id')

        if not image_data or not session_id:
            return JsonResponse({"error": "Missing image or session_id"}, status=400)

        try:
            session = AttendanceSession.objects.get(id=session_id, is_active=True)
        except AttendanceSession.DoesNotExist:
            return JsonResponse({"error": "Phiên điểm danh không hợp lệ."}, status=404)

        current_time = timezone.now()

        # Check 1: Kiểm tra giờ kết thúc
        if current_time > session.end_time:
            session.is_active = False
            session.save()

            # === SỬA LỖI MÚI GIỜ 1 ===
            # Chuyển giờ kết thúc sang giờ địa phương
            local_end_time = timezone.localtime(session.end_time)
            return JsonResponse({"error": f"Phiên điểm danh đã kết thúc lúc {local_end_time.strftime('%H:%M')}"},
                                status=410)

        # --- Xử lý ảnh ---
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        results = recognition_service.recognize(frame)

        new_logs = []
        for res in results:
            if res["ma_sv"] != "Unknown":

                status = "Đúng giờ"
                if current_time > session.class_start_time:
                    status = "Trễ giờ"

                # --- Lưu vào DB ---
                log, created = AttendanceLog.objects.get_or_create(
                    session=session,
                    ma_sv=res["ma_sv"],
                    defaults={
                        "ho_ten": res["ho_ten"],
                        "gio_vao": current_time,  # Lưu giờ UTC (đã đúng)
                        "status": status
                    }
                )
                if created:  # Nếu là SV mới
                    # === SỬA LỖI MÚI GIỜ 2 ===
                    # Chuyển giờ vào sang giờ địa phương trước khi gửi về JSON
                    local_gio_vao = timezone.localtime(log.gio_vao)
                    res["gio_vao"] = local_gio_vao.strftime("%d/%m/%Y, %H:%M:%S")
                    res["status"] = log.status
                    new_logs.append(res)
            else:
                new_logs.append(res)

        return JsonResponse({"results": new_logs})

    except Exception as e:
        print(f"Lỗi API: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def export_session_csv(request, session_id):
    """
    Xử lý việc xuất dữ liệu điểm danh ra tệp CSV.
    """
    session = get_object_or_404(AttendanceSession, id=session_id)

    # === SỬA LỖI MÚI GIỜ 3 (Phần 1) ===
    # Chuyển giờ bắt đầu (để đặt tên tệp) sang giờ địa phương
    local_start_time = timezone.localtime(session.class_start_time)

    # 1. Tạo tên tệp theo yêu cầu
    ngay_str = local_start_time.strftime('%d-%m-%Y')
    gio_str = local_start_time.strftime('%Hh%M')
    mon_str = session.subject_name.replace(' ', '_')
    lop_str = session.class_name.replace(' ', '_')

    filename = f"Diemdanh_{ngay_str}_{mon_str}_{lop_str}_{gio_str}.csv"

    # 2. Tạo HttpResponse
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'

    # 3. Ghi BOM (cho Excel đọc tiếng Việt)
    response.write(u'\ufeff')
    writer = csv.writer(response)

    # 4. Ghi Header
    writer.writerow(['MSSV', 'Ho Ten', 'Gio Vao (VN)', 'Trang Thai'])

    logs = session.logs.all().order_by('ho_ten')
    for log in logs:
        # === SỬA LỖI MÚI GIỜ 3 (Phần 2) ===
        # Chuyển giờ của từng sinh viên sang giờ địa phương
        local_gio_vao = timezone.localtime(log.gio_vao)
        writer.writerow([
            log.ma_sv,
            log.ho_ten,
            local_gio_vao.strftime('%d/%m/%Y %H:%M:%S'),  # In giờ địa phương
            log.status
        ])

    return response


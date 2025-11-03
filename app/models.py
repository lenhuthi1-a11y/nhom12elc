from django.db import models
from django.utils import timezone


class AttendanceSession(models.Model):
    subject_name = models.CharField("Tên môn học", max_length=255)
    subject_code = models.CharField("Mã môn học", max_length=50)
    class_name = models.CharField("Tên lớp", max_length=100)
    class_start_time = models.DateTimeField("Giờ vào lớp (tính trễ)")
    session_start_time = models.DateTimeField("Giờ bắt đầu điểm danh (mở camera)")
    end_time = models.DateTimeField("Giờ kết thúc điểm danh")
    is_active = models.BooleanField(default=True)

    def __str__(self):
        # === SỬA LỖI MÚI GIỜ ===
        # Chuyển đổi sang giờ địa phương trước khi định dạng
        local_start_time = timezone.localtime(self.class_start_time)
        return f"{self.subject_name} ({self.class_name}) - {local_start_time.strftime('%d/%m/%Y %H:%M')}"

class AttendanceLog(models.Model):
    STATUS_CHOICES = (
        ('Đúng giờ', 'Đúng giờ'),
        ('Trễ giờ', 'Trễ giờ'),)
    session = models.ForeignKey(AttendanceSession, on_delete=models.CASCADE, related_name="logs")
    ma_sv = models.CharField(max_length=50)
    ho_ten = models.CharField(max_length=255)
    gio_vao = models.DateTimeField("Giờ vào")
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Đúng giờ')

    class Meta:
        unique_together = ('session', 'ma_sv')

    def __str__(self):

        local_gio_vao = timezone.localtime(self.gio_vao)
        return f"{self.ho_ten} ({self.ma_sv}) - {local_gio_vao.strftime('%d/%m/%Y %H:%M:%S')} - {self.status}"


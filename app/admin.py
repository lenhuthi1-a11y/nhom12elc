from django.contrib import admin
from .models import AttendanceSession, AttendanceLog


# Tùy chỉnh cách hiển thị cho Phiên điểm danh
class AttendanceSessionAdmin(admin.ModelAdmin):
    list_display = (
        'subject_name',
        'class_name',
        'class_start_time',
        'end_time',
        'is_active'
    )
    list_filter = ('class_start_time', 'is_active', 'subject_name')
    search_fields = ('subject_name', 'subject_code', 'class_name')


# Tùy chỉnh cách hiển thị cho Nhật ký điểm danh
class AttendanceLogAdmin(admin.ModelAdmin):
    list_display = (
        'ho_ten',
        'ma_sv',
        'gio_vao',
        'status',
        'session'
    )
    list_filter = ('status', 'session__subject_name')  # Lọc theo trạng thái và tên môn học
    search_fields = ('ho_ten', 'ma_sv')

    # Giúp CSDL tải nhanh hơn khi lấy tên 'session'
    list_select_related = ('session',)


# Đăng ký 2 bảng vào trang admin
admin.site.register(AttendanceSession, AttendanceSessionAdmin)
admin.site.register(AttendanceLog, AttendanceLogAdmin)

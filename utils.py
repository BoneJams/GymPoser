"""
Các hàm tiện ích: tính góc, vẽ text/đường lên frame.
"""

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# HÌNH HỌC
# ----------------------------------------------------------------------------
def find_angle(p1, p2, ref_pt=np.array([0, 0])):
    """
    Góc (độ) tại điểm tham chiếu ref_pt, tạo bởi hai vector ref->p1 và ref->p2.

        theta = arccos( (P1ref . P2ref) / (|P1ref| * |P2ref|) )

    Để lấy góc của một đoạn thẳng so với PHƯƠNG ĐỨNG, ta truyền:
        p1     = đầu kia của đoạn (vd: vai)
        ref_pt = gốc của đoạn      (vd: hông)
        p2     = (ref_x, 0)        -> điểm nằm trên đường thẳng đứng qua ref
    """
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    denom = (np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref)) + 1e-9
    cos_theta = np.dot(p1_ref, p2_ref) / denom
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return int(180.0 / np.pi * theta)


def get_landmark_xy(landmarks, idx, frame_w, frame_h):
    """Trả về toạ độ pixel (x, y) của 1 landmark MediaPipe."""
    lm = landmarks[idx]
    return np.array([int(lm.x * frame_w), int(lm.y * frame_h)])


# ----------------------------------------------------------------------------
# VẼ
# ----------------------------------------------------------------------------
def draw_text(
    img, text, pos=(0, 0),
    font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, thickness=2,
    text_color=(255, 255, 255), box_color=(0, 0, 0), box_alpha=0.6,
):
    """Vẽ chữ có nền mờ phía sau cho dễ đọc."""
    x, y = pos
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - 4, y - th - 6),
        (x + tw + 4, y + base + 2),
        box_color, -1,
    )
    cv2.addWeighted(overlay, box_alpha, img, 1 - box_alpha, 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return th + base + 8  # chiều cao đã dùng, để xếp dòng tiếp theo


def draw_dotted_line(img, start, end, color=(0, 255, 255), thickness=2, gap=8):
    """Vẽ đường nét đứt từ start đến end."""
    dist = int(np.hypot(end[0] - start[0], end[1] - start[1]))
    if dist == 0:
        return
    for i in range(0, dist, gap * 2):
        r = i / dist
        x = int(start[0] + (end[0] - start[0]) * r)
        y = int(start[1] + (end[1] - start[1]) * r)
        cv2.circle(img, (x, y), thickness, color, -1)

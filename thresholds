"""
Ngưỡng (thresholds) cho ứng dụng phân tích squat.

Tất cả các góc đều tính BẰNG ĐỘ và là góc giữa một đoạn thẳng trên cơ thể
so với phương THẲNG ĐỨNG (vertical), theo đúng cách làm của bài LearnOpenCV.

  - HIP_KNEE_VERT : các khoảng góc đầu gối-phương đứng để xác định trạng thái s1/s2/s3
  - HIP_THRESH    : [thấp, cao] góc thân (vai-hông) so với phương đứng
                    -> dùng cho feedback "ngả người ra sau / cúi người về trước"
  - KNEE_THRESH   : [a, b, c] mốc góc đầu gối; >c là squat quá sâu
  - ANKLE_THRESH  : nếu góc cẳng chân (gối-cổ chân) vượt ngưỡng -> đầu gối vượt mũi chân
  - OFFSET_THRESH : nếu góc lệch vai-mũi vượt ngưỡng -> người đang quay mặt ra trước cam
  - INACTIVE_THRESH: số giây không hoạt động -> reset bộ đếm
  - CNT_FRAME_THRESH: số frame giữ thông báo feedback trên màn hình

Toàn bộ ngưỡng đặt theo kinh nghiệm (heuristic) — cứ chỉnh lại cho hợp dáng/khung hình của bạn.
"""


def get_thresholds_beginner():
    angle_hip_knee_vert = {
        "NORMAL": (0, 32),
        "TRANS":  (35, 65),
        "PASS":   (70, 95),
    }

    thresholds = {
        "HIP_KNEE_VERT": angle_hip_knee_vert,

        "HIP_THRESH":   [10, 50],
        "ANKLE_THRESH": 45,
        "KNEE_THRESH":  [50, 70, 95],

        "OFFSET_THRESH":   35.0,
        "INACTIVE_THRESH": 15.0,

        "CNT_FRAME_THRESH": 50,
    }
    return thresholds


def get_thresholds_pro():
    angle_hip_knee_vert = {
        "NORMAL": (0, 32),
        "TRANS":  (35, 65),
        "PASS":   (80, 95),
    }

    thresholds = {
        "HIP_KNEE_VERT": angle_hip_knee_vert,

        "HIP_THRESH":   [15, 50],
        "ANKLE_THRESH": 30,
        "KNEE_THRESH":  [50, 80, 95],

        "OFFSET_THRESH":   35.0,
        "INACTIVE_THRESH": 15.0,

        "CNT_FRAME_THRESH": 50,
    }
    return thresholds

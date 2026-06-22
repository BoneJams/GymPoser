"""
ProcessFrame: lõi phân tích squat.

Quy trình mỗi frame (theo flowchart của LearnOpenCV):
  1. Lấy landmark: mũi, vai, hông, gối, cổ chân (dùng phía bên TRÁI cơ thể vì quay nghiêng).
  2. Tính offset angle (vai-mũi) -> kiểm tra người có đứng nghiêng đúng không.
  3. Nếu nghiêng đúng: tính 3 góc so với phương đứng:
        - vai-hông   (thân)   -> hip_vertical_angle
        - hông-gối   (đùi)    -> knee_vertical_angle  (điều khiển trạng thái)
        - gối-cổchân (cẳng)   -> ankle_vertical_angle
  4. Xác định trạng thái s1/s2/s3 từ knee_vertical_angle, cập nhật state_sequence.
  5. Khi quay về s1: dựa vào state_sequence + cờ tư thế sai -> đếm CORRECT / INCORRECT.
  6. Sinh feedback (cúi trước / ngả sau / hạ thấp hông / gối vượt mũi chân / squat quá sâu).
  7. Đếm thời gian không hoạt động -> reset bộ đếm.
"""

import time

import cv2
import numpy as np

from utils import find_angle, get_landmark_xy, draw_text, draw_dotted_line


# Chỉ số landmark theo BlazePose (33 điểm). Ta dùng các điểm phía TRÁI.
L_SHLDR, R_SHLDR = 11, 12
L_HIP = 23
L_KNEE = 25
L_ANKLE = 27
NOSE = 0


class ProcessFrame:
    def __init__(self, thresholds, flip_frame=False):
        self.thresholds = thresholds
        self.flip_frame = flip_frame

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # màu
        self.COLOR = {
            "blue":   (255, 127, 0),
            "red":    (50, 50, 255),
            "green":  (0, 200, 0),
            "yellow": (0, 255, 255),
            "white":  (255, 255, 255),
            "gray":   (170, 170, 170),
        }

        # bộ nhớ trạng thái giữa các frame
        self.state_tracker = {
            "state_seq": [],
            "start_inactive_time": time.perf_counter(),
            "INACTIVE_TIME": 0.0,
            "prev_state": None,
            "curr_state": None,
            "SQUAT_COUNT": 0,
            "IMPROPER_SQUAT": 0,
            "INCORRECT_POSTURE": False,
            "DISPLAY_TEXT": np.full(5, False),
            "COUNT_FRAMES": np.zeros(5, dtype=np.int64),
            "LOWER_HIPS": False,
        }

        # 5 thông báo feedback
        self.FEEDBACK = {
            0: ("HONG QUA THAP - DUNG THANG HON",  self.COLOR["yellow"]),
            1: ("CUI NGUOI VE TRUOC",              self.COLOR["yellow"]),
            2: ("NGA NGUOI RA SAU",                self.COLOR["yellow"]),
            3: ("GOI VUOT MUI CHAN",               self.COLOR["red"]),
            4: ("SQUAT QUA SAU",                   self.COLOR["red"]),
        }

    # --------------------------------------------------------------------- #
    def _get_state(self, knee_angle):
        """Trả về 's1'/'s2'/'s3' từ góc đầu gối so với phương đứng."""
        t = self.thresholds["HIP_KNEE_VERT"]
        if t["NORMAL"][0] <= knee_angle <= t["NORMAL"][1]:
            return "s1"
        if t["TRANS"][0] <= knee_angle <= t["TRANS"][1]:
            return "s2"
        if t["PASS"][0] <= knee_angle <= t["PASS"][1]:
            return "s3"
        return None

    def _update_state_sequence(self, state):
        seq = self.state_tracker["state_seq"]
        if state == "s2":
            # chỉ thêm s2 ở chiều đi xuống (chưa có s3) hoặc chiều đi lên (đã qua s3)
            if (("s3" not in seq) and (seq.count("s2") == 0)) or \
               (("s3" in seq) and (seq.count("s2") == 1)):
                seq.append(state)
        elif state == "s3":
            if (state not in seq) and ("s2" in seq):
                seq.append(state)

    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):
        if lower_hips_disp:
            draw_text(frame, self.FEEDBACK[0][0], pos=(30, 80),
                      text_color=(0, 0, 0), box_color=self.FEEDBACK[0][1])
        for idx in np.where(c_frame)[0]:
            draw_text(frame, dict_maps[idx][0], pos=(30, 110 + int(idx) * 30),
                      text_color=(255, 255, 230), box_color=dict_maps[idx][1])
        return frame

    # --------------------------------------------------------------------- #
    def process(self, frame, pose):
        """
        frame: ảnh BGR (numpy).
        pose : đối tượng mp.solutions.pose.Pose đã khởi tạo.
        Trả về (frame_đã_vẽ, count_correct, count_incorrect).
        """
        frame_h, frame_w = frame.shape[:2]
        play_sound = None

        # MediaPipe cần RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        keypoints = pose.process(rgb)

        if not keypoints.pose_landmarks:
            # KHÔNG phát hiện người -> tính thời gian không hoạt động
            self._handle_inactivity(reset_seq=True)
            draw_text(frame, "KHONG PHAT HIEN NGUOI", pos=(30, 40),
                      text_color=(255, 255, 255), box_color=self.COLOR["red"])
            self._draw_counters(frame, frame_w)
            self.state_tracker["prev_state"] = None
            return frame, self.state_tracker["SQUAT_COUNT"], self.state_tracker["IMPROPER_SQUAT"]

        lm = keypoints.pose_landmarks.landmark

        nose   = get_landmark_xy(lm, NOSE,    frame_w, frame_h)
        l_shl  = get_landmark_xy(lm, L_SHLDR, frame_w, frame_h)
        r_shl  = get_landmark_xy(lm, R_SHLDR, frame_w, frame_h)
        hip    = get_landmark_xy(lm, L_HIP,   frame_w, frame_h)
        knee   = get_landmark_xy(lm, L_KNEE,  frame_w, frame_h)
        ankle  = get_landmark_xy(lm, L_ANKLE, frame_w, frame_h)

        # 1) góc lệch vai-mũi: kiểm tra có đứng nghiêng không
        offset_angle = find_angle(l_shl, r_shl, nose)

        if offset_angle > self.thresholds["OFFSET_THRESH"]:
            # người đang quay mặt ra trước -> không phân tích, đếm inactivity
            self._handle_inactivity(reset_seq=True)
            cv2.circle(frame, tuple(nose),  6, self.COLOR["white"], -1)
            cv2.circle(frame, tuple(l_shl), 6, self.COLOR["yellow"], -1)
            cv2.circle(frame, tuple(r_shl), 6, self.COLOR["green"], -1)
            draw_text(frame, "HAY DUNG NGHIENG VOI CAMERA", pos=(30, 40),
                      text_color=(0, 0, 0), box_color=self.COLOR["yellow"])
            draw_text(frame, f"OFFSET: {offset_angle}", pos=(30, 75),
                      text_color=(255, 255, 255), box_color=self.COLOR["gray"])
            self._draw_counters(frame, frame_w)
            self.state_tracker["prev_state"] = None
            return frame, self.state_tracker["SQUAT_COUNT"], self.state_tracker["IMPROPER_SQUAT"]

        # 2) ba góc so với phương đứng
        hip_vertical_angle   = find_angle(l_shl, np.array([hip[0],   0]), hip)
        knee_vertical_angle  = find_angle(hip,   np.array([knee[0],  0]), knee)
        ankle_vertical_angle = find_angle(knee,  np.array([ankle[0], 0]), ankle)

        # vẽ khung xương đơn giản
        for a, b in [(l_shl, hip), (hip, knee), (knee, ankle)]:
            cv2.line(frame, tuple(a), tuple(b), self.COLOR["gray"], 3, cv2.LINE_AA)
        for p, col in [(l_shl, "yellow"), (hip, "blue"), (knee, "green"), (ankle, "blue")]:
            cv2.circle(frame, tuple(p), 6, self.COLOR[col], -1)
        # đường thẳng đứng tham chiếu tại hông và gối
        draw_dotted_line(frame, tuple(hip),  (hip[0],  hip[1] - 80))
        draw_dotted_line(frame, tuple(knee), (knee[0], knee[1] - 60))

        # 3) trạng thái
        curr_state = self._get_state(knee_vertical_angle)
        self.state_tracker["curr_state"] = curr_state
        self._update_state_sequence(curr_state)

        # 4) feedback theo từng trạng thái
        self.state_tracker["LOWER_HIPS"] = False
        kt = self.thresholds["KNEE_THRESH"]
        ht = self.thresholds["HIP_THRESH"]

        if curr_state == "s2":
            # đang đi xuống / đi lên ở pha chuyển tiếp
            if (kt[0] < knee_vertical_angle < kt[1]) and \
               self.state_tracker["state_seq"].count("s2") == 1:
                self.state_tracker["LOWER_HIPS"] = True

        elif curr_state == "s3":
            # ở pha sâu nhất -> kiểm tra tư thế
            if hip_vertical_angle > ht[1]:
                self.state_tracker["DISPLAY_TEXT"][1] = True            # cúi quá nhiều
            elif hip_vertical_angle < ht[0] and \
                    self.state_tracker["state_seq"].count("s2") == 1:
                self.state_tracker["DISPLAY_TEXT"][2] = True            # ngả ra sau

            if ankle_vertical_angle > self.thresholds["ANKLE_THRESH"]:
                self.state_tracker["DISPLAY_TEXT"][3] = True            # gối vượt mũi chân
                self.state_tracker["INCORRECT_POSTURE"] = True

            if knee_vertical_angle > kt[2]:
                self.state_tracker["DISPLAY_TEXT"][4] = True            # squat quá sâu
                self.state_tracker["INCORRECT_POSTURE"] = True

        # 5) đếm khi quay về s1
        if curr_state == "s1":
            seq = self.state_tracker["state_seq"]
            if len(seq) == 3 and not self.state_tracker["INCORRECT_POSTURE"]:
                self.state_tracker["SQUAT_COUNT"] += 1
                play_sound = "correct"
            elif "s2" in seq and len(seq) == 1:
                self.state_tracker["IMPROPER_SQUAT"] += 1   # chưa xuống đủ sâu
                play_sound = "incorrect"
            elif self.state_tracker["INCORRECT_POSTURE"]:
                self.state_tracker["IMPROPER_SQUAT"] += 1
                play_sound = "incorrect"

            self.state_tracker["state_seq"] = []
            self.state_tracker["INCORRECT_POSTURE"] = False

        # 6) thời gian không hoạt động (state không đổi)
        self._update_inactivity(curr_state)

        # hiển thị các góc
        draw_text(frame, f"Hip : {hip_vertical_angle}",   pos=(frame_w - 230, 40),
                  text_color=(255, 255, 255), box_color=self.COLOR["gray"])
        draw_text(frame, f"Knee: {knee_vertical_angle}",  pos=(frame_w - 230, 75),
                  text_color=(255, 255, 255), box_color=self.COLOR["gray"])
        draw_text(frame, f"Ankle:{ankle_vertical_angle}", pos=(frame_w - 230, 110),
                  text_color=(255, 255, 255), box_color=self.COLOR["gray"])

        # đếm frame giữ feedback rồi tự tắt
        self.state_tracker["COUNT_FRAMES"][self.state_tracker["DISPLAY_TEXT"]] += 1
        frame = self._show_feedback(
            frame,
            self.state_tracker["DISPLAY_TEXT"],
            self.FEEDBACK,
            self.state_tracker["LOWER_HIPS"],
        )
        off = self.state_tracker["COUNT_FRAMES"] > self.thresholds["CNT_FRAME_THRESH"]
        self.state_tracker["DISPLAY_TEXT"][off] = False
        self.state_tracker["COUNT_FRAMES"][off] = 0

        self._draw_counters(frame, frame_w)
        self.state_tracker["prev_state"] = curr_state

        return frame, self.state_tracker["SQUAT_COUNT"], self.state_tracker["IMPROPER_SQUAT"]

    # --------------------------------------------------------------------- #
    def _update_inactivity(self, curr_state):
        if self.state_tracker["prev_state"] == curr_state:
            end = time.perf_counter()
            self.state_tracker["INACTIVE_TIME"] += end - self.state_tracker["start_inactive_time"]
            self.state_tracker["start_inactive_time"] = end
            if self.state_tracker["INACTIVE_TIME"] >= self.thresholds["INACTIVE_THRESH"]:
                self.state_tracker["SQUAT_COUNT"] = 0
                self.state_tracker["IMPROPER_SQUAT"] = 0
                self.state_tracker["INACTIVE_TIME"] = 0.0
        else:
            self.state_tracker["start_inactive_time"] = time.perf_counter()
            self.state_tracker["INACTIVE_TIME"] = 0.0

    def _handle_inactivity(self, reset_seq=False):
        end = time.perf_counter()
        self.state_tracker["INACTIVE_TIME"] += end - self.state_tracker["start_inactive_time"]
        self.state_tracker["start_inactive_time"] = end
        if self.state_tracker["INACTIVE_TIME"] >= self.thresholds["INACTIVE_THRESH"]:
            self.state_tracker["SQUAT_COUNT"] = 0
            self.state_tracker["IMPROPER_SQUAT"] = 0
            self.state_tracker["INACTIVE_TIME"] = 0.0
        if reset_seq:
            self.state_tracker["state_seq"] = []
            self.state_tracker["INCORRECT_POSTURE"] = False

    def _draw_counters(self, frame, frame_w):
        draw_text(frame, f"DUNG : {self.state_tracker['SQUAT_COUNT']}",
                  pos=(frame_w - 230, frame.shape[0] - 60),
                  text_color=(255, 255, 255), box_color=self.COLOR["green"])
        draw_text(frame, f"SAI  : {self.state_tracker['IMPROPER_SQUAT']}",
                  pos=(frame_w - 230, frame.shape[0] - 25),
                  text_color=(255, 255, 255), box_color=self.COLOR["red"])

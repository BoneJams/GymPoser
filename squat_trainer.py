import argparse
import sys
import time

import cv2
import numpy as np
import mediapipe as mp

from thresholds import get_thresholds_beginner, get_thresholds_pro
from process_frame import ProcessFrame


# --------------------------------------------------------------------------- #
# NGUỒN HÌNH ẢNH
# --------------------------------------------------------------------------- #
class PiCameraSource:
    """Đọc frame từ Camera Module qua picamera2 (BGR cho OpenCV)."""

    def __init__(self, width, height):
        from picamera2 import Picamera2  # import trễ để máy không có Pi vẫn chạy
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)  # cảm biến ổn định

    def read(self):
        frame = self.picam2.capture_array()       # RGB888
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame

    def release(self):
        self.picam2.stop()


class CV2Source:
    """USB webcam (chỉ số) hoặc file video qua OpenCV."""

    def __init__(self, source, width, height):
        self.cap = cv2.VideoCapture(source)
        # với webcam, cố ép độ phân giải thấp cho nhẹ; với file thì lệnh này bị bỏ qua
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Khong mo duoc nguon: {source}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def build_source(src_arg, width, height):
    if src_arg == "picamera":
        return PiCameraSource(width, height)
    # số -> webcam index; còn lại -> đường dẫn file
    if src_arg.isdigit():
        return CV2Source(int(src_arg), width, height)
    return CV2Source(src_arg, width, height)


# --------------------------------------------------------------------------- #
# PHẢN HỒI GPIO (tuỳ chọn) — còi/đèn báo khi squat sai
# --------------------------------------------------------------------------- #
class GPIOFeedback:
    def __init__(self, buzzer_pin=None, led_pin=None):
        self.buzzer = None
        self.led = None
        if buzzer_pin is None and led_pin is None:
            return
        try:
            from gpiozero import LED, Buzzer
            if buzzer_pin is not None:
                self.buzzer = Buzzer(buzzer_pin)
            if led_pin is not None:
                self.led = LED(led_pin)
        except Exception as e:
            print(f"[GPIO] Bo qua GPIO ({e}).")

    def signal(self, kind):
        if kind == "incorrect" and self.buzzer is not None:
            self.buzzer.beep(on_time=0.15, off_time=0.1, n=2, background=True)
        if kind == "correct" and self.led is not None:
            self.led.blink(on_time=0.2, off_time=0.1, n=1, background=True)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="AI Squat Trainer cho Raspberry Pi 4")
    ap.add_argument("--source", default="picamera",
                    help="'picamera' | chi so webcam (vd 0) | duong dan video")
    ap.add_argument("--mode", default="beginner", choices=["beginner", "pro"])
    ap.add_argument("--width", type=int, default=480, help="be rong frame xu ly")
    ap.add_argument("--height", type=int, default=640, help="chieu cao frame xu ly")
    ap.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2],
                    help="model_complexity cua MediaPipe (0 = nhe nhat, nen dung tren Pi)")
    ap.add_argument("--no-display", action="store_true",
                    help="khong mo cua so (chay headless)")
    ap.add_argument("--flip", action="store_true", help="lat guong frame")
    ap.add_argument("--buzzer-pin", type=int, default=None, help="chan GPIO coi (BCM)")
    ap.add_argument("--led-pin", type=int, default=None, help="chan GPIO den (BCM)")
    args = ap.parse_args()

    thresholds = get_thresholds_pro() if args.mode == "pro" else get_thresholds_beginner()
    analyzer = ProcessFrame(thresholds=thresholds, flip_frame=args.flip)
    gpio = GPIOFeedback(args.buzzer_pin, args.led_pin)

    # MediaPipe Pose — dùng API solutions (khớp bài gốc, chạy tốt trên mediapipe 0.10.x)
    pose = mp.solutions.pose.Pose(
        model_complexity=args.complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        source = build_source(args.source, args.width, args.height)
    except Exception as e:
        print(f"Loi mo nguon hinh: {e}", file=sys.stderr)
        return 1

    prev_correct = 0
    prev_incorrect = 0
    fps_t0 = time.perf_counter()
    fps_n = 0
    fps = 0.0

    print("Bat dau. Nhan Ctrl+C de dung (hoac 'q' tren cua so).")
    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("Het frame / khong doc duoc. Dung.")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            # chuẩn hoá kích thước xử lý cho ổn định FPS
            frame = cv2.resize(frame, (args.width, args.height))

            frame, correct, incorrect = analyzer.process(frame, pose)

            # phát tín hiệu GPIO khi bộ đếm thay đổi
            if correct > prev_correct:
                gpio.signal("correct")
            if incorrect > prev_incorrect:
                gpio.signal("incorrect")
            prev_correct, prev_incorrect = correct, incorrect

            # FPS
            fps_n += 1
            if fps_n >= 10:
                now = time.perf_counter()
                fps = fps_n / (now - fps_t0)
                fps_t0, fps_n = now, 0

            if args.no_display:
                print(f"\rFPS {fps:4.1f} | Dung {correct} | Sai {incorrect}", end="")
            else:
                cv2.putText(frame, f"FPS {fps:4.1f}", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("AI Squat Trainer - Pi 4", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    analyzer.state_tracker["SQUAT_COUNT"] = 0
                    analyzer.state_tracker["IMPROPER_SQUAT"] = 0
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        pose.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("\nDa dung.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

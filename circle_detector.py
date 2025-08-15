import cv2
import numpy as np
import os
from utils.logger_config import LOGGER
from datetime import datetime
import time


class CircleDetector:
    def __init__(self, results_dir="results"):
        """
        :param results_dir: thư mục gốc để lưu kết quả
        """
        self.results_dir = results_dir

    def _get_next_run_dir(self):
        os.makedirs(self.results_dir, exist_ok=True)
        existing = [int(d) for d in os.listdir(self.results_dir) if d.isdigit()]
        next_id = max(existing, default=0) + 1
        run_dir = os.path.join(self.results_dir, str(next_id))
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _detect_hough(self, frame):
        """Phát hiện hình tròn bằng Hough Circle Transform"""
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=30,
            param1=50, 
            param2=30, 
            minRadius=0, 
            maxRadius=30
        )

        count = 0
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(result, (x, y), r, (0, 255, 0), 2)
                cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(result, f"R:{r}", (x-20, y-r-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            count = len(circles)
        return result, count

    def _detect_contour(self, frame):
        """Phát hiện hình tròn bằng Contour Detection"""
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circle_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 500:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.55:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(result, center, radius, (0, 255, 0), 2)
                cv2.circle(result, center, 2, (0, 0, 255), 3)
                cv2.putText(result, f"C:{circularity:.2f}",
                            (center[0]-30, center[1]-radius-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                circle_count += 1
        return result, circle_count

    def predict(self, source=0, mode="hough", show=True, verbose=True, save=False):
        """
        :param source: int (webcam), str (path ảnh/video), hoặc np.ndarray (ảnh)
        :param mode: "hough" hoặc "contour"
        :param show: bool -> hiển thị kết quả
        :param verbose: bool -> log chi tiết
        :param save: bool -> lưu kết quả
        """
        if not verbose:
            LOGGER.setLevel("WARNING")

        mode = mode.lower()
        if mode not in ("hough", "contour"):
            raise ValueError("Mode phải là 'hough' hoặc 'contour'.")

        run_id_dir = self._get_next_run_dir() if save else None
        if save:
            LOGGER.info(f"Lưu kết quả vào: {run_id_dir}")

        # Xác định loại input
        if isinstance(source, np.ndarray):
            frames = [source]
            is_video = False
            cap = None
        elif isinstance(source, str):
            img = cv2.imread(source)
            if img is not None:
                frames = [img]
                is_video = False
                cap = None
            else:
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    LOGGER.error(f"Không mở được file/video: {source}")
                    return
                is_video = True
        elif isinstance(source, int):
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                LOGGER.error(f"Không mở được webcam index={source}")
                return
            is_video = True
        else:
            raise ValueError("source phải là int, str hoặc np.ndarray.")

        # Video writer
        writer = None
        if save and is_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = os.path.join(run_id_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            LOGGER.info(f"Lưu video tại: {video_path}")

        frame_idx = 0
        while True:
            t0 = time.time()

            if is_video:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                frame = frames[0]

            if mode == "hough":
                result, count = self._detect_hough(frame)
            else:
                result, count = self._detect_contour(frame)

            # Vẽ số lượng hình tròn cho ảnh/webcam
            cv2.putText(result, f"Circles: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Vẽ FPS cho webcam
            if is_video:
                fps = 1.0 / max(1e-6, (time.time() - t0))
                cv2.putText(result, f"FPS: {fps:.1f}", (result.shape[1] - 140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if verbose: 
                if is_video:
                    LOGGER.info(f"FPS: {fps:.1f}: {count} hình tròn")
                else:
                    LOGGER.info(f"Frame {frame_idx}: {count} hình tròn")

            if save:
                if is_video:
                    if writer is None:
                        h, w = result.shape[:2]
                        writer = cv2.VideoWriter(video_path, fourcc, 20, (w, h))
                    writer.write(result)
                else:
                    out_path = os.path.join(run_id_dir, f"result_{frame_idx:04d}.jpg")
                    cv2.imwrite(out_path, result)

            if show:
                cv2.imshow("Detected Circles", result)
                if is_video:
                    # video/webcam: non-blocking, nhấn 'q' để thoát
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # ảnh tĩnh: giữ cửa sổ đến khi bấm 'q' (hoặc ESC) mới thoát
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key in (ord('q'), 27):  # 'q' hoặc ESC
                            break

            frame_idx += 1
            if not is_video:
                break

        if writer:
            writer.release()
        if is_video and cap:
            cap.release()
        if show:
            cv2.destroyAllWindows()
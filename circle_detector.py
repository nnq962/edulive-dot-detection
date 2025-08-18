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

    def get_next_run_dir(self):
        os.makedirs(self.results_dir, exist_ok=True)
        existing = [int(d) for d in os.listdir(self.results_dir) if d.isdigit()]
        next_id = max(existing, default=0) + 1
        run_dir = os.path.join(self.results_dir, str(next_id))
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    # ---------- DOT DETECT (Hough / Contour) ----------
    def detect_hough(self, frame):
        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=50,
            param2=30,
            minRadius=0,
            maxRadius=30
        )

        centers = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                centers.append((x, y))
                cv2.circle(result, (x, y), r, (0, 255, 0), 2)
                cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(result, f"R:{r}", (x-20, y-r-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return result, len(centers), np.array(centers, dtype=float)

    def _clahe_L(self, frame, clip=2.0, tile=(8,8)):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        L = clahe.apply(L)
        lab = cv2.merge([L, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _illumination_normalize(self, gray, sigma=25):
        # khử nền sáng không đều: gray_norm ≈ gray / blur
        blur = cv2.GaussianBlur(gray, (0,0), sigmaX=sigma, sigmaY=sigma)
        norm = cv2.divide(gray, blur, scale=128)  # scale cho lại range
        return norm

    def detect_contour(self, frame):
        """
        Phát hiện dot bền với loá sáng / bóng đổ.
        Trả về: result_img, count, centers(np.float32 Nx2)
        """
        result = frame.copy()

        # 1) Tăng tương phản kênh sáng + khử nền
        eq = self._clahe_L(frame, clip=2.0, tile=(8,8))
        gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
        gray = self._illumination_normalize(gray, sigma=25)

        # 2) Adaptive threshold (đảo để chấm đen -> trắng)
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            blockSize=35, C=5
        )

        # 3) Morphology nhẹ để dọn nhiễu và làm tròn dot
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 4) Contour + lọc theo diện tích & độ tròn
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        H, W = gray.shape
        # scale ngưỡng theo ảnh (đỡ phải chỉnh tay)
        scale = max(H, W)
        min_area = max(20, int(0.00002 * scale * scale))   # ~0.002% ảnh
        max_area = int(0.003 * scale * scale)              # ~0.3% ảnh
        min_circ = 0.6

        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            per = cv2.arcLength(cnt, True)
            if per == 0:
                continue
            circ = 4 * np.pi * area / (per * per)
            if circ < min_circ:
                continue

            (x, y), r = cv2.minEnclosingCircle(cnt)
            x, y, r = int(round(x)), int(round(y)), int(round(r))
            centers.append((x, y))

            # vẽ minh hoạ
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
            cv2.putText(result, f"A:{int(area)} C:{circ:.2f}", (x-30, y-r-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        return result, len(centers), np.array(centers, dtype=float)

    # ---------- LINE FIT (TLS/PCA) ----------
    def fit_line_tls(self, points):
        P = np.asarray(points, float)
        mu = P.mean(axis=0)
        X  = P - mu
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        n = Vt[-1]        # pháp tuyến (đơn vị)
        A, B = n
        C = -n @ mu       # line: A x + B y + C = 0
        ratio = (S[-1] / (S[0] + 1e-12)) if S[0] > 0 else 0.0
        return (A, B, C), ratio

    # ---------- LINE UTILS ----------
    def _line_from_two_points(self, p1, p2):
        x1, y1 = p1; x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = -(A*x1 + B*y1)
        norm = (A*A + B*B)**0.5 + 1e-12
        A /= norm; B /= norm; C /= norm
        if C > 0:  # chuẩn hoá dấu
            A, B, C = -A, -B, -C
        return (A, B, C)

    def _angle_rho(self, line_ABC):
        A, B, C = line_ABC
        theta = np.arctan2(B, A)
        if theta < 0: theta += np.pi
        rho = -C
        return theta, rho

    def _angle_diff(self, a, b):
        d = abs(a - b)
        return min(d, np.pi - d)

    def draw_inf_line(self, img, line_ABC, color, thickness=2):
        h, w = img.shape[:2]
        A, B, C = line_ABC
        pts = []
        if abs(B) > 1e-9:
            y0 = (-A*0 - C)/B
            yw = (-A*(w-1) - C)/B
            if 0 <= y0 < h: pts.append((0, int(round(y0))))
            if 0 <= yw < h: pts.append((w-1, int(round(yw))))
        if abs(A) > 1e-9:
            x0 = (-B*0 - C)/A
            xh = (-B*(h-1) - C)/A
            if 0 <= x0 < w: pts.append((int(round(x0)), 0))
            if 0 <= xh < w: pts.append((int(round(xh)), h-1))
        if len(pts) >= 2:
            p0 = pts[0]
            for p1 in pts[1:]:
                if p1 != p0:
                    cv2.line(img, p0, p1, color, thickness, lineType=cv2.LINE_AA)
                    break

    # ---------- FIND 6-DOT LINES ----------
    def find_six_collinear_groups(self, points, eps=3.0, angle_tol_deg=3.0, rho_tol=5.0, max_lines=4):
        """
        Trả về list candidate:
          {"line": (A,B,C), "idx": idx6, "rms": rms, "theta": theta, "rho": rho}
        """
        P = np.asarray(points, float)
        N = len(P)
        if N < 6:
            return []

        angle_tol = np.deg2rad(angle_tol_deg)
        candidates = []

        for i in range(N):
            for j in range(i+1, N):
                line0 = self._line_from_two_points(P[i], P[j])
                A, B, C = line0
                d = np.abs(A*P[:,0] + B*P[:,1] + C)
                inliers = np.where(d <= eps)[0]
                if len(inliers) >= 6:
                    idx6 = inliers[np.argsort(d[inliers])[:6]]
                    pts6 = P[idx6]
                    line_refined, _ = self.fit_line_tls(pts6)
                    A,B,C = line_refined
                    d2 = np.abs(A*P[:,0] + B*P[:,1] + C)
                    idx6 = np.argsort(d2)[:6]
                    pts6 = P[idx6]
                    line_final, _ = self.fit_line_tls(pts6)
                    A,B,C = line_final
                    d6 = np.abs(A*pts6[:,0] + B*pts6[:,1] + C)
                    rms = float(np.sqrt(np.mean(d6**2)))
                    theta, rho = self._angle_rho((A,B,C))

                    # dedup theo (theta, rho)
                    dup = False
                    for cand in candidates:
                        if (self._angle_diff(theta, cand["theta"]) < angle_tol) and (abs(rho - cand["rho"]) < rho_tol):
                            dup = True
                            if rms < cand["rms"]:
                                cand.update({"line": (A,B,C), "idx": idx6, "rms": rms, "theta": theta, "rho": rho})
                            break
                    if not dup:
                        candidates.append({"line": (A,B,C), "idx": idx6, "rms": rms, "theta": theta, "rho": rho})

        candidates.sort(key=lambda c: c["rms"])
        return candidates[:max_lines]

    # ---------- CORNERS BY DOT MEMBERSHIP ----------
    def find_corners_from_lines(self, centers, lines, tol_px):
        """
        Corner = dot thuộc >= 2 line (khoảng cách vuông góc <= tol_px).
        """
        P = np.asarray(centers, float)
        if len(P) == 0 or len(lines) == 0:
            return np.array([], dtype=int), np.empty((0,2), float)

        masks = []
        for (A,B,C) in lines:
            d = np.abs(A*P[:,0] + B*P[:,1] + C)
            masks.append(d <= tol_px)
        M = np.stack(masks, axis=1)  # (N, L)

        counts = M.sum(axis=1)
        idx = np.where(counts >= 2)[0]

        if len(idx) > 4:
            scores = []
            for i in idx:
                d_all = [abs(A*P[i,0] + B*P[i,1] + C) for (A,B,C) in lines]
                d_all.sort()
                scores.append(sum(d_all[:2]))
            keep_idx = np.argsort(scores)[:4]
            idx = idx[keep_idx]

        return idx, P[idx]

    def draw_line_within_corners(self, img, line_ABC, corners_xy, color,
                                 thickness=2, tol_px=3.0, inset=0):
        """
        Vẽ đoạn thẳng chỉ nằm trong phạm vi 2 góc thuộc cùng line.
        - corners_xy: (4,2) toạ độ 4 góc (dot)
        - tol_px: ngưỡng để coi góc “nằm trên” line
        - inset: lùi vào trong mỗi đầu đoạn một chút (px) -> đẹp hơn, tránh đè lên chấm vàng
        """
        if corners_xy is None or len(corners_xy) < 2:
            # fallback: nếu thiếu góc, vẽ full line
            self.draw_inf_line(img, line_ABC, color, thickness)
            return False

        A,B,C = line_ABC
        corners = np.asarray(corners_xy, float)
        # signed distance để chiếu vuông góc
        s = (A*corners[:,0] + B*corners[:,1] + C)
        d = np.abs(s)

        # chọn các góc nằm trên line
        idx = np.where(d <= tol_px)[0]
        if len(idx) < 2:
            # không tìm đủ 2 góc khớp line -> fallback
            self.draw_inf_line(img, line_ABC, color, thickness)
            return False

        # chiếu các góc lên line cho “thẳng hàng” tuyệt đối
        n = np.array([A, B])             # pháp tuyến đơn vị
        proj = corners[idx] - s[idx,None]*n   # điểm chiếu trên line

        # chọn hai đầu mút theo trục tiếp tuyến t = (-B, A)
        t = np.array([-B, A])
        t /= (np.linalg.norm(t) + 1e-12)
        scal = proj @ t
        p1 = proj[np.argmin(scal)]
        p2 = proj[np.argmax(scal)]

        # lùi vào trong cho đẹp (optional)
        if inset > 0:
            p1 = p1 + inset * t
            p2 = p2 - inset * t

        p1i = tuple(np.round(p1).astype(int))
        p2i = tuple(np.round(p2).astype(int))
        cv2.line(img, p1i, p2i, color, thickness, lineType=cv2.LINE_AA)
        return True

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

        run_id_dir = self.get_next_run_dir() if save else None
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
                result, count, centers = self.detect_hough(frame)
            else:
                result, count, centers = self.detect_contour(frame)

            # ---- Find 6-dot lines & draw ----
            if centers is not None and len(centers) >= 6:
                h, w = result.shape[:2]
                eps = max(2.0, 0.002 * max(h, w))  # scale theo ảnh

                # ---- Find 6-dot lines ----
                groups = self.find_six_collinear_groups(centers, eps=eps, angle_tol_deg=3.0, rho_tol=5.0, max_lines=4)

                # ---- Find corners (dot on >= 2 lines) ----
                corner_xy = None
                if len(groups) >= 2:
                    lines = [g["line"] for g in groups]
                    tol_corner = max(eps * 1.2, 2.5)
                    corner_idx, corner_xy = self.find_corners_from_lines(centers, lines, tol_px=tol_corner)

                # ---- Draw lines as segments within corners + draw 6-dot per line ----
                palette = [(0,0,255), (0,255,0), (255,0,0), (255,0,255)]
                for k, g in enumerate(groups):
                    color = palette[k % len(palette)]
                    # vẽ đoạn trong phạm vi 4 góc (nếu corner có), fallback full line nếu thiếu
                    self.draw_line_within_corners(
                        result, g["line"], corner_xy, color,
                        thickness=2, tol_px=tol_corner if corner_xy is not None else eps,
                        inset=8  # chỉnh số này để lùi vào, ví dụ 8px
                    )
                    # vẽ 6 dot của line
                    pts6 = centers[g["idx"]].astype(int)
                    for (x, y) in pts6:
                        cv2.circle(result, (int(x), int(y)), 5, color, -1)
                    cv2.putText(result, f"line{k+1}: rms={g['rms']:.2f}",
                                (10, 60 + 22*k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # ---- Draw corners (màu vàng) ----
                if corner_xy is not None and len(corner_xy) >= 2:
                    corner_color = (0, 255, 255)
                    corner_r = getattr(self, "corner_radius", 12)  # tuỳ chỉnh
                    for (x, y) in corner_xy.astype(int):
                        # viền đen + lõi vàng cho nổi
                        cv2.circle(result, (int(x), int(y)), corner_r+2, (0,0,0), 2)
                        cv2.circle(result, (int(x), int(y)), corner_r,   corner_color, -1)
                    cv2.putText(result, f"corners: {len(corner_xy)}",
                                (10, 60 + 22*len(groups)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, corner_color, 2)

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
import cv2
import numpy as np
import glob
import os
class PlaneTracker:
    def __init__(self, use_sift=True, pyramid_scales=(1.0, 0.75, 0.5, 0.35)):
        if use_sift:
            self.detector = cv2.SIFT_create(nfeatures=3000)
            FLANN_INDEX_KDTREE = 1
            self.matcher = cv2.FlannBasedMatcher(
                dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict()
            )
        else:
            self.detector = cv2.ORB_create(nfeatures=3000)
            self.matcher = cv2.FlannBasedMatcher(
                dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
                dict()
            )

        self.pyramid_scales = pyramid_scales
        self.pyramid = []          # [(scale, kp, desc, (h_scaled, w_scaled))]

        self.target_image = None
        self.prev_gray = None
        self.prev_corners = None
        self.last_confirmed_corners = None
        self.lost_counter = 0

    def set_target(self, image):
        self.target_image = image
        self.pyramid = []
        h0, w0 = image.shape[:2]
        for s in self.pyramid_scales:
            if s == 1.0:
                img = image
            else:
                img = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            kp, desc = self.detector.detectAndCompute(img, None)
            if desc is None or len(desc) < 10:
                continue
            hs, ws = img.shape[:2]
            self.pyramid.append((s, kp, desc, (hs, ws)))

        if not self.pyramid:
            raise ValueError("Не удалось построить пирамиду шаблона — объект слишком однородный.")

    @staticmethod
    def _is_valid_quad(quad, frame_shape, min_frac=0.0015):
        pts = quad.reshape(-1, 2).astype(np.float32)
        if not cv2.isContourConvex(pts.astype(np.int32)):
            return False
        area = cv2.contourArea(pts)
        h, w = frame_shape[:2]
        if area < min_frac * w * h or area > 0.95 * w * h:
            return False
        d = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        if min(d) < 6 or max(d) / max(min(d), 1e-6) > 12:
            return False
        return True

    def _match_scale(self, gray, kp_t, desc_t):
        frame_kp, frame_desc = self.detector.detectAndCompute(gray, None)
        if frame_desc is None or len(frame_desc) < 4:
            return None

        matches = self.matcher.knnMatch(desc_t, frame_desc, k=2)  # ВАЖНО: desc_t первым
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 6:
            return None

        # m.queryIdx -> kp_t (шаблон), m.trainIdx -> frame_kp (кадр)
        src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or mask is None:
            return None

        inliers = int(mask.sum())
        if inliers < 6 or inliers / len(good) < 0.3:
            return None

        return H, inliers

    def _track_optical_flow(self, gray, seed_corners):
        if self.prev_gray is None or seed_corners is None:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            seed_corners.reshape(-1, 1, 2).astype(np.float32),
            None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if p1 is None or st is None or st.sum() < 3:
            return None
        return p1.reshape(-1, 2)

    def track(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best = None  # (inliers, corners, scale)

        for scale, kp_t, desc_t, (hs, ws) in self.pyramid:
            res = self._match_scale(gray, kp_t, desc_t)
            if res is None:
                continue
            H, inliers = res
            # ВАЖНО: углы в координатах ТОГО масштаба, с которым матчились
            obj_pts = np.float32(
                [[0, 0], [0, hs - 1], [ws - 1, hs - 1], [ws - 1, 0]]
            ).reshape(-1, 1, 2)
            cand = cv2.perspectiveTransform(obj_pts, H).reshape(-1, 2)
            if not self._is_valid_quad(cand, frame.shape):
                continue
            # при равенстве inlier'ов предпочитаем более КРУПНЫЙ масштаб (точнее)
            key = (inliers, scale)
            if best is None or key > (best[0], best[2]):
                best = (inliers, cand, scale)

        confirmed = False
        corners = None

        if best is not None:
            _, corners, _ = best
            confirmed = True
            self.lost_counter = 0
            self.last_confirmed_corners = corners.copy()

        if corners is None:
            seed = self.prev_corners if self.prev_corners is not None else self.last_confirmed_corners
            if seed is not None:
                flowed = self._track_optical_flow(gray, seed)
                if flowed is not None and len(flowed) == 4 and self._is_valid_quad(flowed, frame.shape):
                    corners = flowed
            self.lost_counter += 1

        if corners is not None:
            self.prev_corners = corners.astype(np.float32)
        self.prev_gray = gray

        if corners is None:
            return None, False
        return corners.reshape(-1, 1, 2), confirmed


def process_video(video_path, out_dir, show=True):
    print(f"\n{'=' * 60}")
    print(f"Обработка: {os.path.basename(video_path)}")
    print('=' * 60)

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ОШИБКА: не удалось открыть {video_path}")
        return None

    ret, first_frame = cap.read()
    if not ret:
        print("  ОШИБКА: не удалось прочитать первый кадр")
        cap.release()
        return None

    tracker = PlaneTracker(use_sift=True)
    try:
        tracker.set_target(first_frame)
    except ValueError as e:
        print(f"  {e}")
        cap.release()
        return None

    if show:
        cv2.imshow(f"Target - {os.path.basename(video_path)}", first_frame)
        cv2.waitKey(500)

    # сохраняем в ОТДЕЛЬНУЮ папку
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, base + "_tracked.avi")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    total = tracked = 0
    tracked_confirmed = 0
    user_quit = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1

        corners, confirmed = tracker.track(frame)
        if corners is not None:
            tracked += 1
            if confirmed:
                tracked_confirmed += 1
                color = (0, 255, 0)
                label = "Tracked Object"
            else:
                color = (0, 200, 255)  # жёлтый — «предположительно» (только flow)
                label = "Tracking (flow)"

            ci = np.int32(corners)
            cv2.polylines(frame, [ci], True, color, 3)
            tl = ci[0][0]
            cv2.putText(frame, label,
                        (tl[0], max(tl[1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        writer.write(frame)

        if show:
            cv2.imshow("Tracking", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                user_quit = True
                break
            elif key == ord(' '):
                cv2.waitKey(0)

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    percent = (tracked / total * 100) if total else 0
    percent_conf = (tracked_confirmed / total * 100) if total else 0
    note = " (прервано)" if user_quit else ""
    print(f"  Кадров: {total}, рамка нарисована: {tracked} ({percent:.1f}%), "
          f"из них подтверждено матчингом: {tracked_confirmed} ({percent_conf:.1f}%){note}")
    print(f"  Сохранено: {out_path}")

    return {"file": os.path.basename(video_path),
            "total": total,
            "tracked": tracked,
            "confirmed": tracked_confirmed,
            "percent": percent,
            "percent_conf": percent_conf,
            "out": out_path}


def main():
    video_dir = r"D:\prog\Java\pythonProject\skoe\lab_2\test-videos"
    # новая папка для результатов — рядом с test-videos
    out_dir = r"D:\prog\Java\pythonProject\skoe\lab_2\tracked-output"

    pattern = os.path.join(video_dir, "mona-lis*.avi")
    video_files = sorted(glob.glob(pattern))

    if not video_files:
        print(f"Не найдено видео по маске: {pattern}")
        return

    print(f"Найдено видео: {len(video_files)}")
    for f in video_files:
        print(f"  - {f}")
    print(f"Результаты будут сохранены в: {out_dir}")

    results = []
    for vp in video_files:
        r = process_video(vp, out_dir, show=True)
        if r:
            results.append(r)

    print(f"\n{'=' * 60}")
    print("ИТОГОВАЯ СВОДКА")
    print('=' * 60)
    print(f"{'Файл':<40} {'Кадров':>8} {'Рамка':>8} {'Подтв.':>8} {'%':>6} {'%подтв':>7}")
    print('-' * 80)
    for r in results:
        print(f"{r['file']:<40} {r['total']:>8} {r['tracked']:>8} "
              f"{r['confirmed']:>8} {r['percent']:>5.1f}% {r['percent_conf']:>6.1f}%")


if __name__ == "__main__":
    main()#отлично сделана рамка
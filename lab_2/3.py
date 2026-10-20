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
        self.pyramid = []

        self.target_image = None
        self.prev_gray = None
        self.prev_corners = None
        self.last_confirmed_corners = None
        self.lost_counter = 0

        # ----- ФИКСИРОВАННЫЙ набор точек объекта -----
        self.obj_pts_t = None       # (N,2) точки на ШАБЛОНЕ — задаются один раз
        self.obj_pts_f = None       # (N,2) точки на кадре — трекаются flow'ом
        self.prev_obj_pts_f = None
        self.MAX_POINTS = 200
        self.lost_pts_counter = 0

        # сглаживание рамки
        self.smoothed_corners = None

    def set_target(self, image):
        self.target_image = image
        self.pyramid = []
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

        kp_full = self.pyramid[0][1]
        if len(kp_full) > self.MAX_POINTS:
            order = np.argsort([-k.response for k in kp_full])[:self.MAX_POINTS]
            kp_sel = [kp_full[i] for i in order]
        else:
            kp_sel = kp_full
        self.obj_pts_t = np.array([kp.pt for kp in kp_sel], dtype=np.float32)
        self.obj_pts_f = self.obj_pts_t.copy()

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

        matches = self.matcher.knnMatch(desc_t, frame_desc, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 6:
            return None

        src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or mask is None:
            return None

        inliers = int(mask.sum())
        if inliers < 6 or inliers / len(good) < 0.3:
            return None

        inlier_flags = mask.ravel().astype(bool)
        return {
            "H": H, "inliers": inliers,
            "kp_t": kp_t, "kp_f": frame_kp,
            "good": good, "inlier_flags": inlier_flags,
        }

    def _match_all_scales(self, gray, frame_shape):
        best = None
        for scale, kp_t, desc_t, (hs, ws) in self.pyramid:
            res = self._match_scale(gray, kp_t, desc_t)
            if res is None:
                continue
            H = res["H"]
            obj_pts = np.float32(
                [[0, 0], [0, hs - 1], [ws - 1, hs - 1], [ws - 1, 0]]
            ).reshape(-1, 1, 2)
            cand = cv2.perspectiveTransform(obj_pts, H).reshape(-1, 2)
            if not self._is_valid_quad(cand, frame_shape):
                continue
            key = (res["inliers"], scale)
            if best is None or key > (best[0], best[2]):
                best = (res["inliers"], cand, scale, res, (hs, ws))
        return best

    def _track_optical_flow(self, gray, prev_pts):
        if self.prev_gray is None or prev_pts is None or len(prev_pts) < 4:
            return None, None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            prev_pts.reshape(-1, 1, 2).astype(np.float32),
            None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if p1 is None or st is None:
            return None, None
        mask = st.ravel().astype(bool)
        return p1.reshape(-1, 2), mask

    # ========== восстановление точек по рамке ==========
    def _restore_points_from_corners(self, corners):
        """Восстанавливает obj_pts_f из obj_pts_t через гомографию рамки."""
        if self.obj_pts_t is None or corners is None or len(self.obj_pts_t) < 4:
            return False
        h0, w0 = self.target_image.shape[:2]
        obj_quad = np.float32(
            [[0, 0], [0, h0 - 1], [w0 - 1, h0 - 1], [w0 - 1, 0]]
        ).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(obj_quad, corners.reshape(-1, 1, 2), 0)
        if H is None:
            return False
        new_pts = cv2.perspectiveTransform(
            self.obj_pts_t.reshape(-1, 1, 2), H
        ).reshape(-1, 2).astype(np.float32)
        self.obj_pts_f = new_pts
        return True

    # ========== переинициализация с ПЕРЕИСПОЛЬЗОВАНИЕМ pt_t ==========
    def _reinit_points_from_match(self, res, hs, ws, reuse_dist=8.0):
        """
        Берёт inlier'ы матчинга как новые точки, НО если новая pt_t близка
        к существующей — переиспользует старую (точка остаётся "той же").
        """
        kp_t = res["kp_t"]
        kp_f = res["kp_f"]
        good = res["good"]
        inlier_flags = res["inlier_flags"]

        new_pt_t = np.array([kp_t[m.queryIdx].pt for m in good], dtype=np.float32)
        new_pt_f = np.array([kp_f[m.trainIdx].pt for m in good], dtype=np.float32)
        in_mask = inlier_flags.astype(bool)
        new_pt_t = new_pt_t[in_mask]
        new_pt_f = new_pt_f[in_mask]

        if len(new_pt_t) < 6:
            return False

        if self.obj_pts_t is None or len(self.obj_pts_t) == 0:
            self.obj_pts_t = new_pt_t
            self.obj_pts_f = new_pt_f
            return True

        # стараемся сохранить существующие точки, к которым новые близки
        old_t = self.obj_pts_t
        old_f = self.obj_pts_f
        n_old = len(old_t)

        # для каждой новой pt_t ищем ближайшую старую; если <reuse_dist — переиспользуем
        used_old = np.zeros(n_old, dtype=bool)
        final_t = []
        final_f = []
        for i in range(len(new_pt_t)):
            p = new_pt_t[i]
            if n_old > 0:
                d = np.linalg.norm(old_t - p, axis=1)
                d[used_old] = 1e9
                j = int(np.argmin(d))
                if d[j] < reuse_dist:
                    final_t.append(old_t[j])
                    final_f.append(new_pt_f[i])  # свежее положение
                    used_old[j] = True
                    continue
            final_t.append(new_pt_t[i])
            final_f.append(new_pt_f[i])

        # оставшиеся старые точки (к которым не нашлось новых) — тоже сохраняем
        # с их последним положением, если оно ещё в кадре
        for j in range(n_old):
            if not used_old[j]:
                final_t.append(old_t[j])
                final_f.append(old_f[j])

        final_t = np.array(final_t, dtype=np.float32)
        final_f = np.array(final_f, dtype=np.float32)

        # прореживаем до MAX_POINTS
        if len(final_t) > self.MAX_POINTS:
            idx = np.linspace(0, len(final_t) - 1, self.MAX_POINTS).astype(int)
            final_t = final_t[idx]
            final_f = final_f[idx]

        self.obj_pts_t = final_t
        self.obj_pts_f = final_f
        return True

    # ---------- сглаживание ----------
    def _smooth(self, corners, alpha=0.6):
        if self.smoothed_corners is None:
            self.smoothed_corners = corners.copy()
            return corners
        self.smoothed_corners = (alpha * corners +
                                 (1 - alpha) * self.smoothed_corners).astype(np.float32)
        return self.smoothed_corners

    # ========== публичный метод ==========
    def track(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h0, w0 = self.target_image.shape[:2]
        obj_quad = np.float32(
            [[0, 0], [0, h0 - 1], [w0 - 1, h0 - 1], [w0 - 1, 0]]
        ).reshape(-1, 1, 2)

        confirmed = False
        corners = None

        self.prev_obj_pts_f = (self.obj_pts_f.copy()
                               if self.obj_pts_f is not None else None)

        n_pts_total = len(self.obj_pts_t) if self.obj_pts_t is not None else 0

        # ========== 1. flow по фиксированным точкам ==========
        flow_ok = False
        n_after_flow = 0
        if self.obj_pts_f is not None and len(self.obj_pts_f) >= 6:
            new_pts, mask = self._track_optical_flow(gray, self.obj_pts_f)
            if new_pts is not None and mask.sum() >= 6:
                self.obj_pts_f = new_pts[mask]
                self.obj_pts_t = self.obj_pts_t[mask]
                flow_ok = True
                n_after_flow = len(self.obj_pts_f)

                if len(self.obj_pts_t) >= 6:
                    Hf, _ = cv2.findHomography(
                        self.obj_pts_t.reshape(-1, 1, 2),
                        self.obj_pts_f.reshape(-1, 1, 2),
                        cv2.RANSAC, 4.0,
                    )
                    if Hf is not None:
                        cand = cv2.perspectiveTransform(obj_quad, Hf).reshape(-1, 2)
                        if self._is_valid_quad(cand, frame.shape):
                            corners = cand
                            confirmed = True

        # ЖЁСТКАЯ ПОТЕРЯ ТОЧЕК: <30% от исходного набора, но рамка ещё есть
        # → восстанавливаем точки по рамке
        if flow_ok and n_pts_total > 0 and n_after_flow < 0.3 * n_pts_total:
            if corners is not None:
                self._restore_points_from_corners(corners)
                self.lost_pts_counter = 0

        # ========== 2. матчинг — для коррекции/переинициализации ==========
        best = self._match_all_scales(gray, frame.shape)

        if best is not None:
            _, m_corners, scale, res, (hs, ws) = best

            # 2a. если flow совсем мёртв — переинициализация из матчинга
            if not flow_ok or self.obj_pts_f is None or len(self.obj_pts_f) < 15:
                if self._reinit_points_from_match(res, hs, ws):
                    corners = m_corners
                    confirmed = True
                    self.smoothed_corners = None  # сброс сглаживания при реинициализации
            else:
                # 2b. flow жив — проверяем согласованность рамок
                if corners is None:
                    # рамки от flow нет (не собралась Hf), берём матчинг
                    # но точки оставляем — восстановим по матчинг-рамке
                    corners = m_corners
                    confirmed = True
                    self._restore_points_from_corners(corners)
                else:
                    a_flow = cv2.contourArea(corners.reshape(-1, 2).astype(np.float32))
                    a_match = cv2.contourArea(m_corners.reshape(-1, 2).astype(np.float32))
                    if a_flow > 0 and abs(a_match / a_flow - 1.0) > 0.25:
                        # сильное расхождение — доверяем матчингу, переинициализируем точки
                        if self._reinit_points_from_match(res, hs, ws):
                            corners = m_corners
                            confirmed = True
                            self.smoothed_corners = None

        # ========== 3. восстановление рамки по точкам, если её нет ==========
        if corners is None and self.obj_pts_f is not None and len(self.obj_pts_f) >= 6:
            Hf, _ = cv2.findHomography(
                self.obj_pts_t.reshape(-1, 1, 2),
                self.obj_pts_f.reshape(-1, 1, 2),
                cv2.RANSAC, 4.0,
            )
            if Hf is not None:
                cand = cv2.perspectiveTransform(obj_quad, Hf).reshape(-1, 2)
                if self._is_valid_quad(cand, frame.shape):
                    corners = cand
                    confirmed = False

        # ========== 4. fallback: 4 угла с прошлого кадра ==========
        if corners is None:
            seed = self.prev_corners if self.prev_corners is not None else self.last_confirmed_corners
            if seed is not None:
                flowed, _ = self._track_optical_flow(gray, seed.reshape(-1, 2))
                if flowed is not None and len(flowed) == 4 and self._is_valid_quad(flowed, frame.shape):
                    corners = flowed
                    confirmed = False
            self.lost_counter += 1

        # ========== 5. сглаживание рамки ==========
        if corners is not None:
            if confirmed:
                corners = self._smooth(corners, alpha=0.7)
            self.prev_corners = corners.astype(np.float32)
            self.last_confirmed_corners = corners.copy()
        self.prev_gray = gray

        if corners is None:
            return None, False
        return corners.reshape(-1, 1, 2), confirmed


# ---------- визуализация ----------
def draw_debug(frame, tracker, confirmed, corners):
    vis = frame.copy()
    H_f, W_f = vis.shape[:2]

    if tracker.prev_obj_pts_f is not None and tracker.obj_pts_f is not None:
        n = min(len(tracker.prev_obj_pts_f), len(tracker.obj_pts_f))
        for i in range(n):
            x1, y1 = tracker.prev_obj_pts_f[i]
            x2, y2 = tracker.obj_pts_f[i]
            if (0 <= x1 < W_f and 0 <= y1 < H_f and
                    0 <= x2 < W_f and 0 <= y2 < H_f):
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 200, 255), 1, cv2.LINE_AA)

    if tracker.obj_pts_f is not None:
        for (x, y) in tracker.obj_pts_f:
            xi, yi = int(x), int(y)
            if 0 <= xi < W_f and 0 <= yi < H_f:
                color = (0, 255, 0) if confirmed else (0, 200, 255)
                cv2.circle(vis, (xi, yi), 2, color, -1)

    if corners is not None:
        ci = np.int32(corners)
        color = (0, 255, 0) if confirmed else (0, 200, 255)
        label = "Tracked Object" if confirmed else "Tracking (flow)"
        cv2.polylines(vis, [ci], True, color, 3)
        for p in ci.reshape(-1, 2):
            cv2.circle(vis, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
            cv2.circle(vis, (int(p[0]), int(p[1])), 5, (255, 255, 255), 1)
        tl = ci[0][0]
        cv2.putText(vis, label, (tl[0], max(tl[1] - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if tracker.target_image is not None:
        THUMB_W, THUMB_H, M = 160, 120, 8
        thumb = cv2.resize(tracker.target_image, (THUMB_W, THUMB_H),
                           interpolation=cv2.INTER_AREA)
        if tracker.obj_pts_t is not None:
            h0, w0 = tracker.target_image.shape[:2]
            sx, sy = THUMB_W / w0, THUMB_H / h0
            for (x, y) in tracker.obj_pts_t:
                xi, yi = int(x * sx), int(y * sy)
                if 0 <= xi < THUMB_W and 0 <= yi < THUMB_H:
                    cv2.circle(thumb, (xi, yi), 1, (0, 255, 0), -1)
        x0, y0 = W_f - THUMB_W - M, M
        vis[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = thumb
        cv2.rectangle(vis, (x0 - 1, y0 - 1),
                      (x0 + THUMB_W, y0 + THUMB_H), (255, 255, 255), 1)

    n_pts = len(tracker.obj_pts_f) if tracker.obj_pts_f is not None else 0
    if confirmed:
        text = f"match/flow  points={n_pts}"
    elif corners is not None:
        text = f"flow only  points={n_pts}"
    else:
        text = "no tracking"
    cv2.rectangle(vis, (0, 0), (W_f, 28), (0, 0, 0), -1)
    cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return vis


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

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, base + "_tracked.avi")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    total = tracked = 0
    tracked_confirmed = 0
    user_quit = False
    show_debug = True

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

        vis = draw_debug(frame, tracker, confirmed, corners) if show_debug else frame.copy()
        if not show_debug and corners is not None:
            ci = np.int32(corners)
            color = (0, 255, 0) if confirmed else (0, 200, 255)
            cv2.polylines(vis, [ci], True, color, 3)

        writer.write(vis)

        if show:
            cv2.imshow("Tracking", vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                user_quit = True
                break
            elif key == ord(' '):
                cv2.waitKey(0)
            elif key == ord('d'):
                show_debug = not show_debug

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    percent = (tracked / total * 100) if total else 0
    percent_conf = (tracked_confirmed / total * 100) if total else 0
    note = " (прервано)" if user_quit else ""
    print(f"  Кадров: {total}, рамка: {tracked} ({percent:.1f}%), "
          f"подтверждено: {tracked_confirmed} ({percent_conf:.1f}%){note}")
    print(f"  Сохранено: {out_path}")

    return {"file": os.path.basename(video_path),
            "total": total, "tracked": tracked,
            "confirmed": tracked_confirmed,
            "percent": percent, "percent_conf": percent_conf,
            "out": out_path}


def main():
    video_dir = r"D:\prog\Java\pythonProject\skoe\lab_2\test-videos"
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
    main()#отлично сделаны ключевые точки
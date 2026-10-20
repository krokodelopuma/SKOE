import cv2
import numpy as np
import glob
import os
import csv


class PlaneTracker:
    def __init__(self, use_sift=True,
                 pyramid_scales=(1.0, 0.75, 0.5, 0.4, 0.3, 0.25),
                 log_path=None):
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
        self.obj_pts_t = None
        self.obj_pts_f = None
        self.alive_mask = None
        self.prev_obj_pts_f = None
        self.MAX_POINTS = 300

        # эталон для аварийного сброса
        self.saved_obj_pts_t = None
        self.saved_obj_pts_f = None

        self.scale_obj = 1.0
        self.quality = 1.0

        self.smoothed_corners = None

        self.fallback_streak = 0
        self.FALLBACK_MAX_STREAK = 5

        self.restore_cooldown = 0

        # ----- контроль точек вне рамки -----
        self.OUTSIDE_LIMIT = 100          # порог "плохих" точек
        self.OUTSIDE_STREAK_MAX = 2       # сколько кадров подряд терпим
        self.outside_streak = 0           # текущий счётчик
        self.rebuild_cooldown = 0         # кулдаун пересоздания
        self.REBUILD_COOLDOWN = 3         # не чаще, чем раз в 3 кадра

        # ----- ЛОГИРОВАНИЕ -----
        self.frame_idx = 0
        self.log_path = log_path
        self.log_fh = None
        self.log_writer = None
        if log_path is not None:
            self._open_log()

    # ============ ЛОГИРОВАНИЕ ============
    def _open_log(self):
        self.log_fh = open(self.log_path, "w", newline="", encoding="utf-8-sig")
        self.log_writer = csv.writer(self.log_fh, delimiter=";")
        self.log_writer.writerow([
            "frame", "event", "n_obj_pts", "n_alive_pts", "n_frame_pts",
            "scale_obj", "quality",
            "frame_corners", "target_corners",
            "obj_pts_t", "obj_pts_f",
            "note",
        ])
        self.log_fh.flush()

    def close_log(self):
        if self.log_fh is not None:
            try:
                self.log_fh.flush()
                self.log_fh.close()
            finally:
                self.log_fh = None
                self.log_writer = None

    @staticmethod
    def _corners_to_str(corners):
        if corners is None:
            return ""
        c = np.asarray(corners).reshape(-1, 2)
        return ";".join(f"{x:.2f},{y:.2f}" for x, y in c)

    @staticmethod
    def _pts_to_str(pts):
        if pts is None or len(pts) == 0:
            return ""
        return ";".join(f"{x:.2f},{y:.2f}" for x, y in np.asarray(pts).reshape(-1, 2))

    def _log_event(self, event, corners=None, note="", n_frame_pts=0):
        if self.log_writer is None:
            return
        target_corners = ""
        if self.target_image is not None:
            h0, w0 = self.target_image.shape[:2]
            target_corners = self._corners_to_str(
                np.float32([[0, 0], [0, h0 - 1], [w0 - 1, h0 - 1], [w0 - 1, 0]])
            )
        n_obj = len(self.obj_pts_t) if self.obj_pts_t is not None else 0
        n_alive = int(self.alive_mask.sum()) if self.alive_mask is not None else 0
        self.log_writer.writerow([
            self.frame_idx,
            event,
            n_obj,
            n_alive,
            n_frame_pts,
            f"{self.scale_obj:.3f}",
            f"{self.quality:.3f}",
            self._corners_to_str(corners),
            target_corners,
            self._pts_to_str(self.obj_pts_t),
            self._pts_to_str(self.obj_pts_f),
            note,
        ])
        self.log_fh.flush()

    # ============ ОСНОВНОЕ ============
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

        kp_full = None
        for s, kp, desc, (hs, ws) in self.pyramid:
            if s == 1.0:
                kp_full = kp
                break
        if kp_full is None:
            s0, kp_full, _, _ = self.pyramid[0]
            if s0 != 1.0:
                kp_full = [
                    cv2.KeyPoint(k.pt[0] / s0, k.pt[1] / s0, k.size / s0,
                                 k.angle, k.response, k.octave, k.class_id)
                    for k in kp_full
                ]

        if len(kp_full) > self.MAX_POINTS:
            order = np.argsort([-k.response for k in kp_full])[:self.MAX_POINTS]
            kp_sel = [kp_full[i] for i in order]
        else:
            kp_sel = kp_full

        self.obj_pts_t = np.array([kp.pt for kp in kp_sel], dtype=np.float32)
        self.obj_pts_f = self.obj_pts_t.copy()
        self.alive_mask = np.ones(len(self.obj_pts_t), dtype=bool)

        self.saved_obj_pts_t = self.obj_pts_t.copy()
        self.saved_obj_pts_f = self.obj_pts_t.copy()

    @staticmethod
    def _is_valid_quad(quad, frame_shape, min_frac=0.0015, margin=1.5, min_side=8):
        pts = quad.reshape(-1, 2).astype(np.float32)
        h, w = frame_shape[:2]

        if margin is None:
            m = 0.5 * max(h, w)
        else:
            m = margin

        if np.any(pts[:, 0] < -m) or np.any(pts[:, 0] > w + m):
            return False
        if np.any(pts[:, 1] < -m) or np.any(pts[:, 1] > h + m):
            return False

        if not cv2.isContourConvex(pts.astype(np.int32)):
            hull = cv2.convexHull(pts)
            if len(hull) < 4:
                return False
            a0 = cv2.contourArea(pts)
            a1 = cv2.contourArea(hull)
            if a0 > 0 and a1 / a0 > 1.6:
                return False

        area = cv2.contourArea(pts)
        if area < min_frac * w * h or area > 0.98 * w * h:
            return False

        d = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        if min(d) < min_side or max(d) / max(min(d), 1e-6) > 20:
            return False
        return True

    # ============ ОЦЕНКА КАЧЕСТВА И ПАРАМЕТРЫ ============
    def _assess_quality(self, gray, corners):
        """0..1: чем хуже наклон/размытость — тем ниже."""
        blur_score = 1.0
        if corners is not None and gray is not None:
            x, y, w, h = cv2.boundingRect(np.int32(corners).reshape(-1, 2))
            x0, y0 = max(x, 0), max(y, 0)
            x1 = min(x + w, gray.shape[1])
            y1 = min(y + h, gray.shape[0])
            if x1 > x0 + 8 and y1 > y0 + 8:
                roi = gray[y0:y1, x0:x1]
                lap_var = cv2.Laplacian(roi, cv2.CV_64F).var()
                blur_score = float(np.clip(lap_var / 100.0, 0.2, 1.0))

        tilt_score = 1.0
        if corners is not None and self.target_image is not None:
            pts = np.float32(corners).reshape(-1, 2)
            rect = cv2.minAreaRect(pts)
            rw, rh = rect[1]
            if min(rw, rh) > 1e-3:
                aspect_now = max(rw, rh) / min(rw, rh)
                h0, w0 = self.target_image.shape[:2]
                aspect_ref = max(h0, w0) / max(min(h0, w0), 1)
                tilt_score = float(np.clip(aspect_ref / aspect_now, 0.3, 1.0))

        return float(np.clip(min(blur_score, tilt_score), 0.2, 1.0))

    def _params(self, corners):
        """win, max_level, reproj_t, fb_t, min_inliers, ransac_t, max_area, max_lost.
        Чем меньше scale_obj и чем хуже quality — тем мягче."""
        self.quality = self._assess_quality(self.prev_gray, corners)

        s = self.scale_obj
        q = self.quality

        if s >= 0.75:
            win_base, max_lvl = 21, 3
        elif s >= 0.5:
            win_base, max_lvl = 17, 4
        elif s >= 0.4:
            win_base, max_lvl = 15, 4
        elif s >= 0.3:
            win_base, max_lvl = 13, 4
        else:
            win_base, max_lvl = 11, 4

        if q < 0.6 and s >= 0.4:
            win_base += 4
        win = (win_base, win_base)

        fb_t = 1.5 + (1.0 - q) * 2.5           # 1.5..4.0
        reproj_t = 4.0 + (1.0 - q) * 3.0       # 4.0..7.0

        if s >= 0.75:
            min_inl = 15
        elif s >= 0.5:
            min_inl = 12
        elif s >= 0.4:
            min_inl = 10
        elif s >= 0.3:
            min_inl = 8
        else:
            min_inl = 6
        min_inl = max(6, int(round(min_inl * (0.6 + 0.4 * q))))

        ransac_t = 3.0 + (1.0 - q) * 4.0       # 3.0..7.0
        max_area = 1.35 + (1.0 - q) * 0.9      # 1.35..2.25
        max_lost = 6 if q > 0.6 else 8

        return win, max_lvl, reproj_t, fb_t, min_inl, ransac_t, max_area, max_lost

    # ============ МАТЧИНГ ============
    def _match_scale(self, gray, kp_t, desc_t, scale=1.0, relaxed=False):
        frame_kp, frame_desc = self.detector.detectAndCompute(gray, None)
        if frame_desc is None or len(frame_desc) < 4:
            return None

        matches = self.matcher.knnMatch(desc_t, frame_desc, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            ratio = 0.80 if relaxed else 0.75
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < 8:
            return None

        src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        ransac_thresh = 8.0 if self.scale_obj < 0.4 else 3.0

        try:
            H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_thresh)
        except cv2.error:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None:
            return None

        inliers = int(mask.sum())

        if scale >= 0.75:
            min_inliers = 8
        elif scale >= 0.5:
            min_inliers = 12
        elif scale >= 0.4:
            min_inliers = 15
        elif scale >= 0.3:
            min_inliers = 18
        else:
            min_inliers = 22

        if relaxed:
            base_min = min_inliers
            if self.scale_obj < 0.4:
                base_min = min(base_min, 8)
            min_inliers = max(6, int(base_min * 0.6))
            min_ratio_inliers = 0.15
        else:
            if self.scale_obj < 0.4:
                min_inliers = min(min_inliers, 8)
            min_ratio_inliers = 0.20

        if inliers < min_inliers or inliers / len(good) < min_ratio_inliers:
            return None

        inlier_flags = mask.ravel().astype(bool)
        return {
            "H": H, "inliers": inliers,
            "kp_t": kp_t, "kp_f": frame_kp,
            "good": good, "inlier_flags": inlier_flags,
        }

    def _match_all_scales(self, gray, frame_shape, relaxed=False):
        best = None
        for scale, kp_t, desc_t, (hs, ws) in self.pyramid:
            res = self._match_scale(gray, kp_t, desc_t, scale, relaxed=relaxed)
            if res is None:
                continue
            H = res["H"]
            obj_pts = np.float32(
                [[0, 0], [0, hs - 1], [ws - 1, hs - 1], [ws - 1, 0]]
            ).reshape(-1, 1, 2)
            cand = cv2.perspectiveTransform(obj_pts, H).reshape(-1, 2)
            margin = None if relaxed else 2.0
            if not self._is_valid_quad(cand, frame_shape, margin=margin):
                continue
            key = (res["inliers"], scale)
            if best is None or key > (best[0], best[2]):
                best = (res["inliers"], cand, scale, res, (hs, ws))
        return best

    # ============ ОПТИЧЕСКИЙ ПОТОК ============
    def _track_optical_flow(self, gray, prev_pts, win=(21, 21), max_level=3,
                            fb_thresh=1.5):
        if self.prev_gray is None or prev_pts is None or len(prev_pts) < 4:
            return None, None

        p0 = prev_pts.reshape(-1, 1, 2).astype(np.float32)

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, p0, None,
            winSize=win, maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if p1 is None or st1 is None:
            return None, None

        p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, p1, None,
            winSize=win, maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if p0_back is None or st2 is None:
            return p1.reshape(-1, 2), st1.ravel().astype(bool)

        fb_err = np.linalg.norm(p0.reshape(-1, 2) - p0_back.reshape(-1, 2), axis=1)
        mask = (st1.ravel() > 0) & (st2.ravel() > 0) & (fb_err < fb_thresh)
        return p1.reshape(-1, 2), mask

    def _smooth(self, corners, alpha=0.6):
        if self.smoothed_corners is None:
            self.smoothed_corners = corners.copy()
            return corners
        self.smoothed_corners = (alpha * corners +
                                 (1 - alpha) * self.smoothed_corners).astype(np.float32)
        return self.smoothed_corners

    # ============ ЕДИНСТВЕННЫЙ «РЕМОНТ» ТОЧЕК ============
    def _reset_to_original_points(self):
        if self.saved_obj_pts_t is None:
            return
        self.obj_pts_t = self.saved_obj_pts_t.copy()
        self.obj_pts_f = self.saved_obj_pts_f.copy()
        self.alive_mask = np.ones(len(self.obj_pts_t), dtype=bool)
        self.lost_counter = 0
        self.fallback_streak = 0

    # ============ ПЕРЕСОЗДАНИЕ ТОЧЕК ПО ШАБЛОНУ ВНУТРИ РАМКИ ============
    @staticmethod
    def _count_outside(corners, pts):
        """Сколько точек pts лежит вне четырёхугольника corners."""
        if corners is None or pts is None or len(pts) == 0:
            return 0
        quad = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        inside = np.array([
            cv2.pointPolygonTest(quad, (float(x), float(y)), False) >= 0
            for x, y in p
        ])
        return int((~inside).sum())

    def _rebuild_points_inside_corners(self, corners):
        """Пересоздаёт obj_pts_t/f из шаблона так, чтобы они попали внутрь corners."""
        if self.target_image is None or corners is None:
            return False

        h0, w0 = self.target_image.shape[:2]
        obj_quad = np.float32(
            [[0, 0], [0, h0 - 1], [w0 - 1, h0 - 1], [w0 - 1, 0]]
        ).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(
            obj_quad, np.asarray(corners, np.float32).reshape(-1, 1, 2),
            cv2.RANSAC, 3.0
        )
        if H is None:
            return False

        base_t = self.saved_obj_pts_t if self.saved_obj_pts_t is not None \
            else self.obj_pts_t
        if base_t is None or len(base_t) == 0:
            return False

        proj_f = cv2.perspectiveTransform(
            base_t.reshape(-1, 1, 2).astype(np.float32), H
        ).reshape(-1, 2).astype(np.float32)

        quad = np.asarray(corners, np.float32).reshape(-1, 2)
        inside = np.array([
            cv2.pointPolygonTest(quad, (float(x), float(y)), False) >= 0
            for x, y in proj_f
        ])
        new_t = base_t[inside].copy()
        new_f = proj_f[inside].copy()

        if len(new_t) < 8:
            return False

        if len(new_t) > self.MAX_POINTS:
            idx = np.linspace(0, len(new_t) - 1, self.MAX_POINTS).astype(int)
            new_t = new_t[idx]
            new_f = new_f[idx]

        self.obj_pts_t = new_t
        self.obj_pts_f = new_f
        self.alive_mask = np.ones(len(new_t), dtype=bool)

        self.smoothed_corners = None
        self.outside_streak = 0
        self.rebuild_cooldown = self.REBUILD_COOLDOWN
        return True

    def _check_outside_and_maybe_rebuild(self, corners):
        """Считает точки вне рамки и, если долго плохо — пересоздаёт набор."""
        if corners is None or self.obj_pts_f is None or self.obj_pts_t is None:
            self.outside_streak = 0
            return False

        alive_idx = np.where(self.alive_mask)[0] if self.alive_mask is not None else None
        if alive_idx is None or len(alive_idx) == 0:
            self.outside_streak = 0
            return False

        n_outside = self._count_outside(corners, self.obj_pts_f[alive_idx])
        rebuilt = False

        if n_outside >= self.OUTSIDE_LIMIT:
            self.outside_streak += 1
        else:
            self.outside_streak = 0

        if self.outside_streak >= self.OUTSIDE_STREAK_MAX and \
                self.rebuild_cooldown == 0:
            ok = self._rebuild_points_inside_corners(corners)
            self._log_event(
                "rebuild_points_inside_corners" if ok else "rebuild_points_failed",
                corners=corners,
                n_frame_pts=int(len(self.obj_pts_t)) if self.obj_pts_t is not None else 0,
                note=f"n_outside={n_outside}/{len(alive_idx)} "
                     f"streak={self.outside_streak} ok={ok}"
            )
            rebuilt = ok
            if not ok:
                self.outside_streak = 0

        return rebuilt

    # ============ ОСНОВНОЙ ЦИКЛ ============
    def track(self, frame):
        self.frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h0, w0 = self.target_image.shape[:2]
        obj_quad = np.float32(
            [[0, 0], [0, h0 - 1], [w0 - 1, h0 - 1], [w0 - 1, 0]]
        ).reshape(-1, 1, 2)

        if self.frame_idx == 1:
            corners = obj_quad.reshape(-1, 2).copy()
            self.prev_corners = corners.copy()
            self.last_confirmed_corners = corners.copy()
            self.prev_gray = gray
            self.scale_obj = 1.0
            self._log_event("init", corners=corners, note="first frame")
            return corners.reshape(-1, 1, 2), True

        confirmed = False
        corners = None
        n_frame_pts = 0

        self.prev_obj_pts_f = (self.obj_pts_f.copy()
                               if self.obj_pts_f is not None else None)

        n_pts_total = len(self.obj_pts_t) if self.obj_pts_t is not None else 0
        n_alive = int(self.alive_mask.sum()) if self.alive_mask is not None else 0

        # ----- параметры (учитывают наклон и размытость) -----
        win, max_lvl, reproj_t, fb_t, min_inl_cur, ransac_t_cur, max_area_ratio, max_lost_cur = \
            self._params(self.prev_corners)

        if self.prev_corners is not None:
            a_prev = cv2.contourArea(self.prev_corners.reshape(-1, 2).astype(np.float32))
            a_ref = max((h0 - 1) * (w0 - 1), 1.0)
            if a_prev > 0:
                self.scale_obj = float(np.sqrt(a_prev / a_ref))

        # ========== 1. flow по живым точкам ==========
        flow_ok = False
        n_after_flow = 0
        if (self.obj_pts_f is not None and self.alive_mask is not None
                and self.alive_mask.sum() >= 6):
            alive_idx = np.where(self.alive_mask)[0]
            prev_pts_alive = self.obj_pts_f[alive_idx]

            new_pts, mask = self._track_optical_flow(
                gray, prev_pts_alive,
                win=win, max_level=max_lvl, fb_thresh=fb_t,
            )
            if new_pts is not None and mask.sum() >= 6:
                self.obj_pts_f[alive_idx[mask]] = new_pts[mask]
                fail_idx = alive_idx[~mask]
                if len(fail_idx) > 0:
                    self.alive_mask[fail_idx] = False

                flow_ok = True
                n_after_flow = int(self.alive_mask.sum())
                n_frame_pts = n_after_flow

                if n_after_flow >= 6:
                    src = self.obj_pts_t[self.alive_mask]
                    dst = self.obj_pts_f[self.alive_mask]

                    Hf, maskH = cv2.findHomography(
                        src.reshape(-1, 1, 2),
                        dst.reshape(-1, 1, 2),
                        cv2.RANSAC, reproj_t,
                    )

                    if Hf is not None and maskH is not None and \
                            maskH.sum() >= 0.4 * len(src):
                        cand = cv2.perspectiveTransform(obj_quad, Hf).reshape(-1, 2)
                        if self._is_valid_quad(cand, frame.shape, margin=2.0):
                            corners = cand
                            confirmed = True
            else:
                self._log_event(
                    "flow_lost",
                    corners=self.prev_corners,
                    n_frame_pts=0,
                    note=f"alive={n_alive} scale={self.scale_obj:.2f} q={self.quality:.2f}"
                )
        else:
            self._log_event(
                "flow_skip",
                corners=self.prev_corners,
                n_frame_pts=0,
                note=f"alive={n_alive} scale={self.scale_obj:.2f} q={self.quality:.2f}"
            )

        # ========== 2. жёсткая потеря → ТОЛЬКО сброс к эталону ==========
        if flow_ok and n_pts_total > 0 and n_after_flow < 0.3 * n_pts_total:
            if self.restore_cooldown > 0:
                self._log_event(
                    "reset_skipped_cooldown",
                    corners=corners,
                    n_frame_pts=n_after_flow,
                    note=f"cooldown={self.restore_cooldown} "
                         f"n_before={n_pts_total} alive_after={n_after_flow}"
                )
            else:
                self._reset_to_original_points()
                self.restore_cooldown = 4
                self._log_event(
                    "reset_to_original_points",
                    corners=corners,
                    n_frame_pts=n_after_flow,
                    note=f"hard_loss alive_after={n_after_flow}/{n_pts_total}"
                )

        # ========== 3. матчинг — только для получения углов рамки ==========
        if corners is None or not confirmed:
            best = self._match_all_scales(gray, frame.shape)
            if best is not None:
                _, m_corners, scale, res, (hs, ws) = best
                if res["inliers"] >= max(8, min_inl_cur):
                    corners = m_corners
                    confirmed = True
                    self.smoothed_corners = None
                    self._log_event(
                        "corners_from_match",
                        corners=corners,
                        n_frame_pts=len(res["kp_f"]),
                        note=f"scale={scale} inliers={res['inliers']} q={self.quality:.2f}"
                    )
                else:
                    self._log_event(
                        "corners_from_match_skipped",
                        corners=m_corners,
                        n_frame_pts=len(res["kp_f"]),
                        note=f"scale={scale} inliers={res['inliers']} < {max(8, min_inl_cur)}"
                    )

        # ========== 4. восстановление рамки по живым точкам ==========
        if corners is None and self.alive_mask is not None and self.alive_mask.sum() >= 6:
            src = self.obj_pts_t[self.alive_mask]
            dst = self.obj_pts_f[self.alive_mask]

            Hf, maskH = cv2.findHomography(
                src.reshape(-1, 1, 2),
                dst.reshape(-1, 1, 2),
                cv2.RANSAC, reproj_t,
            )

            if Hf is not None and maskH is not None and \
                    maskH.sum() >= 0.4 * len(src):
                cand = cv2.perspectiveTransform(obj_quad, Hf).reshape(-1, 2)
                if self._is_valid_quad(cand, frame.shape, margin=2.0):
                    corners = cand
                    confirmed = False

        # ========== 5. fallback: 4 угла с прошлого кадра ==========
        if corners is None:
            seed = self.prev_corners if self.prev_corners is not None \
                else self.last_confirmed_corners
            if seed is not None:
                flowed, _ = self._track_optical_flow(
                    gray, seed.reshape(-1, 2),
                    win=win, max_level=max_lvl, fb_thresh=fb_t,
                )
                if flowed is not None and len(flowed) == 4 and \
                        self._is_valid_quad(flowed, frame.shape, margin=0.0):
                    corners = flowed
                    confirmed = False
                    self.fallback_streak += 1
                    self._log_event(
                        "fallback_corners_flow",
                        corners=corners,
                        n_frame_pts=4,
                        note=f"streak={self.fallback_streak}"
                    )
            self.lost_counter += 1

            if corners is None:
                self._log_event(
                    "lost",
                    corners=self.last_confirmed_corners,
                    note=f"lost_counter={self.lost_counter} max={max_lost_cur}"
                )
                if self.lost_counter >= 3 and self.saved_obj_pts_t is not None:
                    self._reset_to_original_points()
                    self._log_event(
                        "reset_to_original_points",
                        corners=self.last_confirmed_corners,
                        note=f"lost_counter={self.lost_counter} scale_obj={self.scale_obj:.2f}"
                    )
        else:
            self.fallback_streak = 0

        # ========== 6. сглаживание + проверка точек вне рамки ==========
        if corners is not None:
            if confirmed:
                corners = self._smooth(corners, alpha=0.7)

            # много точек вне рамки 2 кадра подряд → пересборка по шаблону
            self._check_outside_and_maybe_rebuild(corners)

            self.prev_corners = corners.astype(np.float32)
            self.last_confirmed_corners = corners.copy()

            a_cur = cv2.contourArea(corners.reshape(-1, 2).astype(np.float32))
            a_ref = max((h0 - 1) * (w0 - 1), 1.0)
            if a_cur > 0:
                self.scale_obj = float(np.sqrt(a_cur / a_ref))

        self.prev_gray = gray

        if self.restore_cooldown > 0:
            self.restore_cooldown -= 1
        if self.rebuild_cooldown > 0:
            self.rebuild_cooldown -= 1

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
        alive = tracker.alive_mask
        for i, (x, y) in enumerate(tracker.obj_pts_f):
            xi, yi = int(x), int(y)
            if 0 <= xi < W_f and 0 <= yi < H_f:
                if alive is not None and i < len(alive) and not alive[i]:
                    color = (120, 120, 120)
                else:
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

    n_alive = int(tracker.alive_mask.sum()) if tracker.alive_mask is not None else 0
    n_total = len(tracker.obj_pts_t) if tracker.obj_pts_t is not None else 0
    if confirmed:
        text = (f"match/flow  alive={n_alive}/{n_total}  "
                f"scale={tracker.scale_obj:.2f}  q={tracker.quality:.2f}")
    elif corners is not None:
        text = (f"flow only  alive={n_alive}/{n_total}  "
                f"scale={tracker.scale_obj:.2f}  q={tracker.quality:.2f}")
    else:
        text = f"no tracking  scale={tracker.scale_obj:.2f}  q={tracker.quality:.2f}"
    cv2.rectangle(vis, (0, 0), (W_f, 28), (0, 0, 0), -1)
    cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return vis


def process_video(video_path, out_dir, show=True, log_dir=None):
    print(f"\n{'=' * 60}")
    print(f"Обработка: {os.path.basename(video_path)}")
    print('=' * 60)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    log_path = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, base + "_track.log.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ОШИБКА: не удалось открыть {video_path}")
        return None

    ret, first_frame = cap.read()
    if not ret:
        print("  ОШИБКА: не удалось прочитать первый кадр")
        cap.release()
        return None

    tracker = PlaneTracker(use_sift=True, log_path=log_path)
    try:
        tracker.set_target(first_frame)
    except ValueError as e:
        print(f"  {e}")
        cap.release()
        tracker.close_log()
        return None

    out_path = os.path.join(out_dir, base + "_tracked.avi")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    total = tracked = 0
    tracked_confirmed = 0
    user_quit = False
    show_debug = True

    pending = first_frame

    while True:
        if pending is not None:
            frame = pending
            pending = None
        else:
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
    tracker.close_log()
    if show:
        cv2.destroyAllWindows()

    percent = (tracked / total * 100) if total else 0
    percent_conf = (tracked_confirmed / total * 100) if total else 0
    note = " (прервано)" if user_quit else ""
    print(f"  Кадров: {total}, рамка: {tracked} ({percent:.1f}%), "
          f"подтверждено: {tracked_confirmed} ({percent_conf:.1f}%){note}")
    print(f"  Сохранено: {out_path}")
    if log_path:
        print(f"  Лог: {log_path}")

    return {"file": os.path.basename(video_path),
            "total": total, "tracked": tracked,
            "confirmed": tracked_confirmed,
            "percent": percent, "percent_conf": percent_conf,
            "out": out_path,
            "log": log_path}


def main():
    video_dir = r"D:\prog\Java\pythonProject\skoe\lab_2\test-videos"
    out_dir = r"D:\prog\Java\pythonProject\skoe\lab_2\tracked-output"
    log_dir = os.path.join(out_dir, "logs")

    pattern = os.path.join(video_dir, "mona-lis*.avi")
    video_files = sorted(glob.glob(pattern))

    if not video_files:
        print(f"Не найдено видео по маске: {pattern}")
        return

    print(f"Найдено видео: {len(video_files)}")
    for f in video_files:
        print(f"  - {f}")
    print(f"Результаты будут сохранены в: {out_dir}")
    print(f"Логи будут сохранены в: {log_dir}")

    results = []
    for vp in video_files:
        r = process_video(vp, out_dir, show=True, log_dir=log_dir)
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
    main()
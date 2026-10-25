"""
Lane / active-lane (drivable-area) detection — mengikuti pipeline
BoltTaha/edge-road-perception (src/lane_detection.py + src/visualizer.py).

Grayscale → Gaussian blur → Canny → ROI trapesium (∩ mask cat marka
PUTIH / KUNING) → HoughLinesP →
grouping kiri/kanan berdasarkan slope → fit x = m*y + b per sisi →
EMA smoothing endpoint garis antar frame.

Kalau kedua sisi (kiri + kanan) terdeteksi, area di antaranya = ACTIVE LANE
(jalur kendaraan sendiri) dan di-shade hijau translusen; garis lane digambar
kuning tebal semi-transparan.

Nomor jalur dihitung dari MARKA JALAN (tanda "|"), bukan dari lebar:
    | 1 | 2 | 3 (posisi sekarang) | 4 |
- Titik hilang (vanishing point, VP) = perpotongan garis kiri & kanan
  active lane. Di jalan datar semua marka lajur mengarah ke VP yang sama.
- Hough kedua dijalankan di mask cat marka putih (top-hat: stripe tipis
  terang + saturasi rendah) + kuning (HSV) selebar frame. Segmen yang arahnya searah sinar
  dari VP dianggap marka; posisinya di baris bawah dinormalisasi ke satuan
  lebar active lane: u = (x - x_kiri) / (x_kanan - x_kiri).
  Marka lajur sebelah jatuh di u ≈ -1, -2, ... (kiri) dan 2, 3, ... (kanan),
  karena di perspektif jarak antar marka di satu baris horizontal sama.
- Tiap marka (termasuk garis kiri/kanan active lane) diklasifikasi
  PUTUS-PUTUS vs UTUH dari fraksi cat di sepanjang garisnya (EMA temporal).
  Aturan marka jalan:
    putus-putus  = pemisah antar jalur → masih ada jalur di baliknya
    utuh         = tepi jalan          → hitungan berhenti
  lane_number = 1 + jumlah garis putus-putus berurutan di kiri (mulai dari
                garis kiri active lane),
  total_lanes = lane_number + jumlah garis putus-putus berurutan di kanan.
- Voting temporal: skor kehadiran tiap marka di-decay tiap recompute, jadi
  noise sesaat (truk, bayangan, zebra/hatching) tersaring.
- Marka putih & kuning sama-sama dipakai (`use_yellow=True`); aturan
  utuh/putus-putus berlaku untuk keduanya. `use_yellow=False` = hanya putih.

Tambahan agar cocok dengan pipeline dashcam ini:
- Deteksi jalan di resolusi kerja `work_h` (default 360) lalu koordinat
  di-scale balik ke resolusi frame asli. Parameter piksel Hough
  (threshold/min_line_len/max_line_gap) di-scale relatif terhadap tinggi
  540 px (resolusi video referensi), jadi perilakunya setara referensi.
- Recompute tiap `skip` frame; di antaranya pakai hasil cache.
"""
import cv2
import numpy as np

# Tinggi frame video referensi (Udacity 960x540) — basis scaling parameter piksel
_REF_H = 540.0

# Counting marka
_MARK_MAX_K = 6         # maksimal 6 marka di tiap sisi yang dicek
_MARK_U_TOL = 0.3       # toleransi |u - k| (satuan lebar lane)
_MARK_ANGLE_TOL = 3.0   # beda sudut segmen vs sinar dari VP maks (derajat)
_MARK_VP_MARGIN = 0.15  # abaikan area dekat VP (fraksi jarak VP→bawah frame)
_MARK_SLOPE_MIN = 0.1   # marka samping lebih landai dari ego lane, tapi bukan horizontal
_MARK_TOPHAT = 25       # min kecerahan cat marka di atas aspal sekitarnya (top-hat)
_MARK_ROAD_TOL = 22     # kedua sisi marka harus mirip aspal (selisih gray maks)
_MARK_SIDE_OFF = 0.015  # jarak sampel sisi kiri/kanan marka (fraksi lebar frame)
_MARK_DECAY = 0.9       # decay skor tiap recompute
_MARK_ON = 2.5          # skor minimal marka dianggap ada (steady state max = 10)
_MARK_WHITE_SAT = 55    # saturasi HSV maks cat putih (kuning/oranye/beton berwarna ditolak)
_MARK_SOLID_COV = 0.65  # fraksi cat di sepanjang garis ≥ ini = garis UTUH, < ini = PUTUS-PUTUS
_MARK_DASH_MIN = 0.12   # < ini = hampir tak ada cat (garis salah posisi) → bukan putus-putus


class LaneDetector:
    def __init__(self, work_h: int = 360, skip: int = 2,
                 canny_low: int = 50, canny_high: int = 150,
                 hough_threshold: int = 20, hough_min_line_len: int = 20,
                 hough_max_line_gap: int = 300, slope_min: float = 0.4,
                 roi_top_width_frac: float = 0.12, roi_bottom_width_frac: float = 1.0,
                 roi_top_y_frac: float = 0.60, roi_bottom_y_frac: float = 1.0,
                 smoothing_alpha: float = 0.2, total_lanes: int = None,
                 use_yellow: bool = True, interp: int = cv2.INTER_AREA):
        """`total_lanes` = override jumlah jalur (None = hitung otomatis
        dari marka jalan). `use_yellow` = ikutkan marka kuning sebagai cat
        marka (default True: putih + kuning; False: hanya marka putih)."""
        self.use_yellow = use_yellow
        self.work_h = work_h
        self.total_lanes = total_lanes
        self.skip = max(1, skip)
        self.interp = interp

        self.canny_low = canny_low
        self.canny_high = canny_high
        self.slope_min = slope_min
        self.roi_top_width_frac = roi_top_width_frac
        self.roi_bottom_width_frac = roi_bottom_width_frac
        self.roi_top_y_frac = roi_top_y_frac
        self.roi_bottom_y_frac = roi_bottom_y_frac
        self.alpha = smoothing_alpha

        k = work_h / _REF_H
        self.hough_threshold = max(1, int(round(hough_threshold * k)))
        self.hough_min_line_len = max(1, int(round(hough_min_line_len * k)))
        self.hough_max_line_gap = max(1, int(round(hough_max_line_gap * k)))
        # Hough marka (ROI lebar) — gap lebih kecil agar tidak menyambung
        # objek yang tidak berhubungan di samping jalan
        self.mark_max_line_gap = max(1, int(round(40 * k)))

        self._n = 0
        self.result = None
        self._smoothed_left = None   # (x_bottom, y_bottom, x_top, y_top) di work-res
        self._smoothed_right = None
        self._roi_cache = None       # (sw, mask, polygon)
        self._dilate_kernel = np.ones((5, 5), np.uint8)
        # Per posisi marka k (0 = garis kiri active lane, 1 = garis kanan,
        # k<0 marka di kiri, k>=2 marka di kanan):
        ks = range(-_MARK_MAX_K, _MARK_MAX_K + 2)
        self._mark_score = {k: 0.0 for k in ks if k not in (0, 1)}  # voting kehadiran
        self._mark_u = {k: float(k) for k in ks}                     # posisi aktual (EMA)
        self._mark_cov = {k: None for k in ks}                       # fraksi cat (EMA)

    def _region_of_interest(self, edges: np.ndarray):
        """Mask trapesium (di-cache per lebar kerja)."""
        h, w = edges.shape[:2]
        cache = self._roi_cache
        if cache is None or cache[0] != w:
            top_y = int(h * self.roi_top_y_frac)
            bottom_y = int(h * self.roi_bottom_y_frac)
            top_half_w = (w * self.roi_top_width_frac) / 2
            bottom_half_w = (w * self.roi_bottom_width_frac) / 2
            cx = w / 2
            polygon = np.array([[
                (cx - bottom_half_w, bottom_y),
                (cx - top_half_w, top_y),
                (cx + top_half_w, top_y),
                (cx + bottom_half_w, bottom_y),
            ]], dtype=np.int32)
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, polygon, 255)
            cache = self._roi_cache = (w, mask, polygon)
        return cv2.bitwise_and(edges, cache[1]), cache[2]

    def _paint_mask(self, bgr: np.ndarray, gray: np.ndarray, y0: int, y1: int):
        """Mask cat marka PUTIH di baris y0..y1: stripe tipis terang via
        top-hat horizontal + vertikal (marka jauh di samping tampak landai,
        jadi tipisnya terlihat di arah vertikal) dan saturasi rendah. Aspal/
        beton, sambungan pelat beton (gelap), bayangan, marka kuning, barrier
        oranye & bodi kendaraan berwarna tidak lolos.
        Kalau `use_yellow`, marka kuning (HSV) ikut dimasukkan."""
        mask = np.zeros_like(gray)
        if y1 - y0 < 2:
            return mask
        kw = max(3, int(gray.shape[1] * 0.04)) | 1
        kh = max(3, int(gray.shape[0] * 0.03)) | 1
        band = gray[y0:y1]
        tophat = cv2.max(
            cv2.morphologyEx(band, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))),
            cv2.morphologyEx(band, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))),
        )
        hsv = cv2.cvtColor(bgr[y0:y1], cv2.COLOR_BGR2HSV)
        white = (tophat > _MARK_TOPHAT) & (hsv[..., 1] < _MARK_WHITE_SAT)
        m = white.astype(np.uint8) * 255
        if self.use_yellow:
            m = cv2.bitwise_or(m, cv2.inRange(hsv, (15, 70, 90), (40, 255, 255)))
        mask[y0:y1] = m
        return mask

    @staticmethod
    def _fit_side(segments):
        """Rata-rata segmen (x1,y1,x2,y2) jadi satu garis via linear fit.
        Fit x = m*y + b (lebih stabil dari y = m*x + b untuk garis hampir vertikal)."""
        if not segments:
            return None
        seg = np.asarray(segments, dtype=np.float64)
        xs = np.concatenate([seg[:, 0], seg[:, 2]])
        ys = np.concatenate([seg[:, 1], seg[:, 3]])
        if len(xs) < 2 or np.ptp(ys) == 0:
            return None
        m, b = np.polyfit(ys, xs, 1)
        return m, b

    @staticmethod
    def _on_asphalt(gray, seg, dx, dy, left, right, y0, vy):
        """Marka asli = stripe terang yang KEDUA sisinya aspal. Guardrail,
        tepi jalan/rumput, bodi kendaraan gagal di sini walau arahnya ke VP.
        Warna aspal diambil dari median gray di dalam active lane. Jarak
        sampel sisi ikut perspektif (mengecil ke arah VP) supaya marka jauh
        tidak tersampel sampai ke barrier/tepi jalan di sebelahnya."""
        h, w = gray.shape[:2]
        lxb, yb, lxt, yt = left
        rxb, _, rxt, _ = right
        # Sampel aspal: tengah active lane (jauh dari marka ego), baris y0..yb
        ys = np.linspace(max(y0, yt), yb - 1, 12)
        t = (ys - yb) / (yt - yb)
        cl = lxb + (lxt - lxb) * t
        cr = rxb + (rxt - rxb) * t
        road = np.concatenate([
            gray[ys.astype(int).clip(0, h - 1), (cl + (cr - cl) * f).astype(int).clip(0, w - 1)]
            for f in (0.35, 0.5, 0.65)
        ])
        road_ref = float(np.median(road))

        n = len(seg)
        ok = np.zeros(n, dtype=bool)
        length = np.hypot(dx, dy)
        valid = length > 0
        nx = np.where(valid, -dy / np.where(valid, length, 1), 0)
        ny = np.where(valid, dx / np.where(valid, length, 1), 0)
        ts = np.linspace(0.1, 0.9, 7)[None, :]
        px = seg[:, [0]] + dx[:, None] * ts
        py = seg[:, [1]] + dy[:, None] * ts
        off = np.maximum(1.5, _MARK_SIDE_OFF * w * (py - vy) / max(1.0, yb - vy))

        def sample(ox, oy):
            xi = (px + ox).astype(int).clip(0, w - 1)
            yi = (py + oy).astype(int).clip(0, h - 1)
            return np.median(gray[yi, xi].astype(np.float32), axis=1)

        side_a = sample(nx[:, None] * off, ny[:, None] * off)
        side_b = sample(-nx[:, None] * off, -ny[:, None] * off)
        ok = valid & (np.abs(side_a - road_ref) < _MARK_ROAD_TOL) & \
             (np.abs(side_b - road_ref) < _MARK_ROAD_TOL)
        return ok

    @staticmethod
    def _coverage(paint_d: np.ndarray, vx, vy, xb, yb, y0):
        """Fraksi titik sampel berisi cat di sepanjang garis VP → (xb, yb),
        baris y0..yb. None kalau bagian garis yang masuk frame terlalu pendek."""
        h, w = paint_d.shape[:2]
        ys = np.linspace(y0, yb - 1, 32)
        xs = vx + (xb - vx) * (ys - vy) / (yb - vy)
        inside = (xs >= 0) & (xs < w - 1)
        if inside.sum() < 8:
            return None
        return float((paint_d[ys[inside].astype(int), xs[inside].astype(int)] > 0).mean())

    def _count_markings(self, gray: np.ndarray, paint: np.ndarray, left, right):
        """Update voting + klasifikasi utuh/putus-putus marka, return
        (n_kiri, n_kanan, marka_terpakai [(u, solid)], conf).
        `left`/`right` = (x_bottom, y_bottom, x_top, y_top) active lane di work-res."""
        h, w = gray.shape[:2]
        lxb, yb, lxt, yt = left
        rxb, _, rxt, _ = right
        lane_w = rxb - lxb
        if lane_w <= 1 or yb == yt:
            return None

        # x = m*y + c tiap sisi → vanishing point
        ml, mr = (lxt - lxb) / (yt - yb), (rxt - rxb) / (yt - yb)
        cl, cr = lxb - ml * yb, rxb - mr * yb
        if abs(ml - mr) < 1e-6:
            return None
        vy = (cr - cl) / (ml - mr)
        vx = ml * vy + cl

        hits, hit_u = set(), {}
        y0 = int(max(h * self.roi_top_y_frac, vy + _MARK_VP_MARGIN * (yb - vy)))
        y1 = int(h * self.roi_bottom_y_frac)
        paint = paint.copy()
        paint[:y0] = 0
        lines = cv2.HoughLinesP(
            paint, rho=2, theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.mark_max_line_gap,
        )
        if lines is not None and vy < y0:
            seg = lines.reshape(-1, 4).astype(np.float64)
            x1, y1_, x2, y2 = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
            dx, dy = x2 - x1, y2 - y1_
            # Arah segmen harus searah sinar dari VP ke titik tengah segmen
            rx, ry = (x1 + x2) / 2 - vx, (y1_ + y2) / 2 - vy
            with np.errstate(divide='ignore', invalid='ignore'):
                cos = np.abs(dx * rx + dy * ry) / (np.hypot(dx, dy) * np.hypot(rx, ry))
                slope = np.abs(dy / dx)
                xb = vx + (yb - vy) * rx / ry   # perpanjang sinar VP ke baris bawah
            ok = (dy != 0) & (ry > 0) & (slope >= _MARK_SLOPE_MIN) & \
                 (cos >= np.cos(np.radians(_MARK_ANGLE_TOL)))
            ok &= self._on_asphalt(gray, seg, dx, dy, left, right, y0, vy)
            u = (xb[ok] - lxb) / lane_w
            ks = np.round(u)
            near = np.abs(u - ks) < _MARK_U_TOL
            for k, uu in zip(ks[near].astype(int), u[near]):
                if k in self._mark_score:
                    hits.add(k)
                    hit_u.setdefault(k, []).append(uu)

        for k in self._mark_score:
            self._mark_score[k] = self._mark_score[k] * _MARK_DECAY + (1.0 if k in hits else 0.0)
            if k in hit_u:
                self._mark_u[k] += 0.3 * (float(np.median(hit_u[k])) - self._mark_u[k])

        # Fraksi cat di sepanjang tiap garis (ego + marka yang ada) → utuh / putus-putus
        paint_d = cv2.dilate(paint, self._dilate_kernel)
        for k in self._mark_u:
            if k in (0, 1) or self._mark_score[k] >= _MARK_ON:
                xb = lxb + (0.0 if k == 0 else 1.0 if k == 1 else self._mark_u[k]) * lane_w
                cov = self._coverage(paint_d, vx, vy, xb, yb, y0)
                if cov is not None:
                    prev = self._mark_cov[k]
                    self._mark_cov[k] = cov if prev is None else prev + 0.15 * (cov - prev)

        def solid(k):
            """True = berhenti di sini (garis utuh / tepi jalan). Tak terukur
            atau hampir tanpa cat (garis salah posisi) juga dianggap berhenti."""
            cov = self._mark_cov[k]
            return cov is None or not (_MARK_DASH_MIN <= cov < _MARK_SOLID_COV)

        def chain(ego_k, ks):
            """Dari garis active lane ke luar: tiap garis PUTUS-PUTUS = ada satu
            jalur lagi di baliknya; berhenti di garis UTUH (tepi jalan) atau
            kalau marka berikutnya tidak terlihat."""
            n, used, scores = 0, [], []
            for k in (ego_k, *ks):
                if k != ego_k:
                    if self._mark_score[k] < _MARK_ON:
                        break
                    scores.append(self._mark_score[k])
                    used.append((self._mark_u[k], solid(k)))
                if solid(k):
                    break
                n += 1
            return n, used, scores

        n_left, ul, sl = chain(0, range(-1, -_MARK_MAX_K - 1, -1))
        n_right, ur, sr = chain(1, range(2, _MARK_MAX_K + 2))
        scores = sl + sr
        conf = float(np.mean(scores)) * (1 - _MARK_DECAY) if scores else 1.0
        return n_left, n_right, ul + ur, min(1.0, conf)

    def detect(self, frame: np.ndarray) -> dict:
        """Recompute tiap `skip` pemanggilan; di antaranya pakai cache.

        Return dict (atau None kalau belum ada garis sama sekali):
            left_line / right_line : np.array([x_bottom, y_bottom, x_top, y_top])
                                     di resolusi frame asli, atau None
            active_lane            : True kalau kiri + kanan terdeteksi & tidak menyilang
            lane_number            : jalur ke-n dari kiri (posisi sekarang)
            total_lanes            : jumlah jalur dari marka (atau override)
            confidence             : kekuatan voting marka (0..1)
            markings               : garis marka jalur lain [x_b, y_b, x_t, y_t]
            left_solid/right_solid : garis active lane utuh (tepi jalan) atau putus-putus
            roi_polygon            : polygon ROI di resolusi frame asli
        """
        self._n += 1
        if self.result is not None and (self._n - 1) % self.skip:
            return self.result

        h, w = frame.shape[:2]
        scale = self.work_h / h
        sw = max(2, int(w * scale))
        small = cv2.resize(frame, (sw, self.work_h), interpolation=self.interp)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        masked_edges, roi_polygon = self._region_of_interest(edges)

        # Hanya tepi yang berasal dari cat marka putih — sambungan pelat
        # beton, retakan, bayangan & marka kuning tidak ikut jadi kandidat
        paint = self._paint_mask(small, blurred, int(self.work_h * self.roi_top_y_frac),
                                 int(self.work_h * self.roi_bottom_y_frac))
        masked_edges = cv2.bitwise_and(masked_edges, cv2.dilate(paint, self._dilate_kernel))

        raw_lines = cv2.HoughLinesP(
            masked_edges, rho=2, theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap,
        )

        left_segments, right_segments = [], []
        if raw_lines is not None:
            for x1, y1, x2, y2 in raw_lines.reshape(-1, 4):
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < self.slope_min:
                    continue  # hampir horizontal, bukan batas lane
                if slope < 0:
                    left_segments.append((x1, y1, x2, y2))
                else:
                    right_segments.append((x1, y1, x2, y2))

        y_bottom = int(self.work_h * self.roi_bottom_y_frac)
        y_top = int(self.work_h * self.roi_top_y_frac)

        for segments, attr in ((left_segments, '_smoothed_left'),
                               (right_segments, '_smoothed_right')):
            fit = self._fit_side(segments)
            if fit is None:
                continue  # tahan garis terakhir kalau sisi ini tidak terdeteksi
            m, b = fit
            candidate = (m * y_bottom + b, y_bottom, m * y_top + b, y_top)
            prev = getattr(self, attr)
            if prev is None:
                smoothed = candidate
            else:
                smoothed = tuple(self.alpha * c + (1 - self.alpha) * p
                                 for c, p in zip(candidate, prev))
            setattr(self, attr, smoothed)

        if self._smoothed_left is None and self._smoothed_right is None:
            return self.result

        inv = 1.0 / scale

        def to_full(line):
            if line is None:
                return None
            return np.round(np.asarray(line) * inv).astype(int)

        left_line = to_full(self._smoothed_left)
        right_line = to_full(self._smoothed_right)
        # Active lane valid hanya kalau garis kiri ada di kiri garis kanan di
        # sepanjang ROI (tidak menyilang / bentuk "X")
        active = left_line is not None and right_line is not None and \
            left_line[0] < right_line[0] and left_line[2] < right_line[2]

        # Nomor jalur dari marka — kalau active lane belum ada, pakai hasil terakhir
        prev = self.result or {}
        lane_no = prev.get('lane_number', 1)
        total = prev.get('total_lanes', self.total_lanes or 1)
        conf = prev.get('confidence', 0.0)
        markings = prev.get('markings', [])
        ego_style = {k: prev.get(k) for k in ('left_solid', 'right_solid')}
        counted = self._count_markings(blurred, paint, self._smoothed_left, self._smoothed_right) \
            if active else None
        if counted is not None:
            n_left, n_right, used, conf = counted
            lane_no = n_left + 1
            total = lane_no + n_right
            if self.total_lanes:
                total = self.total_lanes
                lane_no = min(lane_no, total)

            # Garis marka (lewat VP) untuk visualisasi |1|2|3|4|
            (lxb, yb, lxt, yt), (rxb, _, rxt, _) = self._smoothed_left, self._smoothed_right
            markings = [
                np.round(np.array([lxb + u * (rxb - lxb), yb,
                                   lxt + u * (rxt - lxt), yt]) * inv).astype(int)
                for u, _ in used
            ]
            ego_style = {'left_solid': n_left == 0, 'right_solid': n_right == 0}

        self.result = {
            'lane_number': lane_no, 'confidence': conf, 'total_lanes': total,
            'left_line': left_line,
            'right_line': right_line,
            'active_lane': active,
            'markings': markings,
            **ego_style,
            'roi_polygon': np.round(roi_polygon * inv).astype(np.int32),
        }
        return self.result


def draw_lane(frame: np.ndarray, result: dict, draw_roi: bool = False,
              draw_lines: bool = True, draw_fill: bool = False) -> np.ndarray:
    """Gambar garis lane + marka jalur lain (`draw_lines`, default on) dan
    nomor jalur di tengah ACTIVE LANE secara in-place. Shading hijau active
    lane (`draw_fill`) default off — visual sama dengan visualizer.draw_lanes
    di referensi."""
    if not result:
        return frame

    lane_lines = [l for l in (result.get('left_line'), result.get('right_line')) if l is not None]
    if not lane_lines:
        return frame

    # Garis lane: kuning tebal, blend 40%; marka jalur lain: putih tipis
    if draw_lines:
        overlay = frame.copy()
        for x1, y1, x2, y2 in result.get('markings', []):
            cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
        for x1, y1, x2, y2 in lane_lines:
            cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 8)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, dst=frame)

    # Active lane — hanya kalau kiri + kanan ada
    if result.get('active_lane'):
        (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2) = lane_lines
        if draw_fill:  # drivable area: hijau, blend 15%
            poly = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]], dtype=np.int32)
            drivable = frame.copy()
            cv2.fillPoly(drivable, [poly], (0, 200, 0))
            cv2.addWeighted(drivable, 0.15, frame, 0.85, 0, dst=frame)

        # Nomor jalur di tengah active lane, sedikit di atas bawah frame
        t = 0.75
        cx = int((lx1 + (lx2 - lx1) * t + rx1 + (rx2 - rx1) * t) / 2)
        cy = int(ly1 + (ly2 - ly1) * t)
        label = str(result.get('lane_number', ''))
        # Ukuran ikut resolusi (1.2 @ 540p); outline hitam supaya terbaca di beton terang
        fs = 1.2 * frame.shape[0] / 540.0
        thick = max(2, int(round(3 * frame.shape[0] / 540.0)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        org = (cx - tw // 2, cy + th // 2)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick + 4, cv2.LINE_AA)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick, cv2.LINE_AA)

    if draw_roi and result.get('roi_polygon') is not None:
        cv2.polylines(frame, [result['roi_polygon']], isClosed=True,
                      color=(255, 0, 0), thickness=1)
    return frame

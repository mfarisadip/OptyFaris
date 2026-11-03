"""
Lane / active-lane detection berbasis TITIK HILANG (vanishing point, VP) yang
dikalibrasi otomatis dari video — tanpa ROI trapesium tetap, jadi tetap jalan
untuk posisi kamera / horizon / kap mesin / dashboard yang berbeda-beda.

Pipeline tiap recompute (di resolusi kerja `work_h`):
1. Fitur cat marka: filter stripe terang-di-antara-gelap per baris (SLT),
   jarak sampel ikut perspektif (makin dekat VP makin tipis), di kanal gray
   (marka putih) + kanal kekuningan (marka kuning — tetap kontras di beton
   terang). NMS per baris → titik pusat stripe (x, y, kontras).
2. VP: segmen Hough dari stripe kuat divoting ke peta kandidat yang
   diakumulasi antar frame. Selama VP belum "dipercaya", beberapa puncak
   peta diuji dengan bukti marka (jumlah baris bercat sepanjang sinar dari
   VP) — pantulan dashboard / kap mesin yang statis tidak menang di sini.
   Setelah dipercaya, VP diperhalus lokal (vy lambat, vx ikut tikungan).
3. Tiap titik stripe diproyeksikan sepanjang sinar dari VP ke baris bawah
   frame: u = posisi garis di baris bawah. Marka = puncak histogram u (Hough
   1-D lewat VP); dukungan = baris yang punya cat di sepanjang sinar.
4. Ego lane = pasangan marka terdekat kiri & kanan pusat kendaraan dengan
   lebar lajur masuk akal (prior geometri K*(H - vy), lalu hasil belajar).
5. Batas bawah jalan (kap mesin / dashboard) dari profil dukungan marka ego
   per baris: di bawah kap tidak pernah ada cat yang searah sinar VP.
6. Garis di-fit ulang dari titik cat di sekitar sinar (ikut tikungan:
   polyline kuadrat), diperhalus antar frame (reset saat pindah lajur).
7. Nomor jalur: marka lain diberi indeks k dari posisinya relatif ego lane
   (diukur di baris tempat catnya terlihat): k = 0 garis kiri ego, 1 kanan,
   -1, -2, ... ke kiri, 2, 3, ... ke kanan. Tiap garis diklasifikasi
   PUTUS-PUTUS (pemisah lajur) vs UTUH (tepi jalan) dari CELAH yang terlihat:
   baris yang aspalnya tampak polos tanpa cat. Baris tertutup kendaraan,
   kotak teks overlay, kap mesin, atau di luar frame tidak dihitung, jadi
   garis tepi yang tertutup sebagian tidak terbaca putus-putus.
       tepi kiri | jalur 1 | jalur 2 | jalur 3 | tepi kanan
   jalur = 1 + jumlah garis putus-putus berurutan dari garis kiri ego ke
   kiri (berhenti di garis utuh / tak terlihat); total = jalur + jumlah
   garis putus-putus berurutan dari garis kanan ego ke kanan.
"""
import cv2
import numpy as np

_STRIPE_FRAC = 0.06        # jarak sampel SLT ≈ fraksi lebar lajur di baris itu
_RESP_MIN = 10             # kontras min stripe di atas KEDUA sisinya (gray level)
_VP_RESP = 20              # kontras min stripe untuk segmen voting VP
_SAT_MAX = 70              # chroma maks titik "putih" (kuning lewat kanal kuning)
_SEG_SLOPE = (0.08, 8.0)   # |dx/dy| segmen yang ikut voting VP
_VP_TRUST = 3              # deteksi berturut-turut berbukti (VP stabil) sebelum dipercaya
_VP_SEARCH_TOP = 0.2       # VP belum dipercaya: cari stripe dari 20% tinggi frame ke bawah
_VP_CHALLENGE = 8          # VP dipercaya: tiap N recompute uji ulang kandidat lain
_FAR_BAD_MAX = 10          # VP dianggap basi setelah N recompute sinar tak cocok di bagian jauh
_ROAD_CHROMA = 40          # permukaan jalan: abu-abu (chroma rendah) ...
_ROAD_TEX = 60             # ... dan halus (bukan dedaunan / bodi kendaraan)
_NEAR_FRAC = 0.22          # baris histogram: mulai 22% jarak VP→bawah jalan
_TOP_FRAC = 0.06           # garis digambar sampai 6% jarak VP→bawah jalan
_SUP_MIN = 0.14            # fraksi baris bercat marka "kuat"
_SUP_WEAK = 0.05           # marka lemah (mis. 1 strip putus-putus) — hanya lewat pasangan
_MARK_CONTRAST = 40        # kontras (persentil 90) cat marka "penuh"; di bawahnya kualitas turun
_STRONG_CON = 25           # kontras min marka kuat
_LW_K = 2.6                # prior lebar lajur di baris bawah ≈ K * (H - vy)
_LW_K_RANGE = (1.3, 3.6)   #   (lebar lajur / tinggi kamera; jalan datar)
_MAX_MISS = 6              # recompute tanpa bukti sebelum sisi dianggap hilang
_MAX_MISS_PRED = 30        #   idem kalau sisi lain terlihat (posisi diprediksi dari lebar lajur)
_N_SAMP = 12               # titik polyline per garis
_SMOOTH_PX = 15.0          # smoothing adaptif: pergeseran (px kerja) yang langsung diikuti penuh
_SMOOTH_U = 0.12           #   idem untuk posisi u (fraksi lebar lajur)
_GAP_SOLID = 0.2           # fraksi celah (EMA) ≤ ini = garis UTUH (tepi jalan; terukur ≤ 0.16)
_GAP_DASH = 0.3            # ≥ ini = PUTUS-PUTUS (pemisah lajur; terukur ≥ 0.35); di antaranya status lama
_GAP_NOPAINT = 0.9         # hampir tanpa cat = bukan marka
_GAP_MIN_ROWS = 5          # baris terlihat (cat + celah, berbobot) minimal untuk update
_GAP_ALPHA = 0.15
_GAP_NEAR = 0.2            # baris uji celah mulai 20% jarak VP→bawah jalan (bagian jauh: garis
                           #   tipis & lintasan bisa melenceng dari cat → celah palsu)
_MARK_MAX_K = 6
_MARK_DECAY = 0.95         # marka lain yang posisinya terlihat tapi tak terdeteksi meluruh (~2.5 dtk
_MARK_ON = 2.5             #   dari steady state 1 / (1 - decay) ke _MARK_ON); tertutup kendaraan / di
                           #   luar frame / di balik overlay = ditahan sampai terlihat lagi, pindah
                           #   lajur, atau reset


def _lsq1(y, x, wt):
    """Fit berbobot x = a*y + b (closed form, lebih ringan dari np.polyfit)."""
    sw = wt.sum()
    my, mx = (wt * y).sum() / sw, (wt * x).sum() / sw
    dy = y - my
    var = (wt * dy * dy).sum()
    a = (wt * dy * (x - mx)).sum() / var if var > 1e-9 else 0.0
    return np.array([a, mx - a * my])


class LaneDetector:
    def __init__(self, work_h: int = 360, skip: int = 2,
                 roi_top_y_frac: float = None, roi_bottom_y_frac: float = None,
                 smoothing_alpha: float = 0.35, total_lanes: int = None,
                 use_yellow: bool = True, interp: int = cv2.INTER_AREA, **_legacy):
        """`roi_top_y_frac` / `roi_bottom_y_frac` = batas manual opsional
        (fraksi tinggi frame); None = otomatis dari VP & kap mesin.
        `total_lanes` = override jumlah jalur. `use_yellow` = marka kuning ikut."""
        self.work_h = work_h
        self.skip = max(1, skip)
        self.interp = interp
        self.alpha = smoothing_alpha
        self.total_lanes = total_lanes
        self.use_yellow = use_yellow
        self.roi_top_y_frac = roi_top_y_frac
        self.roi_bottom_y_frac = roi_bottom_y_frac
        self._n = 0
        self.result = None
        self._wh = None
        self._reset_state()

    def _reset_state(self):
        self._vp = None            # (vx, vy) resolusi kerja
        self._vp_ok = 0            # bukti marka berturut-turut untuk VP sekarang
        self._vp_n = 0             # recompute sejak VP dipercaya (jadwal uji ulang)
        self._vp_chal = 0          # kemenangan berturut-turut kandidat penantang
        self._far_bad = 0          # recompute berturut-turut: garis dekat jelas, sinar jauh tidak cocok
        self._vp_grid = None       # grid kandidat VP kasar (x, y)
        self._vp_acc = None        # peta voting VP terakumulasi ('/', '\\')
        self._row_sup = None       # EMA dukungan marka ego per baris (kap/dashboard)
        self._row_n = 0
        self._road_bot = None      # baris bawah jalan (di atas kap/dashboard)
        self._reset_lanes()

    def _reset_lanes(self):
        self._lwk = None           # lebar lajur / (H - vy): ~ lebar lajur / tinggi kamera
        self._xc = None            # pusat kendaraan di baris bawah
        self._ego = [None, None]   # u kiri/kanan (diperhalus)
        self._miss = [0, 0]
        self._samp = [None, None]  # x polyline diperhalus per sisi
        self._samp_ty = None       # baris (y) sampel polyline
        self._geo = None           # (vx, vy, y_top, y_bot) recompute terakhir
        self._vxc = None           # vx per frame terakhir (VP bagian dekat, ikut tikungan)
        self._reset_votes()

    # ── fitur cat marka ───────────────────────────────────────────────────
    def _stripes(self, gray, ych, bgr, y0, y1, vy, lw_px):
        """Titik pusat stripe cat di baris y0..y1 → (x, y, kontras, mask_vp).
        SLT: r(x) = min(I(x) - I(x-d), I(x) - I(x+d)); positif hanya untuk
        stripe yang lebih sempit dari ~2d (tepi aspal/barrier/bodi lebar gagal).
        `vy` None (VP belum dipercaya) → multi-skala d ∈ {2, 4, 7, 11}; kalau
        VP ada, d per baris ikut perspektif (lebih selektif & lebih murah)."""
        h, w = gray.shape
        ys = np.arange(y0, y1)
        if vy is None:
            plan = [(d, 0, len(ys)) for d in (2, 4, 7, 11)]
        else:
            d = np.clip(np.round(_STRIPE_FRAC * lw_px * (ys - vy) / max(1.0, h - vy)) + 1, 2, 18).astype(int)
            # d naik monoton ke bawah → tiap nilai d = satu blok baris kontigu
            cut = np.nonzero(np.diff(d))[0] + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, len(d)]
            plan = [(int(d[a]), int(a), int(e)) for a, e in zip(starts, ends)]
        resp = np.zeros((len(ys), w), np.int16)
        chans = [gray] + ([ych] if self.use_yellow else [])
        for ci, ch in enumerate(chans):
            band = ch[y0:y1].astype(np.int16)
            for dv, a, e in plan:
                if 2 * dv >= w or e <= a:
                    continue
                g = band[a:e]
                c = g[:, dv:w - dv]
                r = np.minimum(c - g[:, :w - 2 * dv], c - g[:, 2 * dv:])
                if ci:  # kanal kuning: skala ke gray-level setara
                    r <<= 1
                np.maximum(resp[a:e, dv:w - dv], r, out=resp[a:e, dv:w - dv])
        pk = (resp[:, 1:-1] >= _RESP_MIN) & (resp[:, 1:-1] >= resp[:, :-2]) & (resp[:, 1:-1] > resp[:, 2:])
        yy, xx = np.nonzero(pk)
        xx = xx + 1
        r = resp[yy, xx].astype(np.float32)
        yy = yy + y0
        # marka putih harus netral (kuning lolos lewat kanal kuning)
        px = bgr[yy, xx].astype(np.int16)
        chroma = px.max(axis=1) - px.min(axis=1)
        yel = ych[yy, xx] > 25 if self.use_yellow else np.zeros(len(yy), bool)
        ok = (chroma < _SAT_MAX) | yel
        xx, yy, r = xx[ok], yy[ok], r[ok]
        # Garis vertikal sempurna (tepi kotak overlay/watermark, tiang) menumpuk
        # di satu kolom; marka lajur miring menyebar ke banyak kolom (kecuali
        # tepat saat pindah lajur). Kolom dengan run panjang → buang.
        col = np.bincount(xx, minlength=w)
        col3 = col + np.r_[0, col[:-1]] + np.r_[col[1:], 0]
        vert = col3 >= max(25, 0.12 * (y1 - y0))
        vmask = resp >= _VP_RESP
        if vert.any():
            keep = ~vert[xx]
            xx, yy, r = xx[keep], yy[keep], r[keep]
            vcols = np.nonzero(vert)[0]
            vmask[:, vcols] = False
        return xx.astype(np.float32), yy.astype(np.float32), r, vmask

    # ── VP ────────────────────────────────────────────────────────────────
    @staticmethod
    def _segments(mask, y_off):
        """Segmen Hough dari mask stripe kuat → parameter garis untuk voting."""
        if mask.shape[0] < 4:
            return None
        segs = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 1, np.pi / 180, threshold=10,
                               minLineLength=max(8, int(mask.shape[1] * 0.02)), maxLineGap=3)
        if segs is None:
            return None
        s = segs.reshape(-1, 4).astype(np.float32)
        s[:, [1, 3]] += y_off
        dx, dy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
        ln = np.hypot(dx, dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.abs(dx / dy)
        ok = (dy != 0) & (inv > _SEG_SLOPE[0]) & (inv < _SEG_SLOPE[1])
        if ok.sum() < 3:
            return None
        s, dx, dy, ln = s[ok], dx[ok], dy[ok], ln[ok]
        nx, ny = dy / ln, -dx / ln
        c = -(nx * s[:, 0] + ny * s[:, 1])
        return nx, ny, c, np.minimum(ln, 60), np.minimum(s[:, 1], s[:, 3]), dx / dy < 0

    @staticmethod
    def _votes(segs, gx, gy, sig):
        """Dukungan VP di grid kandidat dari segmen '/' dan '\\' terpisah."""
        nx, ny, c, ln, ymin, left = segs
        GX, GY = np.meshgrid(gx, gy)
        cx, cy = GX.ravel()[:, None], GY.ravel()[:, None]
        dist = np.abs(nx[None] * cx + ny[None] * cy + c[None])
        wv = np.clip(1 - dist / sig, 0, None) * ln[None]
        wv *= (cy < ymin[None] - 4)          # VP harus di atas segmen
        sl = (wv * left[None]).sum(1)
        sr = (wv * ~left[None]).sum(1)
        return sl.reshape(GX.shape), sr.reshape(GX.shape)

    def _vp_accumulate(self, segs, w, h):
        if self._vp_grid is None:
            self._vp_grid = (np.arange(0.05 * w, 0.95 * w, 8, dtype=np.float32),
                             np.arange(0.02 * h, 0.85 * h, 8, dtype=np.float32))
            n = (len(self._vp_grid[1]), len(self._vp_grid[0]))
            self._vp_acc = [np.zeros(n, np.float32), np.zeros(n, np.float32)]
        self._vp_acc[0] *= 0.93
        self._vp_acc[1] *= 0.93
        if segs is not None:
            sl, sr = self._votes(segs, *self._vp_grid, 8.0)
            self._vp_acc[0] += sl
            self._vp_acc[1] += sr

    def _vp_peaks(self, k=4):
        """Puncak lokal teratas peta VP terakumulasi (butuh dukungan '/' DAN '\\')."""
        score = np.minimum(*self._vp_acc) * 2
        sc = score.copy()
        gx, gy = self._vp_grid
        out = []
        for _ in range(k):
            i = np.unravel_index(int(np.argmax(sc)), sc.shape)
            if sc[i] < 60:
                break
            out.append((float(gx[i[1]]), float(gy[i[0]])))
            sc[max(0, i[0] - 3):i[0] + 4, max(0, i[1] - 3):i[1] + 4] = 0
        return out

    def _vp_refine(self, segs, cx, cy, rx, ry, step=2.0):
        """Puncak voting halus di sekitar (cx, cy) dari segmen frame ini."""
        if segs is None:
            return None
        fx = np.arange(cx - rx, cx + rx + 1, step, dtype=np.float32)
        fy = np.arange(cy - ry, cy + ry + 1, step, dtype=np.float32)
        fl, fr = self._votes(segs, fx, fy, 3.0)
        fs = np.minimum(fl, fr) * 2
        j = np.unravel_index(int(np.argmax(fs)), fs.shape)
        if fs[j] < 30:
            return None
        return float(fx[j[1]]), float(fy[j[0]])

    # ── geometri sinar dari VP ────────────────────────────────────────────
    @staticmethod
    def _ray_x(u, y, vx, vy, h):
        return vx + (u - vx) * (y - vy) / (h - vy)

    def _hist_peaks(self, xs, ys, rs, vx, vy, h, w, y_lo, y_hi, wide=True):
        """Puncak histogram u (posisi marka di baris bawah) + vx yang paling
        mempertajam puncak (koreksi VP kecil / tikungan)."""
        sel = (ys >= y_lo) & (ys <= y_hi)
        x, y, r = xs[sel], ys[sel], rs[sel]
        if len(x) < 10:
            return [], vx
        f = (h - vy) / (y - vy)
        wt = np.minimum(r, 50) / 50.0
        lo, hi, b = -w, 2 * w, 2.0
        nb = int((hi - lo) / b)
        def hist(vxx):
            bi = ((vxx + (x - vxx) * f - lo) / b).astype(np.int32)
            ok = (bi >= 0) & (bi < nb)
            hh = np.bincount(bi[ok], weights=wt[ok], minlength=nb).astype(np.float32)
            hh = cv2.GaussianBlur(hh[None], (0, 0), 1.5)[0]
            return float(np.dot(hh, hh)), vxx, hh

        # vx per frame (tikungan menggeser VP bagian dekat): kasar ±20% lalu halus
        # lebar (search): ±20% kasar; sempit (pelacakan, VP dipercaya): ±10% di
        # sekitar vx frame sebelumnya — tikungan berubah halus antar frame
        span = 0.2 if wide else 0.1
        best = max((hist(vx + d) for d in np.linspace(-span * w, span * w, 9)), key=lambda t: t[0])
        vx = best[1]
        best = max((hist(vx + d) for d in np.linspace(-0.025 * w, 0.025 * w, 5)), key=lambda t: t[0])
        _, vx2, hh = best
        thr = 0.02 * max(1.0, y_hi - y_lo)
        loc = (hh[1:-1] >= hh[:-2]) & (hh[1:-1] > hh[2:]) & (hh[1:-1] > thr)
        peaks = []
        for i in sorted(np.nonzero(loc)[0] + 1, key=lambda i: -hh[i]):
            a, c0, d = hh[i - 1], hh[i], hh[i + 1]
            den = a - 2 * c0 + d
            off = 0.5 * (a - d) / den if den < 0 else 0.0   # sub-bin (parabola)
            u = lo + (i + 0.5 + off) * b
            # puncak berdekatan (garis lebar / marka ganda) = satu marka
            if all(abs(u - p) > 0.03 * w for p in peaks):
                peaks.append(u)
        return sorted(peaks), vx2

    def _support(self, u, xs, ys, rs, vx, vy, h, w, y_lo, y_hi, lw, profile=False, contrast=False):
        """Dukungan cat sepanjang sinar u di baris y_lo..y_hi (yang di dalam
        frame) → (fraksi baris, jumlah baris, titik pendukung). `profile` =
        juga (baris hit, baris teruji) sampai bawah frame (deteksi kap)."""
        y_end = h - 1 if profile else y_hi
        r0, r1 = int(np.ceil(y_lo)), int(y_end)
        sel = (ys >= r0) & (ys <= r1)
        x, y, r = xs[sel], ys[sel], rs[sel]
        if r1 < r0 or len(x) == 0:
            return (None, 0, None, None) if (profile or contrast) else (None, 0, None)
        rows = np.arange(r0, r1 + 1)
        xr = self._ray_x(u, rows, vx, vy, h)
        inside = (xr >= 1) & (xr < w - 1)
        in_fit = inside & (rows <= y_hi)
        tol = np.maximum(1.5, 0.07 * lw * (y - vy) / (h - vy))
        ok = np.abs(x - self._ray_x(u, y, vx, vy, h)) < tol
        fit = ok & (y <= y_hi)
        if fit.sum() >= 6:
            # noise tekstur aspal / pantulan jauh lebih lemah dari cat di sinar yang sama
            ok &= r >= 0.4 * np.median(r[fit])
            fit = ok & (y <= y_hi)
        hit = np.zeros(len(rows), bool)
        hit[y[ok].astype(int) - r0] = True
        n_fit = int(in_fit.sum())
        n_hit = int((hit & in_fit).sum())
        frac = None if n_fit < 6 else n_hit / float(n_fit)
        if profile:
            return frac, n_hit, (x[fit], y[fit]), (rows[hit & inside], rows[inside])
        if contrast:
            return frac, n_hit, (x[fit], y[fit]), (float(np.percentile(r[fit], 90)) if fit.any() else 0.0)
        return frac, n_hit, (x[fit], y[fit])

    def _lane_w(self, h, vy):
        """Lebar lajur (satuan u, baris bawah frame) untuk VP sekarang."""
        return (self._lwk if self._lwk is not None else _LW_K) * (h - vy)

    def _pick_ego(self, cand, w, h, vy):
        """Pilih pasangan marka (kiri, kanan) ego lane dari kandidat
        (u, fraksi, n_baris, titik). Lebar lajur harus masuk akal: hasil
        belajar (kalau ada) atau prior geometri K*(H - vy). Marka kuat di
        antara pasangan = pasangan salah (ego lane dibatasi marka TERDEKAT).
        Marka lemah hanya dipakai sebagai bagian pasangan yang lebarnya cocok."""
        xc = self._xc if self._xc is not None else w / 2.0
        lefts = sorted([c for c in cand if c[0] < xc], key=lambda c: -c[0])[:4]
        rights = sorted([c for c in cand if c[0] >= xc], key=lambda c: c[0])[:4]
        # batas keras = prior geometri; lebar hasil belajar hanya penalti lunak,
        # jadi tetap bisa belajar ulang (ramp, lajur menyempit, VP bergeser)
        ref = self._lane_w(h, vy)
        lo, hi = _LW_K_RANGE[0] * (h - vy), _LW_K_RANGE[1] * (h - vy)

        def strong(c):
            return c[1] >= _SUP_MIN and c[5] >= _STRONG_CON

        best, best_s = None, -1e9
        for i, L in enumerate(lefts):
            for j, R in enumerate(rights):
                wd = R[0] - L[0]
                if not lo <= wd <= hi:
                    continue
                # marka kuat yang dilewati (lebih dekat ke kendaraan) → kemungkinan pasangan salah
                skip = sum(c[4] for c in lefts[:i] + rights[:j] if strong(c))
                dev = abs(wd / ref - 1)
                sc = L[4] + R[4] - 0.6 * dev - 1.5 * max(0.0, dev - 0.3) - 0.7 * skip
                for k, c in ((0, L), (1, R)):   # kontinuitas dengan frame sebelumnya
                    if self._ego[k] is not None and abs(c[0] - self._ego[k]) < 0.1 * ref:
                        sc += 0.25
                if sc > best_s:
                    best, best_s = [L, R], sc
        if best is not None:
            return best
        # Tidak ada pasangan: sisi tunggal hanya dari marka kuat terdekat
        # Tidak ada pasangan valid: sisi tunggal dari marka kuat terdekat — kalau
        # sedang melacak, hanya yang dekat posisi sebelumnya (dash hilang sesaat
        # jangan sampai melompat ke marka lajur sebelah)
        found = [None, None]
        for k, side in ((0, lefts), (1, rights)):
            st = [c for c in side if strong(c)]
            if self._ego[k] is not None:
                st = [c for c in st if abs(c[0] - self._ego[k]) < 0.2 * ref]
            if st:
                found[k] = st[0]
        if found[0] is not None and found[1] is not None and \
                not lo <= found[1][0] - found[0][0] <= hi:
            found[1 if found[0][4] >= found[1][4] else 0] = None   # lebar tidak masuk akal
        return found

    def _lane_hyp(self, xs, ys, rs, vx, vy, h, w, road_bot, vx_curve=None):
        """Bukti marka untuk VP (vx, vy): puncak, kandidat, pasangan ego, skor."""
        if road_bot - vy < 20:
            return None
        y_lo = vy + _NEAR_FRAC * (road_bot - vy)
        y_top = vy + _TOP_FRAC * (road_bot - vy)
        if self.roi_top_y_frac:
            y_top = min(max(y_top, h * self.roi_top_y_frac), road_bot - 16)
        if vx_curve is not None:
            peaks, vx = self._hist_peaks(xs, ys, rs, vx_curve, vy, h, w, y_lo, road_bot - 1, wide=False)
        else:
            peaks, vx = self._hist_peaks(xs, ys, rs, vx, vy, h, w, y_lo, road_bot - 1)
        lw = self._lane_w(h, vy)
        cand = []
        for u in peaks:
            frac, n, pts, con = self._support(u, xs, ys, rs, vx, vy, h, w, y_top, road_bot - 1, lw, contrast=True)
            if frac is not None and frac >= _SUP_WEAK:
                # q = dukungan x kontras: cat marka asli kontras tinggi; sambungan
                # perkerasan / jejak ban / retak sealant kontras rendah
                cand.append((u, frac, n, pts, frac * min(1.0, con / _MARK_CONTRAST), con))
        found = self._pick_ego(cand, w, h, vy)
        both = found[0] is not None and found[1] is not None
        # VP benar → sinar juga menempel ke cat di sepertiga JAUH pita (VP yang
        # basi/terlalu tinggi hanya cocok di bagian dekat)
        y_far = y_top + (road_bot - 1 - y_top) / 3.0
        far_ok = any(f is not None and (self._support(f[0], xs, ys, rs, vx, vy, h, w, y_top, y_far, lw)[0] or 0) >= 0.1
                     for f in found)
        road = self._road_like(found, vx, vy, h, w, road_bot) if both else 0.0
        others = sorted([c[2] for c in cand if all(c is not f for f in found)], reverse=True)[:3]
        score = ((found[0][2] * min(1.0, found[0][5] / _MARK_CONTRAST) +
                  found[1][2] * min(1.0, found[1][5] / _MARK_CONTRAST)) * road if both else
                 0.25 * max([f[2] for f in found if f is not None], default=0)) + 0.25 * sum(others)
        return {'vx': vx, 'vy': vy, 'y_top': y_top, 'peaks': peaks, 'cand': cand,
                'found': found, 'score': score, 'road': road, 'far_ok': far_ok,
                'strong': both and min(found[0][1], found[1][1]) >= _SUP_MIN and
                min(found[0][5], found[1][5]) >= _STRONG_CON and road >= 0.5,
                # cukup untuk mempercayai VP: satu sisi garis panjang jelas, sisi lain
                # boleh pudar (putus-putus di beton), pasangan di atas permukaan jalan
                'ok': both and road >= 0.5 and min(found[0][1], found[1][1]) >= 0.08 and
                any(f[1] >= 0.3 and f[5] >= _STRONG_CON for f in found)}

    def _road_like(self, found, vx, vy, h, w, road_bot):
        """Fraksi sampel di dalam ego lane (pita tengah jalan) yang tampak
        seperti permukaan jalan: abu-abu & halus. Garis "lajur" palsu dari
        pohon, langit, kotak overlay, bodi kendaraan gagal di sini."""
        bgr, tex, _ = self._cur
        yy = np.linspace(vy + 0.3 * (road_bot - vy), vy + 0.7 * (road_bot - vy), 6)
        xl = self._ray_x(found[0][0], yy, vx, vy, h)
        xr = self._ray_x(found[1][0], yy, vx, vy, h)
        f = np.linspace(0.15, 0.85, 8)[None, :]
        px = (xl[:, None] + (xr - xl)[:, None] * f).ravel()
        py = np.repeat(yy, 8)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if ok.sum() < 12:
            return 0.0
        px, py = px[ok].astype(int), py[ok].astype(int)
        c = bgr[py, px].astype(np.int16)
        chroma = c.max(axis=1) - c.min(axis=1)
        return float(np.mean((chroma < _ROAD_CHROMA) & (tex[py, px] < _ROAD_TEX)))

    # ── kap mesin / dashboard ────────────────────────────────────────────
    def _update_road_bottom(self, hit_rows, tested_rows, h):
        """Batas bawah jalan dari profil dukungan marka ego per baris (EMA).
        Di bawah kap mesin / dashboard tidak pernah ada cat yang searah sinar
        VP, jadi dukungannya turun ke ~0; jalan terlihat sampai baris terakhir
        yang masih punya dukungan."""
        if self._row_sup is None or len(self._row_sup) != h:
            self._row_sup = np.full(h, np.nan, np.float32)
        ind = np.zeros(h, np.float32)
        ind[hit_rows] = 1.0
        t = tested_rows
        known = ~np.isnan(self._row_sup)
        known[t] = False
        self._row_sup[known] *= 0.98      # baris yang tidak teruji lagi meluruh
        cur = self._row_sup[t]
        self._row_sup[t] = np.where(np.isnan(cur), ind[t], cur + 0.06 * (ind[t] - cur))
        self._row_n += 1
        if self._row_n < 15:
            return
        known = np.nonzero(~np.isnan(self._row_sup))[0]
        if len(known) < 20:
            return
        sm = np.convolve(np.nan_to_num(self._row_sup, nan=0.0), np.ones(9) / 9.0, mode='same')
        ref = np.percentile(sm[known], 75)
        good = np.nonzero(sm >= 0.3 * ref)[0]
        if ref < 0.15 or len(good) == 0:
            return
        bot = int(min(h, good.max() + 3))
        if self.roi_bottom_y_frac is not None:
            bot = min(bot, int(h * self.roi_bottom_y_frac))
        self._road_bot = bot

    # ── fit garis ─────────────────────────────────────────────────────────
    def _fit_side(self, pts, xs, ys, rs, vx, vy, u, h, y_top, y_bot, lw):
        """Sampel x polyline (_N_SAMP baris y_top..y_bot) dari titik cat di
        sekitar sinar u. Pita pencarian mengikuti fit sebelumnya (2 iterasi)
        supaya tikungan landai ikut; kuadrat hanya kalau jelas lebih baik."""
        ty = np.linspace(y_top, y_bot, _N_SAMP)
        ray = self._ray_x(u, ty, vx, vy, h)
        x, y = pts if pts is not None else (np.zeros(0), np.zeros(0))
        if len(x) < 6 or np.ptp(y) < 8:
            return ray
        sel = (ys >= y_top) & (ys <= y_bot)
        ax, ay, ar = xs[sel], ys[sel], rs[sel]
        lw_y = np.maximum(1.0, lw * (ay - vy) / (h - vy))
        p = _lsq1(np.append(y, vy), np.append(x, vx), np.append(np.ones(len(x)), max(2.0, 0.05 * len(x))))
        deg = 1
        for _ in range(2):
            ok = np.abs(ax - np.polyval(p, ay)) < np.maximum(1.5, 0.05 * lw_y)
            if ok.sum() >= 6:
                ok &= ar >= 0.4 * np.median(ar[ok])
            if ok.sum() < 6 or np.ptp(ay[ok]) < 8:
                break
            yy, xx = ay[ok], ax[ok]
            p1 = _lsq1(np.append(yy, vy), np.append(xx, vx), np.append(np.ones(len(xx)), max(2.0, 0.05 * len(xx))))
            p, deg = p1, 1
            if len(xx) >= 20 and np.ptp(yy) > 0.35 * (y_bot - y_top):
                p2 = np.polyfit(yy, xx, 2)
                r1 = np.median(np.abs(np.polyval(p1, yy) - xx))
                r2 = np.median(np.abs(np.polyval(p2, yy) - xx))
                if r1 > 1.0 and r2 < 0.75 * r1:
                    p, deg = p2, 2
        if deg == 2:
            # jangan ekstrapolasi kuadrat jauh di luar data → sambung linear
            lin = np.polyfit(yy, xx, 1)
            inside = (ty >= yy.min() - 10) & (ty <= yy.max() + 10)
            return np.where(inside, np.polyval(p, ty), np.polyval(lin, ty))
        return np.polyval(p, ty)

    # ── utuh vs putus-putus ──────────────────────────────────────────────
    def _road_ref(self, uL, uR, rows, vx, vy):
        """Kecerahan aspal per baris + baris yang dalamnya memang jalan. Acuan
        warna = median sampel dalam ego lane di separuh atas pita (selalu
        jalan); baris yang warnanya menyimpang (kap mesin berwarna, kendaraan
        tepat di depan) tidak dipakai untuk uji cat / celah."""
        bgr, tex, gray = self._cur
        h, w = gray.shape
        u = uL + (uR - uL) * np.linspace(0.2, 0.8, 9)[None]
        px = np.clip(self._ray_x(u, rows[:, None], vx, vy, h).astype(int), 0, w - 1)
        py = np.repeat(rows[:, None], 9, axis=1)
        c = bgr[py, px].astype(np.int16)
        chroma = c.max(axis=2) - c.min(axis=2)
        ok = (chroma < _ROAD_CHROMA) & (tex[py, px] < _ROAD_TEX)
        g = gray[py, px].astype(np.float32)
        top = np.arange(len(rows)) < max(1, len(rows) // 2)
        okt = ok & top[:, None]
        g0 = float(np.median(g[okt])) if okt.sum() >= 6 else float(np.median(g))
        c0 = float(np.median(chroma[okt])) if okt.sum() >= 6 else 0.0
        ref = np.full(len(rows), g0, np.float32)
        for b in np.array_split(np.arange(len(rows)), 4):
            if len(b) and ok[b].sum() >= 6:
                ref[b] = np.median(g[b][ok[b]])
        gm, cm = np.median(g, axis=1), np.median(chroma, axis=1)
        road = (np.abs(cm - c0) < 15) & (gm > 0.6 * g0) & (gm < 1.5 * g0)
        return ref, road

    def _paint_gaps(self, u, xs, ys, rs, vx, vy, h, w, rows, lw, ref, road):
        """(cat, celah) berbobot sepanjang sinar marka u (VP per frame — sinar
        yang sama dengan puncak histogram di pita dekat). CAT = stripe kuat di garis;
        CELAH = aspal polos terlihat jelas (abu-abu, halus, kecerahan ≈ aspal
        di baris itu) tanpa stripe berarti di sekitarnya. Baris lain —
        kendaraan, kotak teks overlay, kap, di luar frame, cat yang sedikit
        meleset dari lintasan — tidak diketahui dan tidak dihitung, jadi garis
        UTUH yang tertutup sebagian tetap terbaca utuh."""
        bgr, tex, gray = self._cur
        r0 = int(rows[0])
        x = self._ray_x(u, rows, vx, vy, h)
        tol = np.maximum(1.5, 0.07 * lw * (rows - vy) / (h - vy))
        win = np.maximum(3.0, 2 * tol)
        inside = (x - win >= 0) & (x + win <= w - 1)
        sel = (ys >= r0) & (ys <= rows[-1])
        iy = ys[sel].astype(int) - r0
        d = np.abs(xs[sel] - x[iy])
        r = rs[sel]
        near = d < tol[iy]
        # kontras cat garis ini (persentil 90): coretan basah / tekstur aspal
        # jauh lebih lemah dari cat sehingga tidak ikut terhitung cat
        rc = float(np.percentile(r[near], 90)) if near.sum() >= 6 else float(_MARK_CONTRAST)
        paint = np.zeros(len(rows), bool)
        paint[iy[near & (r >= 0.4 * rc)]] = True
        busy = np.zeros(len(rows), bool)
        busy[iy[(d < win[iy]) & (r >= max(_RESP_MIN, 0.3 * rc))]] = True
        paint &= inside
        xi = np.clip(x.astype(int), 0, w - 1)
        c = bgr[rows, xi].astype(np.int16)
        plain = (np.abs(gray[rows, xi].astype(np.float32) - ref) < 0.35 * ref) & \
                (c.max(axis=1) - c.min(axis=1) < _ROAD_CHROMA) & (tex[rows, xi] < _ROAD_TEX)
        gap = inside & ~paint & ~busy & plain & road
        paint &= road
        # bobot baris ∝ lebar lajur di baris itu: baris dekat (resolusi tinggi,
        # lintasan akurat) lebih dipercaya dari baris jauh
        wt = (rows - vy) / (rows[-1] - vy)
        return float(wt[paint].sum()), float(wt[gap].sum())

    def _lane_pos(self, pts, u, uL, lw):
        """Posisi marka ternormalisasi (0 = garis kiri ego, 1 = kanan) diukur
        di baris tempat catnya terlihat, relatif polyline ego di baris yang
        sama. Marka jauh hanya tampak dekat VP; proyeksi ke baris bawah
        (u) membesar-besarkan galat VP / tikungan, rasio di baris sama tidak."""
        x, y = pts if pts is not None else (np.zeros(0), np.zeros(0))
        if len(x) >= 6 and self._samp[0] is not None and self._samp[1] is not None:
            xl = np.interp(y, self._samp_ty, self._samp[0])
            xr = np.interp(y, self._samp_ty, self._samp[1])
            ok = xr - xl > 4
            if ok.sum() >= 6:
                return float(np.median((x[ok] - xl[ok]) / (xr[ok] - xl[ok])))
        return (u - uL) / lw

    # ── marka lain → nomor jalur ──────────────────────────────────────────
    def _reset_votes(self):
        ks = range(-_MARK_MAX_K, _MARK_MAX_K + 2)
        self._mark_score = {k: 0.0 for k in ks if k not in (0, 1)}
        self._mark_u = {k: None for k in ks}
        self._mark_gap = {k: None for k in ks}     # EMA fraksi celah (aspal polos tanpa cat)
        self._mark_dash = {k: None for k in ks}    # True putus-putus / False utuh (histeresis)

    def _shift_votes(self, d):
        """Indeks marka relatif ego lane bergeser d (+1 = pindah ke kiri satu lajur)."""
        sc, mu, mg, md = self._mark_score, self._mark_u, self._mark_gap, self._mark_dash
        self._reset_votes()
        for k, v in mu.items():
            if k + d in self._mark_u:
                self._mark_u[k + d] = None if v is None else v + d
                self._mark_gap[k + d] = mg[k]
                self._mark_dash[k + d] = md[k]
                if k + d in self._mark_score:
                    # bekas garis ego: baru saja terlihat jelas → skor setengah steady state
                    self._mark_score[k + d] = sc.get(k, 0.5 / (1 - _MARK_DECAY))

    def _upd_gap(self, k, pg):
        """Update EMA fraksi celah marka k dari (cat, celah) recompute ini;
        status utuh/putus-putus pakai histeresis supaya tidak berkedip."""
        n_p, n_g = pg
        if n_p + n_g < _GAP_MIN_ROWS:
            return
        g = n_g / float(n_p + n_g)
        pv = self._mark_gap[k]
        g = g if pv is None else pv + _GAP_ALPHA * (g - pv)
        self._mark_gap[k] = g
        if g >= _GAP_DASH:
            self._mark_dash[k] = True
        elif g <= _GAP_SOLID:
            self._mark_dash[k] = False
        elif self._mark_dash[k] is None:
            self._mark_dash[k] = g > 0.5 * (_GAP_SOLID + _GAP_DASH)

    def _count(self, peaks, uL, uR, pgL, pgR, pg_at):
        """`peaks` = [(posisi ternormalisasi, dukungan, (cat, celah))] marka;
        pgL / pgR = (cat, celah) garis kiri / kanan ego (None = tak terlihat);
        pg_at(u) = (cat, celah) di posisi u mana pun (cek keterlihatan)."""
        lw = uR - uL
        hits = {}
        for k, sup, pg in peaks:
            kr = int(np.round(k))
            if kr in self._mark_score and abs(k - kr) < 0.25:
                if kr not in hits or sup > hits[kr][1]:
                    hits[kr] = (k, sup, pg)
        for k in self._mark_score:
            if k not in hits and self._mark_u[k] is not None and self._mark_score[k] > 0.5:
                # marka yang tak terdeteksi hanya meluruh kalau posisinya memang
                # terlihat (aspal / cat tampak); tertutup kendaraan, keluar
                # frame, di balik kotak overlay = tidak ada bukti → ditahan
                if sum(pg_at(uL + self._mark_u[k] * lw)) < _GAP_MIN_ROWS:
                    continue
            self._mark_score[k] = self._mark_score[k] * _MARK_DECAY + (1.0 if k in hits else 0.0)
            if k in hits:
                un, _, pg = hits[k]
                pu = self._mark_u[k]
                self._mark_u[k] = un if pu is None else pu + 0.3 * (un - pu)
                self._upd_gap(k, pg)
        for k, pg in ((0, pgL), (1, pgR)):
            if pg is not None:
                self._upd_gap(k, pg)

        def solid(k):
            """True = berhenti di sini (garis utuh / tepi jalan). Belum terukur
            atau hampir tanpa cat juga dianggap berhenti."""
            return self._mark_dash[k] is not True or self._mark_gap[k] >= _GAP_NOPAINT

        def chain(ego_k, ks):
            """Dari garis active lane ke luar: tiap garis PUTUS-PUTUS = ada satu
            jalur lagi di baliknya; berhenti di garis UTUH (tepi jalan) atau
            kalau marka berikutnya tidak terlihat."""
            n, used, scores = 0, [], []
            for k in (ego_k, *ks):
                if k != ego_k:
                    if self._mark_score[k] < _MARK_ON or self._mark_u[k] is None:
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

    # ── utama ─────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> dict:
        """Recompute tiap `skip` pemanggilan; di antaranya pakai cache.

        Return dict (atau None kalau belum ada garis sama sekali):
            left_line / right_line : np.array([x_bottom, y_bottom, x_top, y_top])
                                     di resolusi frame asli, atau None
            left_poly / right_poly : polyline [[x, y], ...] mengikuti tikungan, atau None
            active_lane            : True kalau marka kiri + kanan ego lane terlihat
            lane_number            : jalur ke-n dari kiri (posisi sekarang)
            total_lanes            : jumlah jalur dari marka (atau override)
            confidence             : kekuatan voting marka (0..1)
            markings               : garis marka jalur lain [x_b, y_b, x_t, y_t]
            left_solid/right_solid : garis active lane utuh (tepi jalan) atau putus-putus
            vanishing_point        : (x, y) titik hilang di resolusi frame asli, atau None
            roi_polygon            : area jalan yang dipakai di resolusi frame asli
        """
        self._n += 1
        if self.result is not None and (self._n - 1) % self.skip:
            return self.result

        H0, W0 = frame.shape[:2]
        scale = self.work_h / H0
        w, h = max(2, int(W0 * scale)), self.work_h
        if self._wh != (w, h):
            self._wh = (w, h)
            self._reset_state()
            self.result = None
        small = frame if (W0, H0) == (w, h) else cv2.resize(frame, (w, h), interpolation=self.interp)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        b, g, r = cv2.split(small)
        ych = cv2.subtract(cv2.addWeighted(r, 0.5, g, 0.5, 0), b)   # (R+G)/2 - B ≥ 0
        tex = cv2.blur(cv2.add(cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0)),
                               cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1))), (5, 5))
        self._cur = (small, tex, gray)

        road_bot = self._road_bot or (int(h * self.roi_bottom_y_frac) if self.roi_bottom_y_frac else h)
        trusted = self._vp is not None and self._vp_ok >= _VP_TRUST
        if trusted:
            self._vp_n += 1
        # VP belum dipercaya / jadwal uji ulang → cari luas (multi-skala, bisa pindah jauh)
        search = not trusted or self._vp_n % _VP_CHALLENGE == 0
        y0 = max(0.04 * h, self._vp[1] - 0.12 * h) if not search else _VP_SEARCH_TOP * h
        if trusted and search:
            y0 = min(y0, max(0.04 * h, self._vp[1] - 0.12 * h))
        if self.roi_top_y_frac:
            y0 = max(y0, h * self.roi_top_y_frac - 0.12 * h)
        y0 = int(y0)
        if road_bot - y0 < 16:
            return self._lost(scale, road_bot)
        vy_s = None if search else self._vp[1]
        lw_px = None if search else self._lane_w(h, vy_s)
        xs, ys, rs, vmask = self._stripes(gray, ych, small, y0, h, vy_s, lw_px)

        # Hough + voting VP: wajib saat search; saat VP dipercaya cukup tiap 4 recompute
        # (vx per frame untuk tikungan sudah dari ketajaman histogram)
        segs = None
        if search or self._vp_n % 4 == 0:
            segs = self._segments(vmask[:max(0, road_bot - y0)], y0)
            self._vp_accumulate(segs, w, h)
        hyp = None
        if trusted:
            vx, vy = self._vp
            m = self._vp_refine(segs, vx, vy, 0.06 * w, 0.04 * h)
            if m is not None:
                vx, vy = vx + 0.25 * (m[0] - vx), vy + 0.08 * (m[1] - vy)
                self._vp = (vx, vy)
            vxc = self._vxc if self._vxc is not None and abs(self._vxc - vx) < 0.25 * w else vx
            hyp = self._lane_hyp(xs, ys, rs, vx, vy, h, w, road_bot, vx_curve=vxc)
        if search:
            # uji beberapa kandidat VP dengan bukti marka; ambil yang terbaik
            best, cur_h = None, None
            if self._vp is not None and not trusted:
                cur_h = self._lane_hyp(xs, ys, rs, self._vp[0], self._vp[1], h, w, road_bot)
            for cx, cy in self._vp_peaks(4):
                m = self._vp_refine(segs, cx, cy, 10, 10)
                vx, vy = m if m is not None else (cx, cy)
                hh = self._lane_hyp(xs, ys, rs, vx, max(vy, 0.0), h, w, road_bot)
                if hh is not None and (best is None or hh['score'] > best['score']):
                    best = hh
            if cur_h is not None and (cur_h['strong'] or cur_h['ok']) and \
                    (best is None or cur_h['score'] >= 0.85 * best['score']):
                best = cur_h   # histeresis: VP sekarang tetap kecuali kalah jelas
            if trusted:
                # penantang harus jauh lebih kuat 2x berturut-turut untuk menggantikan VP
                far = best is not None and (abs(best['vx'] - self._vp[0]) > 0.04 * w or
                                            abs(best['vy'] - self._vp[1]) > 0.03 * h)
                cur = hyp['score'] if hyp is not None else 0.0
                if far and best['strong'] and best['score'] > 1.3 * cur + 5:
                    self._vp_chal += 1
                else:
                    self._vp_chal = 0
                if self._vp_chal >= 2:
                    self._vp_chal, self._vp_ok, self._vp_n = 0, 0, 0
                    self._reset_lanes()
                    self._row_sup, self._row_n, self._road_bot = None, 0, None   # framing kamera berubah
                    self._vp = (best['vx'], best['vy'])
                    hyp = best
            elif best is not None and best['score'] > 0 and (best['strong'] or best['ok']):
                # VP belum dipercaya: hanya hipotesis berbukti pasangan marka yang
                # diadopsi — jalan tanpa marka tidak menghasilkan garis palsu
                new = (best['vx'], best['vy'])
                stable = self._vp is not None and abs(new[0] - self._vp[0]) <= 0.04 * w and \
                    abs(new[1] - self._vp[1]) <= 0.03 * h
                if not stable:
                    self._vp_ok = 0
                    # lompatan VP besar → geometri u lama tidak berlaku; lompatan kecil
                    # cukup diikuti tracker (reset di sini bikin garis berkedip)
                    if self._vp is None or abs(new[1] - self._vp[1]) > 0.08 * h or \
                            abs(new[0] - self._vp[0]) > 0.15 * w:
                        self._reset_lanes()
                self._vp = new
                hyp = best
        if hyp is None:
            self._vp_ok = max(0, self._vp_ok - 1)
            return self._lost(scale, road_bot)
        f0, f1 = hyp['found']
        near_clear = f0 is not None and f1 is not None and min(f0[1], f1[1]) >= _SUP_MIN
        self._far_bad = self._far_bad + 1 if near_clear and not hyp['far_ok'] else 0
        if self._far_bad >= _FAR_BAD_MAX:
            # garis dekat jelas tapi sinar terus tak cocok di bagian jauh → VP basi
            # (mis. kamera digeser); sesekali tertutup kendaraan tidak apa
            self._vp_ok = max(0, self._vp_ok - 1)
            if self._vp_ok < _VP_TRUST:
                self._vp_n = 0
        elif hyp['strong'] or hyp['ok']:
            self._vp_ok = min(10, self._vp_ok + 1)
        elif f0 is None or f1 is None:
            # turun hanya kalau pasangan hilang; pasangan lemah (marka pudar) tetap
            self._vp_ok = max(0, self._vp_ok - 1)
            if self._vp_ok < _VP_TRUST:
                self._vp_n = 0

        vx, vy, y_top, found = hyp['vx'], hyp['vy'], hyp['y_top'], hyp['found']
        self._vxc = vx
        lw = self._lane_w(h, vy)
        if found[0] is not None and found[1] is not None and None not in self._ego:
            # pindah lajur: marka kiri baru ≈ kanan lama (ke kanan satu lajur) atau
            # kanan baru ≈ kiri lama (ke kiri) → indeks voting marka digeser
            oL, oR = self._ego
            if abs(found[0][0] - oR) < 0.25 * lw:
                self._shift_votes(-1)
            elif abs(found[1][0] - oL) < 0.25 * lw:
                self._shift_votes(+1)
            elif abs(found[0][0] - oL) > 0.25 * lw and abs(found[1][0] - oR) > 0.25 * lw:
                self._reset_votes()
        for s in (0, 1):
            f = found[s]
            if f is not None:
                u = f[0]
                prev = self._ego[s]
                jump = prev is not None and abs(u - prev) > 0.25 * lw
                if prev is None or jump or self._miss[s] > 2:
                    self._ego[s] = u
                else:   # adaptif: diam → halus, bergerak → langsung ikut (tanpa lag)
                    a = min(1.0, self.alpha + abs(u - prev) / (_SMOOTH_U * lw))
                    self._ego[s] = prev + a * (u - prev)
                if jump:
                    self._samp[s] = None
                self._miss[s] = 0
            else:
                self._miss[s] += 1
                o = found[1 - s]
                if o is not None and self._lwk is not None and self._ego[s] is not None:
                    # sisi lain terlihat → posisi sisi ini mengikuti lebar lajur
                    # (marka pudar / tertutup kendaraan sesaat), ditahan lebih lama
                    self._ego[s] = o[0] + (lw if s else -lw)
                    keep = _MAX_MISS_PRED
                else:
                    keep = _MAX_MISS
                if self._miss[s] > keep:
                    self._ego[s] = None
                    self._samp[s] = None
        uL, uR = self._ego
        if found[0] is not None and found[1] is not None:
            wd = found[1][0] - found[0][0]
            k = float(np.clip(wd / (h - vy), *_LW_K_RANGE))
            self._lwk = k if self._lwk is None else self._lwk + 0.1 * (k - self._lwk)
            if hyp['strong']:   # pusat kendaraan (kamera bisa sedikit off-center)
                cen = 0.5 * (found[0][0] + found[1][0])
                xc = self._xc if self._xc is not None else w / 2.0
                self._xc = float(np.clip(xc + 0.005 * (cen - xc), 0.42 * w, 0.58 * w))
            if min(found[0][1], found[1][1]) >= _SUP_MIN and min(found[0][5], found[1][5]) >= _STRONG_CON:
                hits, tested = [], []
                for f in found:
                    prof = self._support(f[0], xs, ys, rs, vx, vy, h, w, y_top, road_bot - 1,
                                         self._lane_w(h, vy), profile=True)[3]
                    if prof is not None:
                        hits.append(prof[0])
                        tested.append(prof[1])
                if len(hits) == 2:
                    self._update_road_bottom(np.union1d(*hits).astype(int),
                                             np.union1d(*tested).astype(int), h)

        y_bot = road_bot - 1
        lw_fit = self._lane_w(h, vy)
        for s, u in enumerate((uL, uR)):
            if u is None:
                continue
            pts = found[s][3] if found[s] is not None else \
                self._support(u, xs, ys, rs, vx, vy, h, w, y_top, y_bot, lw_fit)[2]
            xsamp = self._fit_side(pts, xs, ys, rs, vx, vy, u, h, y_top, y_bot, lw_fit)
            prev = self._samp[s]
            ty = np.linspace(y_top, y_bot, _N_SAMP)
            if prev is not None and self._samp_ty is not None and \
                    np.abs(self._samp_ty - ty).max() > 0.5:
                # grid baris berubah (batas jalan / VP) → sampel lama dipindah ke grid baru
                pt = self._samp_ty
                prev = np.interp(ty, pt, prev, left=np.nan, right=np.nan)
                out = np.isnan(prev)
                if out.any():
                    p1 = np.polyfit(pt, self._samp[s], 1)
                    prev[out] = np.polyval(p1, ty[out])
            if prev is None:
                self._samp[s] = xsamp
            else:   # adaptif: getaran kecil diredam, pergeseran nyata diikuti cepat
                a = min(1.0, self.alpha + float(np.abs(xsamp - prev).mean()) / _SMOOTH_PX)
                self._samp[s] = prev + a * (xsamp - prev)
        self._samp_ty = np.linspace(y_top, y_bot, _N_SAMP)

        counted = None
        y_gap = min(int(y_bot), h - 1)
        if self._road_bot is None and not self.roi_bottom_y_frac:
            # kap mesin / dashboard belum dipelajari: baris uji celah hanya sampai
            # cat garis ego terbawah frame ini (di bawah kap tidak ada cat searah sinar)
            low = [int(f[3][1].max()) for f in found if f is not None and len(f[3][1])]
            if low:
                y_gap = min(y_gap, max(low))
        rows = np.arange(int(np.ceil(vy + _GAP_NEAR * (y_gap - vy))), y_gap + 1)
        if uL is not None and uR is not None and uR - uL > 0.1 * w and len(rows) >= 8:
            lwc = uR - uL
            ref, road = self._road_ref(uL, uR, rows, vx, vy)

            def pg(u):
                return self._paint_gaps(u, xs, ys, rs, vx, vy, h, w, rows, lwc, ref, road)
            ps = []
            for u in hyp['peaks']:
                frac, _, pts = self._support(u, xs, ys, rs, vx, vy, h, w, y_top, y_bot, lwc)
                if frac is not None and frac >= _SUP_MIN:
                    ps.append((self._lane_pos(pts, u, uL, lwc), frac, pg(u)))
            # garis ego: posisi puncak frame ini (sinar yang sama dengan cat-nya)
            counted = self._count(ps, uL, uR, pg(found[0][0]) if found[0] is not None else None,
                                  pg(found[1][0]) if found[1] is not None else None, pg)
        return self._emit((vx, vy, y_top, y_bot), scale, road_bot, counted)

    def _lost(self, scale, road_bot):
        """Recompute tanpa hipotesis: hitung miss tiap sisi; setelah _MAX_MISS
        garis dihapus (active NO) — tidak menahan garis basi selamanya."""
        for s in (0, 1):
            self._miss[s] += 1
            if self._miss[s] > _MAX_MISS:
                self._ego[s] = None
                self._samp[s] = None
        return self._emit(self._geo, scale, road_bot)

    def _emit(self, geo, scale, road_bot, counted=None):
        if geo is None:
            return self.result
        self._geo = geo
        vx, vy, y_top, y_bot = geo
        inv = 1.0 / scale
        ty = np.linspace(y_top, y_bot, _N_SAMP)
        polys, lines = [None, None], [None, None]
        for s in (0, 1):
            xsamp = self._samp[s] if self._ego[s] is not None else None
            if xsamp is None:
                continue
            polys[s] = np.round(np.stack([xsamp, ty], 1) * inv).astype(int)
            # garis lurus (kompatibel): fit berbobot ke bagian dekat
            p = np.polyfit(ty, xsamp, 1, w=np.linspace(0.3, 1.0, _N_SAMP))
            lines[s] = np.round(np.array([np.polyval(p, y_bot), y_bot,
                                          np.polyval(p, y_top), y_top]) * inv).astype(int)
        if polys[0] is not None and polys[1] is not None:
            # di tikungan/ramp ujung jauh kedua polyline bisa bersilangan → buang
            # sampel di atas titik silang; menyilang di bagian dekat = tidak valid
            ok = polys[0][:, 0] < polys[1][:, 0]
            if not ok.all() and ok[_N_SAMP // 3:].all():
                k = int(np.nonzero(~ok)[0].max()) + 1
                polys = [polys[0][k:], polys[1][k:]]
        left_line, right_line = lines
        if left_line is None and right_line is None:
            if self.result is not None:
                self.result = {**self.result, 'left_line': None, 'right_line': None,
                               'left_poly': None, 'right_poly': None,
                               'active_lane': False, 'markings': []}
            return self.result
        active = left_line is not None and right_line is not None and \
            bool(np.all(polys[0][:, 0] < polys[1][:, 0]))
        prev = self.result or {}
        lane_no = prev.get('lane_number', 1)
        total = prev.get('total_lanes', self.total_lanes or 1)
        conf = prev.get('confidence', 0.0)
        markings = []          # marka lain hanya digambar saat active (tidak basi)
        bounds = [0.0, 1.0] if active else None
        if not active:
            conf *= 0.9        # nomor jalur terakhir tetap, kepercayaannya meluruh
        ego_style = {k: prev.get(k) for k in ('left_solid', 'right_solid')}
        if counted is not None and active:
            n_left, n_right, used, conf = counted
            lane_no = n_left + 1
            total = lane_no + n_right
            if self.total_lanes:
                total = self.total_lanes
                lane_no = min(lane_no, total)
            uL, uR = self._ego
            h = self.work_h
            markings = []
            bounds = sorted([0.0, 1.0] + [float(un) for un, _ in used])
            for un, _ in used:
                u = uL + un * (uR - uL)
                xb, xt = self._ray_x(u, y_bot, vx, vy, h), self._ray_x(u, y_top, vx, vy, h)
                markings.append(np.round(np.array([xb, y_bot, xt, y_top]) * inv).astype(int))
            ego_style = {'left_solid': n_left == 0, 'right_solid': n_right == 0}
        W = int(round(self._wh[0] * inv))
        yt, yb = int(round(y_top * inv)), int(round(y_bot * inv))
        self.result = {
            'lane_number': lane_no, 'confidence': conf, 'total_lanes': total,
            'left_line': left_line, 'right_line': right_line,
            'left_poly': polys[0], 'right_poly': polys[1],
            'active_lane': active,
            'markings': markings,
            'lane_bounds': bounds,
            **ego_style,
            'vanishing_point': (int(round(vx * inv)), int(round(vy * inv))),
            'roi_polygon': np.array([[0, yb], [0, yt], [W - 1, yt], [W - 1, yb]], np.int32),
        }
        return self.result


def _x_at(result, side, y):
    """x garis `side` di baris y: polyline (ikut tikungan) kalau ada, kalau
    tidak garis lurus. None kalau garis tidak ada."""
    poly = result.get(f'{side}_poly')
    if poly is not None and len(poly) >= 2:
        p = np.asarray(poly, np.float64)
        if p[0, 1] <= y <= p[-1, 1]:
            return float(np.interp(y, p[:, 1], p[:, 0]))
    line = result.get(f'{side}_line')
    if line is None:
        return None
    xb, yb, xt, yt = map(float, line)
    if yb == yt:
        return None
    return xb + (y - yb) / (yt - yb) * (xt - xb)


def lane_of_point(result: dict, x: float, y: float):
    """Jalur ke-n (1..total_lanes, dari kiri) tempat titik (x, y) berada,
    atau None kalau active lane belum ada / titik di luar jalur yang terhitung
    (bahu jalan di balik garis tepi, lajur yang markanya tidak terlihat).

    Koordinat ternormalisasi: u = (x - x_kiri(y)) / (x_kanan(y) - x_kiri(y))
    di baris y (0 = garis kiri ego lane, 1 = garis kanan). Batas jalur =
    posisi marka hasil ukur (`lane_bounds`, lebar tiap lajur boleh beda);
    tanpa itu dianggap selebar ego lane → jalur = lane_number + floor(u).
    Untuk pothole/crack, pakai titik tengah-bawah bbox (titik kontak dengan jalan).
    """
    if not result or not result.get('active_lane'):
        return None
    xl, xr = _x_at(result, 'left', y), _x_at(result, 'right', y)
    if xl is None or xr is None or xr - xl <= 1:
        return None  # di atas / sekitar titik hilang
    u = (x - xl) / (xr - xl)
    ego = result.get('lane_number', 1)
    total = result.get('total_lanes', 1)
    bounds = result.get('lane_bounds')
    if bounds and len(bounds) >= 2:
        # interval ke-i di antara batas; di luar batas terluar = satu jalur lagi
        # (valid hanya kalau masih dalam total_lanes, mis. di balik marka putus-putus)
        lane = ego + int(np.searchsorted(bounds, u, side='right')) - 1 - bounds.index(0.0)
    else:
        lane = ego + int(np.floor(u))
    return lane if 1 <= lane <= total else None


def draw_lane(frame: np.ndarray, result: dict, draw_roi: bool = False,
              draw_lines: bool = True, draw_fill: bool = False) -> np.ndarray:
    """Gambar garis lane (polyline, ikut tikungan) + marka jalur lain
    (`draw_lines`, default on) dan nomor jalur di tengah ACTIVE LANE secara
    in-place. Shading hijau active lane (`draw_fill`) default off."""
    if not result:
        return frame
    polys = [result.get('left_poly'), result.get('right_poly')]
    lines = [result.get('left_line'), result.get('right_line')]
    if all(l is None for l in lines):
        return frame

    def pts(s):
        if polys[s] is not None:
            return np.asarray(polys[s], np.int32)
        x1, y1, x2, y2 = lines[s]
        return np.array([[x2, y2], [x1, y1]], np.int32)

    # Garis lane: kuning tebal, blend 40%; marka jalur lain: putih tipis
    if draw_lines:
        overlay = frame.copy()
        for x1, y1, x2, y2 in result.get('markings', []):
            cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
        for s in (0, 1):
            if lines[s] is not None:
                cv2.polylines(overlay, [pts(s)], False, (0, 255, 255), 8)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, dst=frame)

    # Active lane — hanya kalau kiri + kanan ada
    if result.get('active_lane'):
        pl, pr = pts(0), pts(1)
        if draw_fill:  # drivable area: hijau, blend 15%
            poly = np.vstack([pl, pr[::-1]])
            drivable = frame.copy()
            cv2.fillPoly(drivable, [poly], (0, 200, 0))
            cv2.addWeighted(drivable, 0.15, frame, 0.85, 0, dst=frame)

        # Nomor jalur di tengah active lane, di bagian bawah lajur
        yb, yt = int(pl[-1, 1]), int(pl[0, 1])
        cy = int(yb - 0.25 * (yb - yt))
        xl, xr = _x_at(result, 'left', cy), _x_at(result, 'right', cy)
        cx = int(0.5 * (xl + xr)) if xl is not None and xr is not None else int(0.5 * (pl[-1, 0] + pr[-1, 0]))
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

"""
OLED Display Manager untuk Dashcam Jetson Orin Nano
Menampilkan 2 screen bergantian tiap SCREEN_INTERVAL detik:
  - Screen 1: Status kamera (LIVE/OFF) & GPS (LOCKED/NO LOCK)
  - Screen 2: Jumlah deteksi Potholes & Crack
Hardware: SSD1306 128x64 via I2C (auto-detect bus)
"""

import threading
import time
import glob
from datetime import datetime

# -------------------------------------------------------
# Optional import — app tetap jalan meski OLED tidak ada
# -------------------------------------------------------
try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    from luma.core.render import canvas
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False

try:
    import smbus2
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False


I2C_ADDRESS     = 0x3C   # Ganti 0x3D jika perlu
WIDTH           = 128
HEIGHT          = 64
SCREEN_INTERVAL = 5      # detik tiap screen berganti


def _auto_detect_bus(address: int = I2C_ADDRESS) -> int | None:
    """Scan semua /dev/i2c-* dan return bus yang berisi OLED."""
    if not SMBUS_AVAILABLE:
        return None
    buses = sorted([
        int(p.replace("/dev/i2c-", ""))
        for p in glob.glob("/dev/i2c-*")
    ])
    for bus in buses:
        try:
            with smbus2.SMBus(bus) as b:
                b.read_byte(address)
            return bus
        except Exception:
            continue
    return None


class OLEDManager:
    """
    Manager OLED display dengan 2 screen bergantian.

    Screen 1 — Status:
        CAM : LIVE / OFF
        GPS : LOCKED / NO LOCK
        Jam real-time

    Screen 2 — Deteksi:
        Potholes : <jumlah>
        Crack    : <jumlah>
        Total    : <jumlah>

    Graceful fallback jika OLED tidak terhubung (app tidak crash).
    """

    def __init__(self, i2c_port: int | None = None, address: int = I2C_ADDRESS):
        self._port    = i2c_port   # None = auto-detect
        self._address = address
        self._device  = None
        self._thread  = None
        self._running = False
        self._lock    = threading.Lock()

        # State screen 1
        self._camera_running = False
        self._gps_locked     = False
        self._gps_position   = {'latitude': 0.0, 'longitude': 0.0}

        # State screen 2 — dict class_name -> count
        self._detection_counts: dict = {}

        # Kontrol pergantian screen
        self._current_screen  = 1
        self._last_switch     = time.time()

        # Scroll offset untuk teks GPS yang panjang
        self._scroll_offset = 0.0
        self._scroll_dir    = -1    # -1 = geser kiri, +1 = geser kanan

        # Cache jam agar tidak format string tiap frame
        self._cached_time   = datetime.now().strftime("%H:%M:%S")
        self._last_time_upd = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Cari OLED & mulai thread update. Return True jika berhasil."""
        if not OLED_AVAILABLE:
            print("⚠️  [OLED] luma.oled tidak terinstall — display dinonaktifkan")
            return False

        port = self._port if self._port is not None else _auto_detect_bus(self._address)
        if port is None:
            print("⚠️  [OLED] Tidak ditemukan di bus manapun — display dinonaktifkan")
            return False

        try:
            serial        = i2c(port=port, address=self._address)
            self._device  = ssd1306(serial, width=WIDTH, height=HEIGHT)
            self._running = True
            self._thread  = threading.Thread(
                target=self._loop, name="oled-display", daemon=True
            )
            self._thread.start()
            print(f"✅  [OLED] Display aktif di /dev/i2c-{port} addr=0x{self._address:02X}")
            return True
        except Exception as e:
            print(f"⚠️  [OLED] Gagal inisialisasi: {e} — display dinonaktifkan")
            return False

    def update(self, camera_running: bool, gps_locked: bool,
               detection_counts: dict | None = None,
               gps_position: dict | None = None):
        """Update semua state yang ditampilkan (thread-safe)."""
        with self._lock:
            self._camera_running   = camera_running
            self._gps_locked       = gps_locked
            if detection_counts is not None:
                self._detection_counts = detection_counts
            if gps_position is not None:
                self._gps_position = gps_position

    def stop(self):
        """Hentikan thread dan bersihkan layar."""
        self._running = False
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=2)
            except (KeyboardInterrupt, Exception):
                pass
        if self._device:
            try:
                self._show_off()
            except Exception:
                pass
            try:
                self._device.clear()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self):
        """Thread loop 20fps — scroll halus, ganti screen tiap SCREEN_INTERVAL detik."""
        SLEEP = 0.05   # 20 fps
        while self._running:
            try:
                now_t = time.time()

                # Update cache jam tiap detik
                if now_t - self._last_time_upd >= 1.0:
                    self._cached_time   = datetime.now().strftime("%H:%M:%S")
                    self._last_time_upd = now_t

                # Ganti screen tiap SCREEN_INTERVAL
                if now_t - self._last_switch >= SCREEN_INTERVAL:
                    self._current_screen = 2 if self._current_screen == 1 else 1
                    self._last_switch = now_t
                    self._scroll_offset = 0.0   # reset scroll saat pindah screen
                    self._scroll_dir = -1

                with self._lock:
                    cam    = self._camera_running
                    gps    = self._gps_locked
                    pos    = dict(self._gps_position)
                    counts = dict(self._detection_counts)
                    screen = self._current_screen

                if screen == 1:
                    self._render_screen1(cam, gps, pos)
                else:
                    self._render_screen2(counts)

            except Exception:
                pass   # Jangan crash meski ada error display
            time.sleep(SLEEP)

    def _render_screen1(self, camera_running: bool, gps_locked: bool,
                        gps_position: dict | None = None):
        """Screen 1 — Status kamera & GPS (dengan koordinat scrolling mulus)."""
        if self._device is None:
            return

        cam_icon  = "*" if camera_running else "-"
        cam_label = "LIVE " if camera_running else "OFF  "

        with canvas(self._device) as draw:
            # Header
            draw.rectangle((0, 0, WIDTH - 1, 13), fill="white")
            draw.text((18, 2), "DASHCAM STATUS", fill="black")

            # Baris kamera
            draw.text((4, 17), f"CAM [{cam_icon}] {cam_label}", fill="white")

            # Garis pemisah
            draw.line((4, 30, WIDTH - 5, 30), fill="white")

            # Baris GPS
            if gps_locked and gps_position:
                lat = gps_position.get('latitude', 0.0)
                lon = gps_position.get('longitude', 0.0)
                gps_text = f"Lat:{lat:.5f} Lon:{lon:.5f}"

                # Lebar teks — 6px per karakter (font default PIL)
                text_w    = len(gps_text) * 6
                max_off   = max(0, text_w - (WIDTH - 8))

                # Bounce scroll 1px/frame @ 20fps = ~20px/detik, mulus
                if max_off > 0:
                    self._scroll_offset += self._scroll_dir * 1
                    if self._scroll_offset <= -max_off:
                        self._scroll_offset = -max_off
                        self._scroll_dir = 1     # balik kanan
                    elif self._scroll_offset >= 0:
                        self._scroll_offset = 0
                        self._scroll_dir = -1    # balik kiri
                else:
                    self._scroll_offset = 0

                draw.text((4 + int(self._scroll_offset), 34), gps_text, fill="white")
            else:
                gps_icon  = "!" if not gps_locked else "+"
                gps_label = "NO LOCK" if not gps_locked else "LOCKED"
                draw.text((4, 34), f"GPS [{gps_icon}] {gps_label}", fill="white")

            # Footer jam (dari cache, update tiap detik)
            draw.line((0, 51, WIDTH - 1, 51), fill="white")
            draw.text((34, 54), self._cached_time, fill="white")

            # Indikator screen
            draw.text((116, 54), "1/2", fill="white")

    def _render_screen2(self, counts: dict):
        """Screen 2 — Jumlah deteksi Potholes & Crack."""
        if self._device is None:
            return

        # Ambil nilai dari counts, nama class case-insensitive
        def get(key):
            for k, v in counts.items():
                if key.lower() in k.lower():
                    return v
            return 0

        potholes = get("pothole")
        crack    = get("crack")
        total    = sum(counts.values()) if counts else 0

        with canvas(self._device) as draw:
            # Header
            draw.rectangle((0, 0, WIDTH - 1, 13), fill="white")
            draw.text((16, 2), "ROAD DETECTION", fill="black")

            # Potholes
            draw.text((4, 17), "Potholes :", fill="white")
            draw.text((84, 17), str(potholes), fill="white")

            # Garis
            draw.line((4, 30, WIDTH - 5, 30), fill="white")

            # Crack
            draw.text((4, 34), "Crack    :", fill="white")
            draw.text((84, 34), str(crack), fill="white")

            # Total
            draw.line((0, 48, WIDTH - 1, 48), fill="white")
            draw.text((4, 51), f"Total : {total}", fill="white")

            # Indikator screen
            draw.text((100, 51), "2/2", fill="white")

    def _show_off(self):
        """Tampilkan layar 'OFF' sebelum shutdown."""
        if self._device is None:
            return
        with canvas(self._device) as draw:
            draw.rectangle((0, 0, WIDTH - 1, HEIGHT - 1), outline="white")
            draw.text((30, 20), "DASHCAM", fill="white")
            draw.text((40, 36), "OFF", fill="white")

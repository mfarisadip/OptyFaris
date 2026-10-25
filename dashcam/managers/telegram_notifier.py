"""
Telegram Notifier untuk Dashcam Jetson Orin Nano
Konfigurasi: edit file telegram.json di root project

Mode:
  - Live dashboard : edit 1 pesan tiap `interval_status` detik
  - Alert deteksi  : kirim pesan baru saat ada pothole/crack baru
                     (cooldown `cooldown_alert` detik)
  - App stop       : kirim pesan "Dashcam OFF" saat app berhenti

Tidak perlu install library tambahan — cukup pakai `requests`.
"""

import json
import os
import threading
import time
from datetime import datetime


# -------------------------------------------------------
# Optional import
# -------------------------------------------------------
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

CONFIG_FILE = "telegram.json"


def _load_config() -> dict | None:
    """Baca telegram.json. Return None jika tidak ada atau tidak valid."""
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        token   = cfg.get("bot_token", "")
        chat_id = cfg.get("chat_id", "")
        if not token or not chat_id:
            return None
        if "GANTI" in token or "GANTI" in str(chat_id):
            return None   # masih template, belum diisi
        return cfg
    except Exception:
        return None


class TelegramNotifier:
    """
    Telegram live dashboard & alert untuk dashcam.

    Cara pakai:
        tg = TelegramNotifier()
        tg.start()
        tg.update(camera_running, gps_locked, gps_pos, detection_counts)
        tg.stop()
    """

    def __init__(self):
        self._cfg     = _load_config()
        self._enabled = bool(self._cfg and REQUESTS_AVAILABLE)

        if not REQUESTS_AVAILABLE:
            print("⚠️  [TG] `requests` tidak terinstall — Telegram dinonaktifkan")
        elif self._cfg is None:
            print("⚠️  [TG] telegram.json tidak ada / belum diisi — Telegram dinonaktifkan")
        else:
            self._token    = self._cfg["bot_token"]
            self._chat_id  = str(self._cfg["chat_id"])
            self._interval = int(self._cfg.get("interval_status", 5))
            self._cooldown = int(self._cfg.get("cooldown_alert", 60))

        # State
        self._lock             = threading.Lock()
        self._running          = False
        self._thread           = None
        self._dashboard_msg_id = None   # ID pesan live dashboard
        self._last_alert_time  = 0.0
        self._last_counts      = {}

        # Data terbaru
        self._camera_running   = False
        self._gps_locked       = False
        self._gps_position     = {}
        self._detection_counts = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Mulai thread notifier. Return True jika aktif."""
        if not self._enabled:
            return False
        # Kirim pesan awal dan simpan message_id
        msg_id = self._send_new("⏳ *DASHCAM STARTING...*\nMenginisialisasi sistem...")
        self._dashboard_msg_id = msg_id

        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="tg-notifier", daemon=True
        )
        self._thread.start()
        print(f"✅  [TG] Telegram notifier aktif (interval: {self._interval}s)")
        return True

    def update(self, camera_running: bool, gps_locked: bool,
               gps_position: dict | None = None,
               detection_counts: dict | None = None):
        """Update state (thread-safe). Dipanggil dari main loop dashcam."""
        if not self._enabled:
            return
        with self._lock:
            self._camera_running = camera_running
            self._gps_locked     = gps_locked
            if gps_position:
                self._gps_position = gps_position
            if detection_counts is not None:
                # Cek apakah ada deteksi BARU sejak update terakhir
                new_detection = any(
                    detection_counts.get(k, 0) > self._last_counts.get(k, 0)
                    for k in detection_counts
                )
                self._detection_counts = detection_counts
                if new_detection:
                    self._trigger_alert()

    def stop(self):
        """Hentikan notifier dan kirim pesan 'OFF'."""
        self._running = False
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=3)
            except Exception:
                pass
        if self._enabled:
            now = datetime.now().strftime("%H:%M:%S")
            self._send_new(f"🔴 *DASHCAM OFF* — {now}\nSistem dihentikan.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self):
        """Thread: edit live dashboard tiap interval_status detik."""
        while self._running:
            try:
                with self._lock:
                    cam    = self._camera_running
                    gps    = self._gps_locked
                    pos    = dict(self._gps_position)
                    counts = dict(self._detection_counts)
                self._edit_dashboard(cam, gps, pos, counts)
            except Exception:
                pass
            time.sleep(self._interval)

    def _build_status_text(self, camera_running, gps_locked, gps_pos, counts) -> str:
        """Buat teks pesan live dashboard."""
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        cam_str = "🟢 LIVE" if camera_running else "🔴 OFF"

        if gps_locked and gps_pos:
            lat = gps_pos.get("latitude", 0.0)
            lon = gps_pos.get("longitude", 0.0)
            gps_str = f"✅ `{lat:.5f}, {lon:.5f}`"
        else:
            gps_str = "⏳ Mencari sinyal..."

        def get(key):
            for k, v in counts.items():
                if key.lower() in k.lower():
                    return v
            return 0

        pothole = get("pothole")
        crack   = get("crack")
        total   = sum(counts.values()) if counts else 0

        return (
            f"📊 *DASHCAM LIVE MONITOR*\n"
            f"🕐 `{now}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📷 Kamera  : {cam_str}\n"
            f"🛰️ GPS     : {gps_str}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🕳️ Potholes : *{pothole}*\n"
            f"🔺 Crack    : *{crack}*\n"
            f"📦 Total    : *{total}*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
          
        )

    def _edit_dashboard(self, cam, gps, pos, counts):
        """Edit pesan dashboard yang sudah ada."""
        if not self._dashboard_msg_id:
            return
        text = self._build_status_text(cam, gps, pos, counts)
        self._api("editMessageText", {
            "chat_id":    self._chat_id,
            "message_id": self._dashboard_msg_id,
            "text":       text,
            "parse_mode": "Markdown"
        })

    def _trigger_alert(self):
        """Kirim alert deteksi baru (dengan cooldown)."""
        now = time.time()
        if now - self._last_alert_time < self._cooldown:
            return   # masih dalam cooldown
        self._last_alert_time = now

        counts = self._detection_counts

        def get(key):
            for k, v in counts.items():
                if key.lower() in k.lower():
                    return v
            return 0

        pothole = get("pothole")
        crack   = get("crack")
        total   = sum(counts.values()) if counts else 0
        ts      = datetime.now().strftime("%H:%M:%S")

        msg = (
            f"🚨 *DETEKSI KERUSAKAN JALAN*\n"
            f"🕐 `{ts}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🕳️ Potholes : *{pothole}*\n"
            f"🔺 Crack    : *{crack}*\n"
            f"📦 Total    : *{total}*"
        )

        if self._gps_locked and self._gps_position:
            lat = self._gps_position.get("latitude", 0.0)
            lon = self._gps_position.get("longitude", 0.0)
            msg += f"\n📍 [`{lat:.5f}, {lon:.5f}`](https://maps.google.com/?q={lat},{lon})"

        threading.Thread(
            target=self._send_new, args=(msg,), daemon=True
        ).start()

    def _send_new(self, text: str) -> int | None:
        """Kirim pesan baru. Return message_id jika berhasil."""
        result = self._api("sendMessage", {
            "chat_id":    self._chat_id,
            "text":       text,
            "parse_mode": "Markdown"
        })
        if result:
            return result.get("result", {}).get("message_id")
        return None

    def _api(self, method: str, payload: dict) -> dict | None:
        """Panggil Telegram Bot API. Return response JSON atau None."""
        try:
            url  = f"https://api.telegram.org/bot{self._token}/{method}"
            resp = requests.post(url, json=payload, timeout=5)
            return resp.json()
        except Exception:
            return None

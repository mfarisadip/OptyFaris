#!/usr/bin/env python3
"""
OLED I2C Test Script untuk Jetson Orin Nano
Hardware : OLED SSD1306 128x64 (atau 128x32)
Koneksi  : VCC -> 3.3V, GND -> GND, SCL -> Pin 5 (GPIO3), SDA -> Pin 3 (GPIO2)
I2C Bus  : /dev/i2c-1  (default Jetson Orin Nano)
I2C Addr : 0x3C (default) atau 0x3D
"""

import time
import sys

# -------------------------------------------------------
# Cek library yang dibutuhkan
# -------------------------------------------------------
try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    from luma.core.render import canvas
    from PIL import ImageFont, ImageDraw
except ImportError:
    print("[ERROR] Library belum terinstall. Jalankan perintah berikut:")
    print()
    print("  pip install luma.oled pillow")
    print()
    sys.exit(1)

# -------------------------------------------------------
# Konfigurasi
# -------------------------------------------------------
I2C_PORT    = None     # None = auto-detect otomatis
I2C_ADDRESS = 0x3C     # Coba 0x3D jika 0x3C tidak bekerja
WIDTH       = 128
HEIGHT      = 64       # Ganti ke 32 jika OLED 128x32


def auto_detect_i2c_bus(address=I2C_ADDRESS):
    """Scan semua /dev/i2c-* pakai smbus2, return bus yang berisi device di `address`."""
    import glob
    import smbus2
    buses = sorted([int(p.replace("/dev/i2c-", "")) for p in glob.glob("/dev/i2c-*")])
    print(f"[AUTO-DETECT] Scanning bus: {buses} ...")
    for bus in buses:
        try:
            with smbus2.SMBus(bus) as b:
                b.read_byte(address)
            print(f"[AUTO-DETECT] OLED ditemukan di /dev/i2c-{bus} addr=0x{address:02X}")
            return bus
        except Exception:
            continue
    return None


def init_oled():
    """Inisialisasi koneksi OLED via I2C (dengan auto-detect bus)."""
    port = I2C_PORT if I2C_PORT is not None else auto_detect_i2c_bus(I2C_ADDRESS)
    if port is None:
        print(f"[ERROR] OLED tidak ditemukan di bus manapun (address 0x{I2C_ADDRESS:02X})")
        print()
        print("Tips troubleshooting:")
        print("  1. Cek koneksi kabel VCC, GND, SCL, SDA")
        print("  2. Jalankan: sudo i2cdetect -y -r 7  → cari alamat 0x3C atau 0x3D")
        print("  3. Pastikan I2C aktif: sudo modprobe i2c-dev")
        sys.exit(1)
    try:
        serial = i2c(port=port, address=I2C_ADDRESS)
        device = ssd1306(serial, width=WIDTH, height=HEIGHT)
        print(f"[OK] OLED SSD1306 siap di I2C bus={port}, addr=0x{I2C_ADDRESS:02X}")
        return device
    except Exception as e:
        print(f"[ERROR] Gagal inisialisasi OLED: {e}")
        sys.exit(1)


def test_1_hello(device):
    """Test 1: Tampilkan teks Hello World."""
    print("[TEST 1] Menampilkan teks Hello World...")
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        draw.text((10, 10), "Hello, Jetson!", fill="white")
        draw.text((10, 28), "OLED I2C Test",  fill="white")
        draw.text((10, 46), "SSD1306 128x64",  fill="white")
    time.sleep(3)


def test_2_animasi_loading(device):
    """Test 2: Animasi progress bar."""
    print("[TEST 2] Animasi loading bar...")
    for i in range(0, 129, 4):
        with canvas(device) as draw:
            draw.text((30, 10), "Loading...", fill="white")
            # Border progress bar
            draw.rectangle((10, 35, 118, 50), outline="white", fill="black")
            # Isi progress bar
            if i > 0:
                draw.rectangle((11, 36, 10 + i, 49), outline="white", fill="white")
            persen = int(i / 128 * 100)
            draw.text((52, 53), f"{persen}%", fill="white")
        time.sleep(0.05)
    time.sleep(1)


def test_3_bouncing_ball(device, durasi=5):
    """Test 3: Bola bergerak (stress test pixel)."""
    print(f"[TEST 3] Bouncing ball selama {durasi} detik...")
    x, y = 10, 10
    vx, vy = 3, 2
    r = 5
    end_time = time.time() + durasi
    while time.time() < end_time:
        with canvas(device) as draw:
            draw.ellipse((x - r, y - r, x + r, y + r), outline="white", fill="white")
        x += vx
        y += vy
        if x - r <= 0 or x + r >= WIDTH:
            vx = -vx
        if y - r <= 0 or y + r >= HEIGHT:
            vy = -vy
        time.sleep(0.03)


def test_4_border_dan_garis(device):
    """Test 4: Menggambar garis dan border."""
    print("[TEST 4] Menggambar garis dan shapes...")
    with canvas(device) as draw:
        # Border luar
        draw.rectangle((0, 0, WIDTH - 1, HEIGHT - 1), outline="white")
        # Garis diagonal
        draw.line((0, 0, WIDTH - 1, HEIGHT - 1), fill="white")
        draw.line((WIDTH - 1, 0, 0, HEIGHT - 1), fill="white")
        # Lingkaran di tengah
        cx, cy = WIDTH // 2, HEIGHT // 2
        draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), outline="white")
    time.sleep(3)


def test_5_info_sistem(device):
    """Test 5: Tampilkan info sistem."""
    import platform
    print("[TEST 5] Menampilkan info sistem...")
    hostname = platform.node()
    arch     = platform.machine()
    now      = time.strftime("%H:%M:%S")
    with canvas(device) as draw:
        draw.text((0,  0), "=== SISTEM INFO ===", fill="white")
        draw.text((0, 14), f"Host: {hostname[:16]}", fill="white")
        draw.text((0, 26), f"Arch: {arch}",         fill="white")
        draw.text((0, 38), f"Time: {now}",           fill="white")
        draw.text((0, 50), "Jetson Orin Nano",       fill="white")
    time.sleep(4)


def test_6_clear(device):
    """Test 6: Bersihkan layar (semua piksel mati)."""
    print("[TEST 6] Membersihkan layar...")
    device.clear()
    time.sleep(1)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    device = init_oled()

    print("=" * 50)
    print("  OLED I2C Test - Jetson Orin Nano")
    print("=" * 50)
    print(f"  I2C Address : 0x{I2C_ADDRESS:02X}")
    print(f"  Resolusi    : {WIDTH}x{HEIGHT}")
    print("=" * 50)
    print()

    tests = [
        ("Hello World",      lambda: test_1_hello(device)),
        ("Loading Bar",      lambda: test_2_animasi_loading(device)),
        ("Bouncing Ball",    lambda: test_3_bouncing_ball(device, durasi=5)),
        ("Lines & Shapes",   lambda: test_4_border_dan_garis(device)),
        ("System Info",      lambda: test_5_info_sistem(device)),
        ("Clear Screen",     lambda: test_6_clear(device)),
    ]

    for idx, (nama, fn) in enumerate(tests, 1):
        print(f"\n--- Test {idx}/{len(tests)}: {nama} ---")
        fn()

    print("\n[SELESAI] Semua test selesai! OLED berfungsi normal.")
    device.clear()


if __name__ == "__main__":
    main()

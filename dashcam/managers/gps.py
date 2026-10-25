"""
GPS data acquisition and NMEA parsing
"""
import serial
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Any

class GPSManager:
    """Optimized GPS manager"""

    def __init__(self):
        self.serial_port = None
        self.nmea_port = None
        self.is_running = False
        self.gps_thread = None
        self.position = {'latitude': 0.0, 'longitude': 0.0}
        self.last_update = None
        self.lock = threading.Lock()
        self.ports = ["/dev/ttyUSB3", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB4", "/dev/ttyUSB5"]
        self.device = "/dev/gps0"

    def _connect_to_port(self, port: str) -> Optional[serial.Serial]:
        """Connect to USB port"""
        try:
            ser = serial.Serial(port, 115200, timeout=1)
            ser.write(b'AT\r\n')
            time.sleep(0.5)
            if ser.inWaiting() and b'OK' in ser.read(ser.inWaiting()):
                return ser
            ser.close()
        except Exception:
            pass
        return None

    def _parse_nmea(self, sentence: str) -> Optional[Dict[str, float]]:
        """Parse NMEA sentence"""
        try:
            if not sentence.startswith('$'):
                return None

            if '*' in sentence:
                sentence = sentence.split('*')[0]

            parts = sentence.split(',')
            valid_prefixes = ['$GPGGA', '$GNGGA', '$GPRMC', '$GNRMC', '$GNGNS']

            if parts[0] not in valid_prefixes:
                return None

            # Extract coordinates based on sentence type
            if parts[0] in ['$GPGGA', '$GNGGA', '$GNGNS']:
                lat_idx, lat_dir_idx, lon_idx, lon_dir_idx = 2, 3, 4, 5
            else:  # RMC
                lat_idx, lat_dir_idx, lon_idx, lon_dir_idx = 3, 4, 5, 6
                if len(parts) < 7 or parts[2] != 'A':
                    return None

            if len(parts) <= lon_dir_idx:
                return None

            lat = self._to_decimal(parts[lat_idx], parts[lat_dir_idx])
            lon = self._to_decimal(parts[lon_idx], parts[lon_dir_idx])

            return {'latitude': lat, 'longitude': lon} if lat and lon else None

        except Exception:
            return None

    def _to_decimal(self, coord: str, direction: str) -> Optional[float]:
        """Convert NMEA coordinate to decimal"""
        try:
            if not coord or not direction or len(coord) < 4:
                return None

            coord = coord.strip()
            direction = direction.strip()

            if direction in ['N', 'S']:
                deg = float(coord[:2])
                min_val = float(coord[2:])
                decimal = deg + min_val / 60
                return -decimal if direction == 'S' else decimal
            elif direction in ['E', 'W']:
                deg = float(coord[:3])
                min_val = float(coord[3:])
                decimal = deg + min_val / 60
                return -decimal if direction == 'W' else decimal
        except Exception:
            pass
        return None

    def initialize(self) -> bool:
        """Initialize GPS module"""
        # Connect to USB for AT commands
        for port in self.ports:
            self.serial_port = self._connect_to_port(port)
            if self.serial_port:
                self.serial_port.write(b'AT+CGPS=1,1\r\n')
                time.sleep(2)
                self.serial_port.close()
                self.serial_port = None
                break

        # Connect to NMEA stream
        for _ in range(3):
            try:
                self.nmea_port = serial.Serial(self.device, 115200, timeout=1)
                if self.nmea_port.is_open:
                    return True
            except Exception:
                time.sleep(1)
        return False

    def _update_loop(self):
        """GPS update thread"""
        buffer = ""

        while self.is_running:
            try:
                if not self.nmea_port or not self.nmea_port.is_open:
                    time.sleep(1)
                    continue

                if self.nmea_port.inWaiting() > 0:
                    data = self.nmea_port.read(self.nmea_port.inWaiting()).decode('ascii', errors='ignore')
                    buffer += data

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        parsed = self._parse_nmea(line)
                        if parsed:
                            with self.lock:
                                self.position.update(parsed)
                                self.last_update = datetime.now()

                time.sleep(0.1)
            except Exception:
                time.sleep(1)

    def start(self) -> bool:
        """Start GPS monitoring"""
        if not self.is_running and self.initialize():
            self.is_running = True
            self.gps_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.gps_thread.start()
            return True
        return False

    def get_position(self) -> Dict[str, Any]:
        """Get current position"""
        with self.lock:
            return {
                'latitude': self.position['latitude'],
                'longitude': self.position['longitude'],
                'last_update': self.last_update
            }

    def stop(self):
        """Stop GPS monitoring"""
        self.is_running = False
        if self.gps_thread and self.gps_thread.is_alive():
            self.gps_thread.join(timeout=1)
        if self.nmea_port:
            try:
                self.nmea_port.close()
            except:
                pass
            self.nmea_port = None

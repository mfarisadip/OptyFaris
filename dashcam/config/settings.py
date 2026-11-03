"""
Configuration and command-line argument parsing
"""
import argparse

# Constants
API_BASE_URL = "https://airoad.roots.web.id"
ANALYZE_URL_DEFAULT = "http://192.168.18.22:8000/analyze"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Ultra-Low Latency Dashcam with Live View')
    
    # Input source
    parser.add_argument('--source', type=str, default='camera', choices=['camera', 'video', 'rtsp'],
                        help='Input source: camera, video file, or rtsp stream')
    parser.add_argument('--video-path', type=str, default='waki.mp4',
                        help='Path to video file')

    # RTSP settings
    parser.add_argument('--rtsp', type=str, default='',
                        help='RTSP stream URL (mis. rtsp://user:pass@ip:554/stream). '
                             'Otomatis set --source rtsp jika diisi.')
    parser.add_argument('--reconnect-delay', type=float, default=3.0,
                        help='Jeda awal (detik) sebelum reconnect setelah stream putus (default: 3.0)')
    parser.add_argument('--max-reconnect', type=int, default=0,
                        help='Batas percobaan reconnect berturut (0=tak terbatas, default: 0)')
    parser.add_argument('--fps-limit', type=float, default=0.0,
                        help='Batasi FPS maksimal pengambilan frame (0=tidak dibatasi)')
    parser.add_argument('--skip', type=int, default=0,
                        help='Frame skip: proses 1 dari N+1 frame (0=off)')
    
    # Camera settings
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera ID')
    parser.add_argument('--camera-type', type=str, default='auto',
                        choices=['auto', 'jetson', 'usb', 'standard'],
                        help='Camera type')
    
    # Output settings
    parser.add_argument('--save-video', action='store_true',
                        help='Save processed video')
    parser.add_argument('--show-cam', action='store_true',
                        help='Show live camera view with detections')
    parser.add_argument('--clean-video', action='store_true',
                        help='Dengan --save-video: rekam frame mentah kamera (lepas dari deteksi); live view tetap annotated')
    parser.add_argument('--save-screenshots', action='store_true',
                        help='Auto-save screenshots on detection')
    
    # Device settings
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Computation device')
    parser.add_argument('--cuda-device', type=int, default=0,
                        help='CUDA device ID')
    
    # Model settings
    parser.add_argument('--model', type=str, default='weights/YOLOV8n320IR8.onnx',
                        help='Model path')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')

    # Detection zone settings
    # Default optimized for forward-facing dashcam (same as reference OptyRizky v1.3)
    #   top_line=0.35   → ignore sky/horizon (top 35%)
    #   bottom_line=0.95 → capture almost entire road area (bottom 95%)
    parser.add_argument('--top-line', type=float, default=0.35,
                        help='Top line position (percentage from top)')
    parser.add_argument('--bottom-line', type=float, default=0.95,
                        help='Bottom line position (percentage from top)')
    parser.add_argument('--no-zone', action='store_true', default=False,
                        help='Nonaktifkan zone filtering (full frame mode, sama seperti yolo_tracker.py)')
    
    # API settings
    parser.add_argument('--vehicle-token', type=str,
                        help='Vehicle API token')
    parser.add_argument('--analyze', action='store_true',
                        help='Send detections to /analyze endpoint')
    parser.add_argument('--analyze-url', type=str, default=ANALYZE_URL_DEFAULT,
                        help='Analyze API URL')
    parser.add_argument('--analyze-source', type=str, default=None,
                        help='Analyze source field (default: camera_{id})')
    parser.add_argument('--analyze-name', type=str, default=None,
                        help='Analyze session name (default: dashcam/YYYYMMDD-HHMMSS)')
    
    # Lane detection settings
    parser.add_argument('--no-lane', action='store_true', default=False,
                        help='Nonaktifkan lane + active lane detection (stripe cat marka -> titik hilang '
                             'otomatis -> marka lewat VP -> pasangan ego lane -> fit + smoothing)')
    parser.add_argument('--lanes', type=int, default=None,
                        help='Override jumlah jalur (mis. 3). Default: deteksi otomatis '
                             'dari marka pembatas (tanda "|") dengan voting temporal')
    parser.add_argument('--lane-work-h', type=int, default=360,
                        help='Tinggi resolusi kerja lane detection (kecil = ringan)')
    parser.add_argument('--lane-skip', type=int, default=2,
                        help='Recompute lane tiap N frame (hasil antara pakai cache)')
    parser.add_argument('--lane-roi-top', type=float, default=None,
                        help='Batas atas manual area lane (fraksi tinggi frame). Default: otomatis '
                             'dari titik hilang (horizon) — isi hanya kalau kalibrasi otomatis gagal')
    parser.add_argument('--lane-roi-bottom', type=float, default=None,
                        help='Batas bawah manual area lane (fraksi tinggi frame, mis. 0.85). Default: '
                             'otomatis — kap mesin / dashboard terdeteksi dari dukungan marka')
    parser.add_argument('--lane-white-only', action='store_true', default=False,
                        help='Hanya pakai marka putih (default: marka putih + kuning)')
    parser.add_argument('--hide-lane-lines', action='store_true', default=False,
                        help='Jangan gambar garis lajur + angka jalur di tengah lajur '
                             '(lane detection, label jalur pothole/crack & info pojok tetap jalan)')

    # Performance settings
    parser.add_argument('--target-fps', type=int, default=30,
                        help='Target FPS for dashcam')
    parser.add_argument('--buffer-size', type=int, default=30,
                        help='Frame buffer size for YOLO processing (higher = less drops, more latency)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='YOLO input image size (default: 640)')
    parser.add_argument('--batch-size', type=int, default=1,
                         help='Batch size for video inference (1 = per-frame; >1 = batch)')
    parser.add_argument('--half', action='store_true', default=True,
                        help='FP16 half-precision on GPU (default: ON)')
    parser.add_argument('--no-half', action='store_false', dest='half',
                        help='Disable FP16 half-precision')
    
    return parser.parse_args()

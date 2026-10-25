#!/bin/bash
# Gabung 3 video: atas kiri = v5_3_960, atas kanan = v5_3_640, bawah (tengah) = 7000.
# Pakai: ./gabung_3video.sh [folder] [durasi_detik_opsional_untuk_tes]

source /mnt/d/workspace/venv/bin/activate

DIR="${1:-/mnt/e/Video Analisa RDD/7000 + NS/New folder}"
DUR="$2"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

if ! mountpoint -q /mnt/e; then
    echo "Drive E: belum di-mount. Jalankan dulu: sudo mount -t drvfs E: /mnt/e"
    exit 1
fi

lbl="scale=960:-2,pad=iw:ih+36:0:36:color=%s,drawtext=fontfile=$FONT:text='%s':x=10:y=8:fontsize=20:fontcolor=white"

for left in "$DIR"/*_v5_3_960.mp4; do
    [ -f "$left" ] || { echo "Tidak ada *_v5_3_960.mp4 di $DIR"; exit 1; }
    base="${left%_v5_3_960.mp4}"
    right="${base}_v5_3_640.mp4"
    bottom="${base}_7000.mp4"
    for f in "$right" "$bottom"; do [ -f "$f" ] || { echo "✗ File tidak ada: $f"; continue 2; }; done
    out="${base}_v5_3_960_vs_640_vs_7000${DUR:+_test}.mp4"
    echo "Menggabungkan: $(basename "$base") -> $out"
    ffmpeg -v error -stats -y ${DUR:+-t $DUR} -i "$left" ${DUR:+-t $DUR} -i "$right" ${DUR:+-t $DUR} -i "$bottom" -filter_complex \
        "[0:v]$(printf "$lbl" 0x5a5a5a 'KIRI\: v5_3 imgsz 960')[a];\
[1:v]$(printf "$lbl" 0x14508c 'KANAN\: v5_3 imgsz 640')[b];\
[2:v]$(printf "$lbl" 0x2e7d32 'BAWAH\: 7000'),pad=1920:ih:480:0:color=black[c];\
[a][b]hstack=inputs=2[top];[top][c]vstack=inputs=2[v]" \
        -map "[v]" -c:v h264_nvenc -preset p5 -cq 23 -pix_fmt yuv420p "$out" \
        && echo "✓ Selesai: $out" || echo "✗ Gagal: $base"
done

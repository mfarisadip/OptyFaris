#!/bin/bash
# Gabung ulang semua hasil per_model: label di bilah atas (pad), tidak menutupi overlay app.
OUT_ROOT="/mnt/e/compare_v5_2_vs_v5_3"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
lbl="scale=960:-2,pad=iw:ih+36:0:36:color=%s,drawtext=fontfile=$FONT:text='%s':x=10:y=8:fontsize=20:fontcolor=white"
for left in "$OUT_ROOT"/*/per_model/*_v5_2.mp4; do
    right="${left%_v5_2.mp4}_v5_3.mp4"
    [ -f "$right" ] || continue
    name=$(basename "${left%_v5_2.mp4}")
    final="$(dirname "$(dirname "$left")")/${name}_v5_2_vs_v5_3.mp4"
    echo "Menggabungkan: $final"
    ffmpeg -v error -y -i "$left" -i "$right" -filter_complex \
        "[0:v]$(printf "$lbl" 0x5a5a5a 'KIRI\: v5_2')[l];[1:v]$(printf "$lbl" 0x14508c 'KANAN\: v5_3')[r];[l][r]hstack=inputs=2[v]" \
        -map "[v]" -c:v h264_nvenc -preset p5 -cq 23 -pix_fmt yuv420p "$final" && echo "✓ $final"
done

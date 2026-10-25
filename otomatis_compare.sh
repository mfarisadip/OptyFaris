#!/bin/bash
# Jalankan app_faris.py dengan v5_2 & v5_3 untuk semua video bpjt & roots,
# lalu gabungkan side-by-side (kiri v5_2, kanan v5_3) ke HDD eksternal E:.

cd /mnt/d/workspace/OptyFaris
source /mnt/d/workspace/venv/bin/activate

GREEN='\033[0;32m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'

OUT_ROOT="/mnt/e/compare_v5_2_vs_v5_3"
MODELS=(v5_2 v5_3)
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
total=0; success=0; failed=0

if ! mountpoint -q /mnt/e; then
    echo -e "${RED}Drive E: belum di-mount. Jalankan dulu: sudo mount -t drvfs E: /mnt/e${NC}"
    exit 1
fi

run_model() {  # $1=video $2=model $3=dest
    python app_faris.py \
        --source video \
        --video-path "$1" \
        --device cuda \
        --cuda-device 0 \
        --model "/mnt/d/workspace/OptyFaris/weights/$2.pt" \
        --no-zone \
        --save-video \
        --no-lane \
        --conf 0.35 \
        --imgsz 960 || return 1
    local base=$(basename "${1%.*}")
    mv "output/${base}_annotated.mp4" "$3" && mv "output/${base}_report.txt" "${3%.mp4}_report.txt" 2>/dev/null
    [ -f "$3" ]
}

for video in video-bpjt/*.[mM][pP]4 video-roots/*.[mM][pP]4; do
    [ -f "$video" ] || continue
    total=$((total + 1))
    folder=$(basename "$(dirname "$video")")
    name=$(basename "${video%.*}")
    raw_dir="$OUT_ROOT/$folder/per_model"
    mkdir -p "$raw_dir"
    final="$OUT_ROOT/$folder/${name}_v5_2_vs_v5_3.mp4"
    echo -e "${GREEN}[${total}] Memproses: ${video}${NC}"

    ok=1
    for m in "${MODELS[@]}"; do
        dest="$raw_dir/${name}_${m}.mp4"
        if [ -f "$dest" ]; then echo "  (skip, sudah ada: $dest)"; continue; fi
        echo -e "${BLUE}  -> model ${m}${NC}"
        run_model "$video" "$m" "$dest" || { ok=0; break; }
    done

    if [ $ok -eq 1 ]; then
        echo -e "${BLUE}  -> menggabungkan side-by-side${NC}"
        # Label di bilah atas (pad) supaya tidak menutupi overlay info dari app
        lbl="scale=960:-2,pad=iw:ih+36:0:36:color=%s,drawtext=fontfile=$FONT:text='%s':x=10:y=8:fontsize=20:fontcolor=white"
        ffmpeg -v error -y -i "$raw_dir/${name}_v5_2.mp4" -i "$raw_dir/${name}_v5_3.mp4" -filter_complex \
            "[0:v]$(printf "$lbl" 0x5a5a5a 'KIRI\: v5_2')[l];[1:v]$(printf "$lbl" 0x14508c 'KANAN\: v5_3')[r];[l][r]hstack=inputs=2[v]" \
            -map "[v]" -c:v h264_nvenc -preset p5 -cq 23 -pix_fmt yuv420p "$final" || ok=0
    fi

    if [ $ok -eq 1 ]; then
        success=$((success + 1)); echo -e "${GREEN}✓ Selesai: ${final}${NC}\n"
    else
        failed=$((failed + 1)); echo -e "${RED}✗ Gagal: ${video}${NC}\n"
    fi
done

echo -e "${BLUE}========================================${NC}"
echo -e "Total: ${total}  ${GREEN}Berhasil: ${success}${NC}  ${RED}Gagal: ${failed}${NC}"
echo -e "Output: ${OUT_ROOT}"

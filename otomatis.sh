#!/bin/bash

# Script untuk menjalankan app_faris.py untuk semua video di folder data/

# Warna untuk output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Counter
total=0
success=0
failed=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Memproses semua video di folder data/${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Loop untuk setiap file video di folder data/
for video in /mnt/d/workspace/OptyFaris/video-analisa/*.[mM][pP]4; do
    # Cek apakah file ada (untuk handle kasus folder kosong)
    if [ ! -f "$video" ]; then
        echo -e "${RED}Tidak ada file video di folder data/${NC}"
        exit 1
    fi
    
    total=$((total + 1))
    
    echo -e "${GREEN}[${total}] Memproses: ${video}${NC}"
    echo "----------------------------------------"

    # Jalankan command
    python app_faris.py \
        --source video \
        --video-path "$video" \
        --device cuda \
        --cuda-device 0 \
        --model /mnt/d/workspace/OptyFaris/weights/v5_4b.pt \
        --no-zone \
        --save-video \
	    --conf 0.45
    
    # Cek status eksekusi
    if [ $? -eq 0 ]; then
        success=$((success + 1))
        echo -e "${GREEN}✓ Berhasil memproses: ${video}${NC}\n"
    else
        failed=$((failed + 1))
        echo -e "${RED}✗ Gagal memproses: ${video}${NC}\n"
    fi
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  RINGKASAN EKSEKUSI${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total video diproses: ${total}"
echo -e "${GREEN}Berhasil: ${success}${NC}"
echo -e "${RED}Gagal: ${failed}${NC}"
echo -e "${BLUE}========================================${NC}"

# Grand Total dari semua JSON result
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GRAND TOTAL DETEKSI${NC}"
echo -e "${BLUE}========================================${NC}"
python3 - <<'EOF'
import json
import os
import glob

output_dir = "output"
json_files = glob.glob(os.path.join(output_dir, "*_result.json"))

if not json_files:
    print("  (Tidak ada file JSON result ditemukan)")
else:
    grand_total = {}
    video_results = []

    for jf in sorted(json_files):
        try:
            with open(jf) as f:
                data = json.load(f)
            video_results.append(data)
            for cls, cnt in data.get("counts", {}).items():
                grand_total[cls] = grand_total.get(cls, 0) + cnt
        except Exception as e:
            print(f"  ⚠️  Gagal membaca {jf}: {e}")

    # Per-video
    print("\n📹 Hasil per Video:")
    print("-" * 50)
    for vr in video_results:
        video_name = os.path.basename(vr.get("video", "unknown"))
        print(f"  {video_name}")
        for cls, cnt in vr.get("counts", {}).items():
            print(f"    • {cls}: {cnt}")
        print(f"    → Total: {vr.get('total', 0)}")

    # Grand total
    print("\n" + "=" * 50)
    print("📊 TOTAL KESELURUHAN:")
    for cls, cnt in sorted(grand_total.items()):
        print(f"  • {cls}: {cnt}")
    print(f"  → Grand Total: {sum(grand_total.values())}")
    print("=" * 50)
EOF
echo -e "${BLUE}========================================${NC}"

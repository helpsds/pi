#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the Base/FT evaluation directory}"
OUTPUT="${OUTPUT:-$ROOT/assets/videos/pi05_base_vs_ft_demo.mp4}"
FONT="${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}"

BASE="$SOURCE_ROOT/base/seed_12000"
FT="$SOURCE_ROOT/ft_030000/seed_12000"
RED=task_1_red_cube_to_blue_bin/rollout.mp4
GREEN=task_3_green_cube_to_blue_bin/rollout.mp4
BOX=task_6_small_box_to_red_bin/rollout.mp4

for video in "$BASE/$RED" "$FT/$RED" "$BASE/$GREEN" "$FT/$GREEN" "$BASE/$BOX" "$FT/$BOX"; do
    [[ -f "$video" ]] || { echo "missing input: $video" >&2; exit 1; }
done
[[ -f "$FONT" ]] || { echo "missing font: $FONT" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"

ffmpeg -y -hide_banner \
  -f lavfi -i "color=c=0x101820:s=1280x720:r=20:d=3" \
  -i "$BASE/$RED" -i "$FT/$RED" \
  -i "$BASE/$GREEN" -i "$FT/$GREEN" \
  -i "$BASE/$BOX" -i "$FT/$BOX" \
  -f lavfi -i "color=c=0x101820:s=1280x720:r=20:d=4" \
  -filter_complex "
    [0:v]drawtext=fontfile='$FONT':text='pi0.5 MuJoCo Sorting':fontcolor=white:fontsize=52:x=(w-text_w)/2:y=245,
         drawtext=fontfile='$FONT':text='Base vs Fine-tuned (30k steps)':fontcolor=0x5BC0EB:fontsize=32:x=(w-text_w)/2:y=325[intro];

    [1:v]scale=600:600:flags=lanczos,tpad=stop_mode=clone:stop_duration=15,trim=duration=15,setpts=PTS-STARTPTS[b1];
    [2:v]scale=600:600:flags=lanczos,tpad=stop_mode=clone:stop_duration=15,trim=duration=15,setpts=PTS-STARTPTS[f1];
    [b1][f1]hstack=inputs=2,pad=1280:720:40:100:0x101820,
         drawtext=fontfile='$FONT':text='Red cube to blue bin':fontcolor=white:fontsize=30:x=(w-text_w)/2:y=20,
         drawtext=fontfile='$FONT':text='BASE - FAIL':fontcolor=0xFF5A5F:fontsize=27:x=225:y=60,
         drawtext=fontfile='$FONT':text='FT 30K - SUCCESS':fontcolor=0x35D07F:fontsize=27:x=825:y=60[s1];

    [3:v]scale=600:600:flags=lanczos,tpad=stop_mode=clone:stop_duration=15,trim=duration=15,setpts=PTS-STARTPTS[b2];
    [4:v]scale=600:600:flags=lanczos,tpad=stop_mode=clone:stop_duration=15,trim=duration=15,setpts=PTS-STARTPTS[f2];
    [b2][f2]hstack=inputs=2,pad=1280:720:40:100:0x101820,
         drawtext=fontfile='$FONT':text='Green cube to blue bin':fontcolor=white:fontsize=30:x=(w-text_w)/2:y=20,
         drawtext=fontfile='$FONT':text='BASE - FAIL':fontcolor=0xFF5A5F:fontsize=27:x=225:y=60,
         drawtext=fontfile='$FONT':text='FT 30K - SUCCESS':fontcolor=0x35D07F:fontsize=27:x=825:y=60[s2];

    [5:v]scale=600:600:flags=lanczos,tpad=stop_mode=clone:stop_duration=15,trim=duration=15,setpts=PTS-STARTPTS[b3];
    [6:v]scale=600:600:flags=lanczos,tpad=stop_mode=clone:stop_duration=15,trim=duration=15,setpts=PTS-STARTPTS[f3];
    [b3][f3]hstack=inputs=2,pad=1280:720:40:100:0x101820,
         drawtext=fontfile='$FONT':text='Small box to red bin':fontcolor=white:fontsize=30:x=(w-text_w)/2:y=20,
         drawtext=fontfile='$FONT':text='BASE - FAIL':fontcolor=0xFF5A5F:fontsize=27:x=225:y=60,
         drawtext=fontfile='$FONT':text='FT 30K - SUCCESS':fontcolor=0x35D07F:fontsize=27:x=825:y=60[s3];

    [7:v]drawtext=fontfile='$FONT':text='Paired 10-seed benchmark':fontcolor=white:fontsize=42:x=(w-text_w)/2:y=205,
         drawtext=fontfile='$FONT':text='Base 0 / 80   |   FT 30K 43 / 80':fontcolor=0x5BC0EB:fontsize=36:x=(w-text_w)/2:y=290,
         drawtext=fontfile='$FONT':text='Task success  0 percent  to  53.75 percent':fontcolor=0x35D07F:fontsize=32:x=(w-text_w)/2:y=365[outro];

    [intro][s1][s2][s3][outro]concat=n=5:v=1:a=0,format=yuv420p[outv]
  " \
  -map "[outv]" -an -c:v libopenh264 -b:v 4M -maxrate 6M -bufsize 8M -movflags +faststart "$OUTPUT"

ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$OUTPUT"

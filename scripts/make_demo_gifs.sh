#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$ROOT/assets/videos"
OUTPUT="$ROOT/assets/gifs"
mkdir -p "$OUTPUT"

render_gif() {
    local name="$1" width="$2" fps="$3" colors="$4"
    ffmpeg -hide_banner -loglevel error -y -i "$INPUT/$name.mp4" \
        -filter_complex "[0:v]fps=$fps,scale=$width:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=$colors:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
        -loop 0 "$OUTPUT/$name.gif"
    printf '%s\n' "$OUTPUT/$name.gif"
}

# Keep the full 52-second paired comparison; reduce only its resolution/FPS.
render_gif pi05_base_vs_ft_demo 640 5 96
render_gif base_red_cube_blue_bin_failure 256 10 128
render_gif ft_red_cube_blue_bin_success 256 10 128
render_gif ft_green_cube_blue_bin_success 256 10 128
render_gif ft_cylinder_place_failure 256 10 128

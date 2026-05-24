#!/bin/bash
# Re-encode LeIsaac mp4 (AV1 codec) → H.264 for DreamZero decord backend.
# AV1 decode is unsupported by decord; H.264 is fast (~17× faster than ffmpeg subprocess backend).
# Output: <SRC_DIR>-h264/ with same dir structure + symlinked data/ and meta/.
#
# Usage:
#   bash reencode_av1_to_h264.sh /root/autodl-tmp/leisaac-pick-orange [PARALLEL=8]
#
# Runtime: 60 demo × 2 cam = 120 mp4, ~30s with 8-worker parallel on 24-thread CPU.

set -e
SRC_ROOT=${1:-/root/autodl-tmp/leisaac-pick-orange}
PARALLEL=${PARALLEL:-8}
DST_ROOT=${SRC_ROOT}-h264

if [ ! -d "$SRC_ROOT" ]; then
    echo "ERROR: $SRC_ROOT not found"
    exit 1
fi

if [ -d "$DST_ROOT" ]; then
    echo "WARNING: $DST_ROOT exists, will overwrite"
fi

mkdir -p "$DST_ROOT"

# Re-encode all mp4 files
cat > /tmp/_reencode_one.sh <<EOF
#!/bin/bash
SRC=\$1
DST=\$(echo \$SRC | sed "s|${SRC_ROOT}/|${DST_ROOT}/|")
mkdir -p \$(dirname \$DST)
ffmpeg -y -loglevel error -i \$SRC -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p \$DST
EOF
chmod +x /tmp/_reencode_one.sh

echo "=== Re-encoding $(find "$SRC_ROOT/videos" -name "*.mp4" | wc -l) mp4 files (parallel=$PARALLEL) ==="
time find "$SRC_ROOT/videos" -name "*.mp4" | xargs -P "$PARALLEL" -I{} /tmp/_reencode_one.sh {}

echo "=== Copy non-video files (data, meta) ==="
for d in data meta; do
    rm -rf "$DST_ROOT/$d"
    cp -a "$SRC_ROOT/$d" "$DST_ROOT/$d"
done

echo "=== Done ==="
find "$DST_ROOT" -name "*.mp4" | wc -l | xargs -I{} echo "Total H.264 files: {}"
du -sh "$DST_ROOT"

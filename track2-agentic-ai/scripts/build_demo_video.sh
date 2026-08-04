#!/bin/bash
# Build the reproducible 1080p fallback demo video.
#
# Combines static frames (JPG) with neural TTS narration (MP3) into
# 7 segments, then concatenates them into the final video.
#
# Output: output/video/track2_demo_1080p_ava.mp4
#
# Requires: ffmpeg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FRAMES_DIR="$PROJECT_ROOT/output/video/frames"
AUDIO_DIR="$PROJECT_ROOT/output/video/audio"
SEGMENTS_DIR="$PROJECT_ROOT/output/video/segments"
OUTPUT="$PROJECT_ROOT/output/video/track2_demo_1080p_ava.mp4"

# scene_name|frame_name  (frame filenames differ for scenes 6-7)
SCENES=(
  "01-hero|01-hero"
  "02-dashboard|02-dashboard"
  "03-capabilities|03-capabilities"
  "04-architecture|04-architecture"
  "05-proof|05-proof"
  "06-live-dsl|06-result-dsl"
  "07-risk|07-result-risk"
)

echo "Building Track 2 demo video..."
echo "  Frames:   $FRAMES_DIR"
echo "  Audio:    $AUDIO_DIR"
echo "  Segments: $SEGMENTS_DIR"
echo "  Output:   $OUTPUT"
echo ""

mkdir -p "$SEGMENTS_DIR"

# Step 1: Build each segment (frame + audio → MP4)
for entry in "${SCENES[@]}"; do
  scene="${entry%%|*}"
  frame_name="${entry##*|}"
  frame="$FRAMES_DIR/${frame_name}.jpg"
  audio="$AUDIO_DIR/${scene}-neural.mp3"
  seg="$SEGMENTS_DIR/${scene%%-*}.mp4"

  if [[ ! -f "$frame" ]]; then
    echo "  [SKIP] Frame missing: $frame"
    continue
  fi
  if [[ ! -f "$audio" ]]; then
    echo "  [SKIP] Audio missing: $audio"
    continue
  fi

  # Get audio duration to set the frame display time
  duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$audio")

  echo "  [$scene] duration=${duration}s"

  ffmpeg -y -loop 1 -i "$frame" -i "$audio" \
    -c:v libx264 -tune stillimage -pix_fmt yuv420p \
    -r 30 -t "$duration" \
    -c:a aac -b:a 192k -ar 44100 \
    -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black" \
    -shortest \
    "$seg" -loglevel warning
done

# Step 2: Concatenate all segments
concat_list="$SEGMENTS_DIR/concat.txt"
> "$concat_list"
for entry in "${SCENES[@]}"; do
  scene="${entry%%|*}"
  seg="$SEGMENTS_DIR/${scene%%-*}.mp4"
  if [[ -f "$seg" ]]; then
    echo "file '$seg'" >> "$concat_list"
  fi
done

echo ""
echo "Concatenating segments..."
ffmpeg -y -f concat -safe 0 -i "$concat_list" \
  -c copy \
  "$OUTPUT" -loglevel warning

echo ""
echo "Done: $OUTPUT"
ls -lh "$OUTPUT"

#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/video_generation/inha_challenge/ChoiSeongYong/Inha_challenge"
PYTHON_BIN="/home/video_generation/inha_challenge/ChoiSeongYong/.miniforge3/envs/inha_sy/bin/python"
DOWNLOAD_LOG="$PROJECT_ROOT/outputs/abot_so100_vace_1800/resilient_download.log"
SUPERVISOR_LOG="$PROJECT_ROOT/outputs/abot_so100_vace_1800/download_supervisor.log"

mkdir -p "$(dirname "$DOWNLOAD_LOG")"
cd "$PROJECT_ROOT" || exit 1

while true; do
    if grep -q "all Wan public weights validated" "$DOWNLOAD_LOG" 2>/dev/null; then
        echo "$(date '+%F %T') download completed" >> "$SUPERVISOR_LOG"
        exit 0
    fi

    echo "$(date '+%F %T') starting downloader" >> "$SUPERVISOR_LOG"
    "$PYTHON_BIN" scripts/download_wan_resilient.py >> "$DOWNLOAD_LOG" 2>&1
    rc=$?
    echo "$(date '+%F %T') downloader exited rc=$rc; partial files preserved; restarting in 15s" >> "$SUPERVISOR_LOG"
    sleep 15
done

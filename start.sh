#!/bin/sh
set -e

cd /app

# 启动 face_recognition 服务
echo "[start.sh] 启动 face_recognition 服务 (FACE_RECOGNITION_PORT=${FACE_RECOGNITION_PORT:-3080})..."
nohup ./face_recognition >> /app/logs/app.log 2>&1 &

# 启动

echo "[start.sh] 服务已在后台运行，日志: /app/logs/app.log"
echo "[start.sh] PID: $!"

exec tail -f /dev/null

# syntax=docker/dockerfile:1

# ---------- 构建阶段 ----------
FROM golang:1.25-alpine AS builder

WORKDIR /build

RUN apk add --no-cache ca-certificates git
RUN go env -w GOPROXY=https://goproxy.cn,direct 

# 复制 face_recognition 目录
COPY face_recognition ./face_recognition

# 构建 face_recognition
WORKDIR /build/face_recognition
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags "-s -w" -o face_recognition .

# 构建 ...

# ---------- 运行阶段 ----------
FROM alpine:3.20

RUN apk add --no-cache ca-certificates tzdata

WORKDIR /app

RUN mkdir -p /app/logs

# 复制 face_recognition 文件
COPY --from=builder /build/face_recognition/face_recognition .
COPY --from=builder /build/face_recognition/templates ./templates/

# 复制 ...

COPY start.sh ./start.sh
RUN chmod +x ./start.sh

ENTRYPOINT ["./start.sh"]

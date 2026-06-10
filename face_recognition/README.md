# Face Recognition - 基于 Immich ML + pgvector 的人脸识别服务

上传图片，调用 Immich ML 服务直接提取人脸特征，通过 PostgreSQL pgvector 与数据库中已有的人脸 embedding 做余弦距离匹配，返回人脸坐标及人物信息。

**只读模式**：不写入数据库，仅查询 `face_search` 和 `asset_face` 表。

## 工作流程

```
用户上传图片
    │
    ▼
┌──────────────────────────────────────┐
│ 1. 调用 Immich ML /predict           │
│    单次调用同时完成检测+识别           │
│    返回 boundingBox + embedding       │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 2. 用 embedding 查询 pgvector        │
│    Step1: searchFaces (N个最近匹配)   │
│    Step2: 找有 personId 的匹配        │
│    Step3: hasPerson=true 兜底搜索     │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 3. 去重 + 返回结果                   │
│    同一图中同一人只保留距离最小的匹配   │
│    返回坐标 + personId + personName   │
└──────────────────────────────────────┘
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `FACE_RECOGNITION_PORT` | `3080` | 服务监听端口 |
| `IMMICH_MACHINE_LEARNING` | `http://10.222.60.98:3003` | Immich ML 服务地址 |
| `ML_MODEL_NAME` | `antelopev2` | 人脸识别模型名称 |
| `PG_DSN` | `host=10.222.60.98 port=5432 ...` | PostgreSQL 连接串（需 pgvector 扩展） |

内部参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `mlMinScore` | 0.7 | 人脸检测最低置信度 |
| `faceMinFaces` | 3 | 最少匹配人脸数（minFaces） |
| `faceMaxDist` | 0.7 | 余弦距离阈值（maxDistance） |

## 启动

```bash
go build -o face_recognition .
./face_recognition
# 服务监听 http://0.0.0.0:3080
```

## API 接口

### `POST /api/recognize`

上传图片进行人脸识别。

**请求**

- Content-Type: `multipart/form-data`
- 字段: `image` — 图片文件（JPG/PNG）

**响应**

成功 (200):

```json
{
  "fileName": "photo.jpg",
  "totalFaces": 2,
  "faces": [
    {
      "boundingBoxX1": 683,
      "boundingBoxY1": 748,
      "boundingBoxX2": 763,
      "boundingBoxY2": 851,
      "imageWidth": 1920,
      "imageHeight": 1080,
      "score": 0.88,
      "personId": "55215f39-ccac-478b-a9a0-b38f3dde22b6",
      "personName": "张三",
      "distance": 0.35
    },
    {
      "boundingBoxX1": 120,
      "boundingBoxY1": 200,
      "boundingBoxX2": 180,
      "boundingBoxY2": 280,
      "imageWidth": 1920,
      "imageHeight": 1080,
      "score": 0.82,
      "distance": 0.61
    }
  ]
}
```

无人脸 (200):

```json
{
  "fileName": "photo.jpg",
  "totalFaces": 0,
  "faces": []
}
```

请求错误 (400):

```json
{ "error": "请上传图片文件" }
```

服务错误 (500):

```json
{ "error": "ML人脸检测失败: ..." }
```

### `GET /`

Web 页面，提供拖拽上传 + 人脸框可视化。

## 人脸匹配逻辑

遵循 Immich server `person.service.ts` 的 `handleRecognizeFaces` 逻辑：

1. **searchFaces**：用 embedding 在 `face_search` 中搜索最近的 N 个匹配（`faceMinFaces`），过滤距离 <= `faceMaxDist`
2. **找有 personId 的匹配**：在结果中找到第一个已分配 `personId` 的人脸，复用该 person
3. **hasPerson 兜底搜索**：如果没找到有 personId 的，再搜索 `personId IS NOT NULL` 的最近一条
4. **去重**：同一张图中同一个 personId 只保留距离最小的匹配

## 数据结构

### FaceResult

| 字段 | 类型 | 说明 |
|------|------|------|
| `boundingBoxX1` | int | 人脸框左上角 X 坐标（原图像素） |
| `boundingBoxY1` | int | 人脸框左上角 Y 坐标（原图像素） |
| `boundingBoxX2` | int | 人脸框右下角 X 坐标（原图像素） |
| `boundingBoxY2` | int | 人脸框右下角 Y 坐标（原图像素） |
| `imageWidth` | int | 原图宽度（像素） |
| `imageHeight` | int | 原图高度（像素） |
| `score` | float64 | 人脸检测置信度 |
| `personId` | string/null | 匹配到的人物 ID，未匹配时 omitempty |
| `personName` | string/null | 匹配到的人物名称，未匹配时 omitempty |
| `distance` | float64/null | pgvector 余弦距离，供调试 |

## 依赖的数据库表

| 表 | 用途 |
|----|------|
| `face_search` | 存储 embedding 向量（pgvector），用于相似度搜索 |
| `asset_face` | 存储人脸记录（boundingBox + personId） |
| `person` | 存储人物信息（name 等） |
| `asset` | 存储图片资产，用于过滤已删除的 |

## 前端人脸框绘制

前端根据 `boundingBoxX1/Y1/X2/Y2` 和 `imageWidth/Height` 计算缩放比：

```javascript
const scaleX = displayWidth / face.imageWidth;
const scaleY = displayHeight / face.imageHeight;
const x = face.boundingBoxX1 * scaleX;
const y = face.boundingBoxY1 * scaleY;
const w = (face.boundingBoxX2 - face.boundingBoxX1) * scaleX;
const h = (face.boundingBoxY2 - face.boundingBoxY1) * scaleY;
```

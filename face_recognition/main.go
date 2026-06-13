package main

import (
	"bytes"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	_ "github.com/lib/pq"
	"golang.org/x/image/draw"
)

// 服务配置
var (
	serverEndpoint = getEnv("FACE_RECOGNITION_PORT", "3080")
	mlEndpoint     = getEnv("IMMICH_MACHINE_LEARNING", "http://127.0.0.1:3003")
	mlModelName    = getEnv("ML_MODEL_NAME", "antelopev2")
	pgDSN          = getEnv("PG_DSN", "host=127.0.0.1 port=5432 user=postgres password=postgres dbname=immich sslmode=disable")
	mlMinScore     = 0.7 // 人脸检测最低置信度
)

var (
	mlClient = &http.Client{Timeout: 60 * time.Second}
	db       *sql.DB
)

// Immich 系统配置（人脸识别参数，与 Immich 管理后台一致）
var (
	faceMinFaces = 3   // 最少匹配人脸数（minFaces），只有 >= 此数量才认为是核心匹配
	faceMaxDist  = 0.7 // 余弦距离阈值（maxDistance），截图embedding与原图差异大，需放宽
	maxImageSize = 640 // 发送给 ML 的图片最大边长（px），超过则等比缩放
)

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

// ==================== 图片压缩 ====================

// resizeImage 将图片等比缩放到 maxSide 以内，返回 JPEG 字节
// 如果图片已小于 maxSide 或解码失败，返回原始数据
func resizeImage(data []byte, maxSide int) []byte {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		log.Printf("[压缩] 图片解码失败，使用原始数据: %v", err)
		return data
	}

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// 不需要缩放
	if w <= maxSide && h <= maxSide {
		return data
	}

	// 计算缩放比例
	scale := float64(maxSide) / float64(max(w, h))
	nw, nh := int(float64(w)*scale), int(float64(h)*scale)

	// 高质量缩放
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, bounds, draw.Over, nil)

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: 90}); err != nil {
		log.Printf("[压缩] JPEG编码失败，使用原始数据: %v", err)
		return data
	}

	log.Printf("[压缩] %dx%d -> %dx%d, %dKB -> %dKB", w, h, nw, nh, len(data)/1024, buf.Len()/1024)
	return buf.Bytes()
}

// ==================== ML /predict 直接调用 ====================

// MLPredictResponse ML 服务返回的 facial-recognition 结构
type MLPredictResponse struct {
	FacialRecognition []MLFace `json:"facial-recognition"`
	ImageWidth        int      `json:"imageWidth"`
	ImageHeight       int      `json:"imageHeight"`
}

// MLFace ML 返回的单个人脸
type MLFace struct {
	BoundingBox MLBoundingBox `json:"boundingBox"`
	Embedding   string        `json:"embedding"` // JSON 序列化的 float32 数组
	Score       float64       `json:"score"`
}

// MLBoundingBox ML 返回的边界框
type MLBoundingBox struct {
	X1 float64 `json:"x1"`
	Y1 float64 `json:"y1"`
	X2 float64 `json:"x2"`
	Y2 float64 `json:"y2"`
}

// detectFaces 调用 Immich ML /predict 直接获取人脸特征
func detectFaces(imageData []byte) (*MLPredictResponse, error) {
	// 构建 multipart form: image + entries
	// entries 格式必须符合 Immich ML 的 PipelineRequest:
	// {"facial-recognition":{"detection":{"modelName":"antelopev2","options":{"minScore":0.7}},"recognition":{"modelName":"antelopev2","options":{}}}}
	entries := fmt.Sprintf(
		`{"facial-recognition":{"detection":{"modelName":"%s","options":{"minScore":%g}},"recognition":{"modelName":"%s","options":{}}}}`,
		mlModelName, mlMinScore, mlModelName,
	)

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// 写入 image 字段
	part, err := writer.CreateFormFile("image", "image.jpg")
	if err != nil {
		return nil, fmt.Errorf("创建表单字段失败: %v", err)
	}
	if _, err := part.Write(imageData); err != nil {
		return nil, fmt.Errorf("写入图片数据失败: %v", err)
	}

	// 写入 entries 字段
	if err := writer.WriteField("entries", entries); err != nil {
		return nil, fmt.Errorf("写入entries失败: %v", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("关闭表单失败: %v", err)
	}

	// 发送请求到 ML /predict
	req, err := http.NewRequest("POST", mlEndpoint+"/predict", &buf)
	if err != nil {
		return nil, fmt.Errorf("创建请求失败: %v", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := mlClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ML请求失败: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取ML响应失败: %v", err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("ML返回错误 (HTTP %d): %s", resp.StatusCode, string(body))
	}

	var result MLPredictResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("解析ML响应失败: %v, body: %s", err, string(body[:min(len(body), 500)]))
	}

	return &result, nil
}

// ==================== Postgres pgvector 匹配 ====================

// FaceResult 返回给前端的人脸结果
type FaceResult struct {
	BoundingBoxX1 int      `json:"boundingBoxX1"`
	BoundingBoxY1 int      `json:"boundingBoxY1"`
	BoundingBoxX2 int      `json:"boundingBoxX2"`
	BoundingBoxY2 int      `json:"boundingBoxY2"`
	ImageWidth    int      `json:"imageWidth"`
	ImageHeight   int      `json:"imageHeight"`
	Score         float64  `json:"score"`
	PersonID      *string  `json:"personId,omitempty"`
	PersonName    *string  `json:"personName,omitempty"`
	Distance      *float64 `json:"distance,omitempty"`
}

// matchPerson 按照 Immich 原始逻辑匹配人脸
// 1. 用 embedding 在 face_search 中搜索最近的 N 个匹配（searchFaces）
// 2. 在匹配结果中找已分配 personId 的人脸 → 复用该 person
// 3. 如果没找到有 personId 的，再单独搜 hasPerson=true 的最近匹配
// 4. 返回匹配到的 personID, personName, distance
func matchPerson(embedding string) (*string, *string, *float64) {
	// 第1步：searchFaces - 搜索最近的 N 个匹配（与 Immich searchFaces SQL 一致）
	// 查找 asset_face → face_search → person，按距离排序
	query := fmt.Sprintf(`
		WITH "cte" AS (
			SELECT
				"asset_face"."id",
				"asset_face"."personId",
				face_search.embedding <=> '%s'::vector as "distance"
			FROM
				"asset_face"
				INNER JOIN "asset" ON "asset"."id" = "asset_face"."assetId"
				INNER JOIN "face_search" ON "face_search"."faceId" = "asset_face"."id"
				LEFT JOIN "person" ON "person"."id" = "asset_face"."personId"
			WHERE
				"asset"."deletedAt" IS NULL
				AND "asset_face"."deletedAt" IS NULL
			ORDER BY
				"distance"
			LIMIT %d
		)
		SELECT cte."id", cte."personId", cte."distance", p.name as "personName"
		FROM cte
		LEFT JOIN person p ON cte."personId" = p.id
		WHERE cte."distance" <= %g
	`, embedding, faceMinFaces, faceMaxDist)

	rows, err := db.Query(query)
	if err != nil {
		log.Printf("[匹配] 查询出错: %v", err)
		return nil, nil, nil
	}
	defer rows.Close()

	type faceMatch struct {
		faceID     string
		personID   *string
		personName *string
		distance   float64
	}

	var matches []faceMatch
	for rows.Next() {
		var m faceMatch
		if err := rows.Scan(&m.faceID, &m.personID, &m.distance, &m.personName); err != nil {
			continue
		}
		matches = append(matches, m)
	}

	// 第2步：在匹配结果中找已分配 personId 的人脸（与 Immich handleRecognizeFaces 逻辑一致）
	// matches.find(match => match.personId)
	for _, m := range matches {
		if m.personID != nil && *m.personID != "" {
			name := ""
			if m.personName != nil {
				name = *m.personName
			}
			log.Printf("[匹配] 匹配到: %s (%s), 距离=%.4f (共%d个匹配)", name, *m.personID, m.distance, len(matches))
			pid := *m.personID
			return &pid, &name, &m.distance
		}
	}

	// 第3步：匹配结果中没人有 personId，再单独搜 hasPerson=true 的最近一条
	// 对应 Immich: searchFaces({ hasPerson: true, numResults: 1 })
	query2 := fmt.Sprintf(`
		WITH "cte" AS (
			SELECT
				"asset_face"."id",
				"asset_face"."personId",
				face_search.embedding <=> '%s'::vector as "distance"
			FROM
				"asset_face"
				INNER JOIN "asset" ON "asset"."id" = "asset_face"."assetId"
				INNER JOIN "face_search" ON "face_search"."faceId" = "asset_face"."id"
				LEFT JOIN "person" ON "person"."id" = "asset_face"."personId"
			WHERE
				"asset"."deletedAt" IS NULL
				AND "asset_face"."deletedAt" IS NULL
				AND "asset_face"."personId" IS NOT NULL
			ORDER BY
				"distance"
			LIMIT 1
		)
		SELECT cte."id", cte."personId", cte."distance", p.name as "personName"
		FROM cte
		LEFT JOIN person p ON cte."personId" = p.id
		WHERE cte."distance" <= %g
	`, embedding, faceMaxDist)

	var m faceMatch
	err = db.QueryRow(query2).Scan(&m.faceID, &m.personID, &m.distance, &m.personName)
	if err == nil && m.personID != nil && *m.personID != "" {
		name := ""
		if m.personName != nil {
			name = *m.personName
		}
		log.Printf("[匹配] hasPerson搜索匹配到: %s (%s), 距离=%.4f", name, *m.personID, m.distance)
		pid := *m.personID
		return &pid, &name, &m.distance
	}

	// 没匹配到任何有 person 的人脸
	if len(matches) > 0 {
		log.Printf("[匹配] %d个匹配但无personId，未识别", len(matches))
		return nil, nil, &matches[0].distance
	}

	log.Printf("[匹配] 无匹配结果")
	return nil, nil, nil
}

// ==================== Gin 路由 ====================

type recognizeJSONRequest struct {
	Base64   string `json:"base64"`
	FileName string `json:"fileName,omitempty"`
}

func readRecognizeImageFromForm(c *gin.Context) ([]byte, string, error) {
	file, header, err := c.Request.FormFile("file")
	if err != nil {
		return nil, "", fmt.Errorf("请上传图片文件")
	}
	defer file.Close()

	fileData, err := io.ReadAll(file)
	if err != nil {
		return nil, "", fmt.Errorf("读取文件失败")
	}
	return fileData, header.Filename, nil
}

func readRecognizeImageFromJSON(c *gin.Context) ([]byte, string, error) {
	var req recognizeJSONRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		return nil, "", fmt.Errorf("JSON 解析失败")
	}
	if req.Base64 == "" {
		return nil, "", fmt.Errorf("请提供 base64 字段")
	}

	b64 := strings.TrimSpace(req.Base64)
	if comma := strings.Index(b64, ","); comma != -1 && strings.HasPrefix(b64, "data:") {
		b64 = b64[comma+1:]
	}

	fileData, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, "", fmt.Errorf("base64 解码失败")
	}

	fileName := req.FileName
	if fileName == "" {
		fileName = "image.jpg"
	}
	return fileData, fileName, nil
}

func readRecognizeImage(c *gin.Context) ([]byte, string, error) {
	contentType := c.GetHeader("Content-Type")
	if strings.HasPrefix(contentType, "application/json") {
		return readRecognizeImageFromJSON(c)
	}
	return readRecognizeImageFromForm(c)
}

func handleRecognize(c *gin.Context) {
	fileData, fileName, err := readRecognizeImage(c)
	if err != nil {
		status := http.StatusBadRequest
		if err.Error() == "读取文件失败" {
			status = http.StatusInternalServerError
		}
		c.JSON(status, gin.H{"error": err.Error()})
		return
	}

	// 压缩图片：超过 maxImageSize 则等比缩放，避免 ML 服务 1024KB 限制
	fileData = resizeImage(fileData, maxImageSize)

	// 1. 直接调用 ML /predict 获取人脸特征（同步，无需轮询）
	mlResp, err := detectFaces(fileData)
	if err != nil {
		log.Printf("[错误] ML检测失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("ML人脸检测失败: %v", err)})
		return
	}

	log.Printf("[ML] 检测到%d张人脸, 图片尺寸: %dx%d",
		len(mlResp.FacialRecognition), mlResp.ImageWidth, mlResp.ImageHeight)

	// 2. 用 embedding 通过 pgvector 匹配 person
	faces := make([]FaceResult, 0, len(mlResp.FacialRecognition))
	for i, f := range mlResp.FacialRecognition {
		result := FaceResult{
			BoundingBoxX1: int(f.BoundingBox.X1),
			BoundingBoxY1: int(f.BoundingBox.Y1),
			BoundingBoxX2: int(f.BoundingBox.X2),
			BoundingBoxY2: int(f.BoundingBox.Y2),
			ImageWidth:    mlResp.ImageWidth,
			ImageHeight:   mlResp.ImageHeight,
			Score:         f.Score,
		}

		if f.Embedding != "" {
			personID, personName, distance := matchPerson(f.Embedding)
			result.PersonID = personID
			result.PersonName = personName
			result.Distance = distance
		}

		log.Printf("[人脸#%d] 坐标(%.0f,%.0f)-(%.0f,%.0f) score=%.2f 匹配=%s",
			i+1, f.BoundingBox.X1, f.BoundingBox.Y1, f.BoundingBox.X2, f.BoundingBox.Y2,
			f.Score, ptrStr(result.PersonName, "未识别"))

		faces = append(faces, result)
	}

	// 3. 去重：同一张图中，同一个 person 只匹配距离最小的那张脸
	dedupPersonMatches(faces)

	c.JSON(http.StatusOK, gin.H{
		"fileName":   fileName,
		"totalFaces": len(faces),
		"faces":      faces,
	})
}

func ptrStr(s *string, fallback string) string {
	if s == nil {
		return fallback
	}
	return *s
}

// dedupPersonMatches 同一张图中同一个 person 只保留距离最小的匹配，其余清空
func dedupPersonMatches(faces []FaceResult) {
	// personID -> 最佳匹配的索引和距离
	bestIdx := make(map[string]int)
	bestDist := make(map[string]float64)

	for i, f := range faces {
		if f.PersonID == nil || f.Distance == nil {
			continue
		}
		pid := *f.PersonID
		dist := *f.Distance
		if _, exists := bestIdx[pid]; !exists || dist < bestDist[pid] {
			bestIdx[pid] = i
			bestDist[pid] = dist
		}
	}

	// 清除非最佳匹配的 personId
	for i, f := range faces {
		if f.PersonID == nil {
			continue
		}
		pid := *f.PersonID
		if bestIdx[pid] != i {
			log.Printf("[去重] 人脸#%d 的 %s 匹配被人脸#%d取代(距离更优)", i+1, pid, bestIdx[pid]+1)
			faces[i].PersonID = nil
			faces[i].PersonName = nil
			// 保留 Distance 以供调试
		}
	}
}

// kill -9 $(lsof -t -i:3010) && cd /home/lh/codebuddy/go_script/face_recognition && go build -ldflags "-s -w" -o face_recognition . 2>&1; echo "EXIT: $?"
func main() {
	var err error
	db, err = sql.Open("postgres", pgDSN)
	if err != nil {
		log.Fatalf("连接数据库失败: %v", err)
	}
	defer db.Close()

	db.SetMaxOpenConns(5)
	db.SetMaxIdleConns(2)
	db.SetConnMaxLifetime(5 * time.Minute)

	if err := db.Ping(); err != nil {
		log.Fatalf("数据库连接测试失败: %v", err)
	}
	log.Println("Postgres 连接成功")

	fmt.Printf("ML Endpoint: %s\n", mlEndpoint)
	fmt.Printf("ML Model: %s\n", mlModelName)

	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.Use(cors.Default())
	r.LoadHTMLGlob("templates/*")

	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", nil)
	})
	r.POST("/api/recognize", handleRecognize)

	fmt.Println("服务启动: http://0.0.0.0:" + serverEndpoint)
	if err := r.Run(":" + serverEndpoint); err != nil {
		fmt.Printf("启动失败: %v\n", err)
	}
}

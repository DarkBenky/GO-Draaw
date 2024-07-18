// TODO [High]: Use Vector 32


// FIXME[High]: UI elements merge into one layer and then draw on the screen
// : Implement the blur function [DONE]
// TODO [High]: Implement the increase contrast function
// TODO [High]: Implement the increase brightness function
// TODO [High]: Implement the decrease brightness function
// [High]: Implement the edge detection function [DONE]
// TODO [High]: Implement the decrease contrast function
// TODO [High]: Implement the sharpen function
// TODO [High]: Implement the eraser tool
// TODO [High]: Implement the fill tool
// TODO [High]: Implement the line tool

package main

import (
	"bufio"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

func LoadOBJ(filename string) (object, error) {
	var obj object

	file, err := os.Open(filename)
	if err != nil {
		return obj, err
	}
	defer file.Close()

	var vertices []Vector
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)

		if len(fields) == 0 || strings.HasPrefix(line, "#") {
			continue // Skip empty lines and comments
		}

		switch fields[0] {
		case "v":
			// Vertex definition
			x, _ := strconv.ParseFloat(fields[1], 64)
			y, _ := strconv.ParseFloat(fields[2], 64)
			z, _ := strconv.ParseFloat(fields[3], 64)
			vertices = append(vertices, Vector{x, y, z})

		case "f":
			// Face (triangle) definition
			if len(fields) < 4 {
				continue // Skip lines without enough vertices
			}

			var v1, v2, v3 Vector
			v1Index, _ := strconv.Atoi(fields[1])
			v2Index, _ := strconv.Atoi(fields[2])
			v3Index, _ := strconv.Atoi(fields[3])

			if v1Index > 0 && v1Index <= len(vertices) {
				v1 = vertices[v1Index-1]
			}
			if v2Index > 0 && v2Index <= len(vertices) {
				v2 = vertices[v2Index-1]
			}
			if v3Index > 0 && v3Index <= len(vertices) {
				v3 = vertices[v3Index-1]
			}

			// Add triangle to object
			obj.triangles = append(obj.triangles, NewTriangle(v1, v2, v3, color.RGBA{255, 255, 255, 255}, 0.0))

			// Other cases can be added as needed (e.g., parsing materials, normals, etc.)
		}
	}

	if err := scanner.Err(); err != nil {
		return obj, err
	}

	// Calculate bounding box for the object
	obj.CalculateBoundingBox()

	return obj, nil
}

type Vector struct {
	x, y, z float64
}

type Vector32 struct {
	x, y, z float32
}

func (v Vector) Add(v2 Vector) Vector {
	return Vector{v.x + v2.x, v.y + v2.y, v.z + v2.z}
}

func (v Vector32) Add(v2 Vector32) Vector32 {
	return Vector32{v.x + v2.x, v.y + v2.y, v.z + v2.z}
}


func (v Vector) Sub(v2 Vector) Vector {
	return Vector{v.x - v2.x, v.y - v2.y, v.z - v2.z}
}

func (v Vector) Mul(scalar float64) Vector {
	return Vector{v.x * scalar, v.y * scalar, v.z * scalar}
}

func (v Vector) Dot(v2 Vector) float64 {
	return v.x*v2.x + v.y*v2.y + v.z*v2.z
}

func (v Vector) Cross(v2 Vector) Vector {
	return Vector{v.y*v2.z - v.z*v2.y, v.z*v2.x - v.x*v2.z, v.x*v2.y - v.y*v2.x}
}

func (v Vector) Normalize() Vector {
	magnitude := math.Sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
	if magnitude == 0 {
		return Vector{0, 0, 0}
	}
	return Vector{v.x / magnitude, v.y / magnitude, v.z / magnitude}
}

type Ray struct {
	origin, direction Vector
}

type Triangle struct {
	v1, v2, v3  Vector
	color       color.RGBA
	BoundingBox [2]Vector
	reflection  float64
}

func CreateCube(center Vector, size float64, color color.RGBA, refection float64) []Triangle {
	halfSize := size / 2

	vertices := [8]Vector{
		{center.x - halfSize, center.y - halfSize, center.z - halfSize},
		{center.x + halfSize, center.y - halfSize, center.z - halfSize},
		{center.x + halfSize, center.y + halfSize, center.z - halfSize},
		{center.x - halfSize, center.y + halfSize, center.z - halfSize},
		{center.x - halfSize, center.y - halfSize, center.z + halfSize},
		{center.x + halfSize, center.y - halfSize, center.z + halfSize},
		{center.x + halfSize, center.y + halfSize, center.z + halfSize},
		{center.x - halfSize, center.y + halfSize, center.z + halfSize},
	}

	return []Triangle{
		NewTriangle(vertices[0], vertices[1], vertices[2], color, refection), // Front face
		NewTriangle(vertices[0], vertices[2], vertices[3], color, refection),

		NewTriangle(vertices[4], vertices[5], vertices[6], color, refection), // Back face
		NewTriangle(vertices[4], vertices[6], vertices[7], color, refection),

		NewTriangle(vertices[0], vertices[1], vertices[5], color, refection), // Bottom face
		NewTriangle(vertices[0], vertices[5], vertices[4], color, refection),

		NewTriangle(vertices[2], vertices[3], vertices[7], color, refection), // Top face
		NewTriangle(vertices[2], vertices[7], vertices[6], color, refection),

		NewTriangle(vertices[1], vertices[2], vertices[6], color, refection), // Right face
		NewTriangle(vertices[1], vertices[6], vertices[5], color, refection),

		NewTriangle(vertices[0], vertices[3], vertices[7], color, refection), // Left face
		NewTriangle(vertices[0], vertices[7], vertices[4], color, refection),
	}
}

func (triangle *Triangle) CalculateBoundingBox() {
	triangle.BoundingBox[0] = Vector{math.Min(triangle.v1.x, math.Min(triangle.v2.x, triangle.v3.x)), math.Min(triangle.v1.y, math.Min(triangle.v2.y, triangle.v3.y)), math.Min(triangle.v1.z, math.Min(triangle.v2.z, triangle.v3.z))}
	triangle.BoundingBox[1] = Vector{math.Max(triangle.v1.x, math.Max(triangle.v2.x, triangle.v3.x)), math.Max(triangle.v1.y, math.Max(triangle.v2.y, triangle.v3.y)), math.Max(triangle.v1.z, math.Max(triangle.v2.z, triangle.v3.z))}
}

func NewTriangle(v1, v2, v3 Vector, color color.RGBA, reflection float64) Triangle {
	triangle := Triangle{v1: v1, v2: v2, v3: v3, color: color, reflection: reflection}
	triangle.CalculateBoundingBox()
	return triangle
}

func (triangle *Triangle) IntersectBoundingBox(ray Ray) bool {
	tMin := (triangle.BoundingBox[0].x - ray.origin.x) / ray.direction.x
	tMax := (triangle.BoundingBox[1].x - ray.origin.x) / ray.direction.x

	if tMin > tMax {
		tMin, tMax = tMax, tMin
	}

	tyMin := (triangle.BoundingBox[0].y - ray.origin.y) / ray.direction.y
	tyMax := (triangle.BoundingBox[1].y - ray.origin.y) / ray.direction.y

	if tyMin > tyMax {
		tyMin, tyMax = tyMax, tyMin
	}

	if (tMin > tyMax) || (tyMin > tMax) {
		return false
	}

	if tyMin > tMin {
		tMin = tyMin
	}

	if tyMax < tMax {
		tMax = tyMax
	}

	tzMin := (triangle.BoundingBox[0].z - ray.origin.z) / ray.direction.z
	tzMax := (triangle.BoundingBox[1].z - ray.origin.z) / ray.direction.z

	if tzMin > tzMax {
		tzMin, tzMax = tzMax, tzMin
	}

	if (tMin > tzMax) || (tzMin > tMax) {
		return false
	}

	// if tzMin > tMin {
	// 	tMin = tzMin
	// }

	if tzMax < tMax {
		tMax = tzMax
	}

	return tMax > 0
}

type Intersection struct {
	PointOfIntersection Vector
	Color               color.RGBA
	Normal              Vector
	Direction           Vector
	Distance            float64
	reflection          float64
}

type Light struct {
	Position  Vector
	Color     color.RGBA
	intensity float64
}

func (light *Light) CalculateLighting(intersection Intersection, triangles []Triangle) color.RGBA {
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	shadowRay := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: lightDir}

	// Check if the point is in shadow
	inShadow := false
	for _, triangle := range triangles {
		if _, intersect := shadowRay.IntersectTriangle(triangle); intersect {
			inShadow = true
			break
		}
	}

	// Ambient light contribution
	ambientFactor := 0.3 // Adjust ambient factor as needed
	ambientColor := color.RGBA{
		uint8(float64(intersection.Color.R) * ambientFactor),
		uint8(float64(intersection.Color.G) * ambientFactor),
		uint8(float64(intersection.Color.B) * ambientFactor),
		intersection.Color.A,
	}

	if inShadow {
		// If in shadow, return ambient color
		return ambientColor
	}

	// Calculate diffuse lighting

	lightIntensity := light.intensity * math.Max(0.0, lightDir.Dot(intersection.Normal))
	finalColor := color.RGBA{
		clampUint8(float64(ambientColor.R) + lightIntensity*float64(intersection.Color.R)),
		clampUint8(float64(ambientColor.G) + lightIntensity*float64(intersection.Color.G)),
		clampUint8(float64(ambientColor.B) + lightIntensity*float64(intersection.Color.B)),
		ambientColor.A,
	}

	// R := clampUint8(float64(ambientColor.R) + lightIntensity*float64(intersection.Color.R))
	// G := clampUint8(float64(ambientColor.G) + lightIntensity*float64(intersection.Color.G))
	// B := clampUint8(float64(ambientColor.B) + lightIntensity*float64(intersection.Color.B))

	return finalColor
}

// Helper function to clamp a float64 value to uint8 range
func clampUint8(value float64) uint8 {
	if value < 0 {
		return 0
	}
	if value > 255 {
		return 255
	}
	return uint8(value)
}

func (ray *Ray) IntersectTriangle(triangle Triangle) (Intersection, bool) {
	if !triangle.IntersectBoundingBox(*ray) {
		return Intersection{}, false
	}

	// Möller–Trumbore intersection algorithm
	edge1 := triangle.v2.Sub(triangle.v1)
	edge2 := triangle.v3.Sub(triangle.v1)
	h := ray.direction.Cross(edge2)
	a := edge1.Dot(h)
	if a > -0.00001 && a < 0.00001 {
		return Intersection{}, false
	}
	f := 1.0 / a
	s := ray.origin.Sub(triangle.v1)
	u := f * s.Dot(h)
	if u < 0.0 || u > 1.0 {
		return Intersection{}, false
	}
	q := s.Cross(edge1)
	v := f * ray.direction.Dot(q)
	if v < 0.0 || u+v > 1.0 {
		return Intersection{}, false
	}
	t := f * edge2.Dot(q)
	if t > 0.00001 {
		point := ray.origin.Add(ray.direction.Mul(t))
		normal := edge1.Cross(edge2).Normalize()
		distance := t // The distance should be the parameter t

		return Intersection{PointOfIntersection: point, Color: triangle.color, Normal: normal, Direction: ray.direction, Distance: distance, reflection: triangle.reflection}, true
	}
	return Intersection{}, false
}

const workerCount = 8

type Camera struct {
	Position  Vector
	Direction Vector
}

type Pixel struct {
	x, y  int
	color color.RGBA
}

func (intersection *Intersection) Scatter(samples int, light Light, o *[]object) color.RGBA {
	var finalColor color.RGBA64

	//  Calculate the direct reflection
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	reflectDir := intersection.Normal.Mul(2 * lightDir.Dot(intersection.Normal)).Sub(lightDir).Normalize()
	reflectRay := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: reflectDir}

	// Find intersection with scene
	reflectedIntersection := Intersection{Distance: math.MaxFloat64}
	for _, object := range *o {
		if object.IntersectBoundingBox(reflectRay) {
			for _, triangle := range object.ConvertToTriangles() {
				tempIntersection, intersect := reflectRay.IntersectTriangle(triangle)
				if intersect && tempIntersection.Distance < reflectedIntersection.Distance {
					reflectedIntersection = tempIntersection
				}
			}
		}
	}

	for i := 0; i < samples; i++ {
		// Generate random direction in hemisphere around the normal (cosine-weighted distribution)
		u := rand.Float64()
		v := rand.Float64()
		r := math.Sqrt(u)
		theta := 2 * math.Pi * v

		// Construct local coordinate system
		w := intersection.Normal
		var uVec, vVec Vector
		if math.Abs(w.x) > 0.1 {
			uVec = Vector{0.0, 1.0, 0.0}
		} else {
			uVec = Vector{1.0, 0.0, 0.0}
		}
		uVec = uVec.Cross(w).Normalize()
		vVec = w.Cross(uVec)

		// Calculate direction in local coordinates
		directionLocal := uVec.Mul(r * math.Cos(theta)).Add(vVec.Mul(r * math.Sin(theta))).Add(w.Mul(math.Sqrt(1 - u)))

		// Transform direction to global coordinates
		direction := directionLocal.Normalize()

		// Create ray starting from intersection point
		ray := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: direction}

		// Find intersection with scene
		scatteredIntersection := Intersection{Distance: math.MaxFloat64}
		// for _, triangle := range triangles {
		// 	tempIntersection, intersect := ray.IntersectTriangle(triangle)
		// 	if intersect && tempIntersection.Distance < scatteredIntersection.Distance {
		// 		scatteredIntersection = tempIntersection
		// 	}
		// }

		for _, object := range *o {
			if object.IntersectBoundingBox(ray) {
				for _, triangle := range object.ConvertToTriangles() {
					tempIntersection, intersect := ray.IntersectTriangle(triangle)
					if intersect && tempIntersection.Distance < scatteredIntersection.Distance {
						scatteredIntersection = tempIntersection
					}
				}
			}
		}

		if scatteredIntersection.Distance != math.MaxFloat64 {
			finalColor.R += uint16(scatteredIntersection.Color.R)
			finalColor.G += uint16(scatteredIntersection.Color.G)
			finalColor.B += uint16(scatteredIntersection.Color.B)
			finalColor.A += uint16(scatteredIntersection.Color.A)
		}
	}

	// Average color by dividing by samples
	if samples > 0 {
		finalColor.R /= uint16(samples)
		finalColor.G /= uint16(samples)
		finalColor.B /= uint16(samples)
		finalColor.A = 255 // Ensure alpha remains fully opaque
	}

	//  mix the direct reflection with the scattered color
	rationScatterToDirect := 1 - intersection.reflection

	return color.RGBA{
		clampUint8(float64(finalColor.R)*rationScatterToDirect + float64(reflectedIntersection.Color.R)*intersection.reflection),
		clampUint8(float64(finalColor.G)*rationScatterToDirect + float64(reflectedIntersection.Color.G)*intersection.reflection),
		clampUint8(float64(finalColor.B)*rationScatterToDirect + float64(reflectedIntersection.Color.B)*intersection.reflection),
		uint8(finalColor.A)}
}

type object struct {
	triangles   []Triangle
	BoundingBox [2]Vector
}

func CreateObject(triangles []Triangle) *object {
	object := &object{
		triangles: triangles,
		BoundingBox: [2]Vector{
			Vector{math.MaxFloat64, math.MaxFloat64, math.MaxFloat64},
			Vector{-math.MaxFloat64, -math.MaxFloat64, -math.MaxFloat64},
		},
	}
	object.CalculateBoundingBox()
	return object
}

func (object *object) CalculateBoundingBox() {
	for _, triangle := range object.triangles {
		// Update minimum coordinates (BoundingBox[0])
		object.BoundingBox[0].x = math.Min(object.BoundingBox[0].x, triangle.BoundingBox[0].x)
		object.BoundingBox[0].y = math.Min(object.BoundingBox[0].y, triangle.BoundingBox[0].y)
		object.BoundingBox[0].z = math.Min(object.BoundingBox[0].z, triangle.BoundingBox[0].z)

		// Update maximum coordinates (BoundingBox[1])
		object.BoundingBox[1].x = math.Max(object.BoundingBox[1].x, triangle.BoundingBox[1].x)
		object.BoundingBox[1].y = math.Max(object.BoundingBox[1].y, triangle.BoundingBox[1].y)
		object.BoundingBox[1].z = math.Max(object.BoundingBox[1].z, triangle.BoundingBox[1].z)
	}
}

func (object *object) IntersectBoundingBox(ray Ray) bool {
	tMin := (object.BoundingBox[0].x - ray.origin.x) / ray.direction.x
	tMax := (object.BoundingBox[1].x - ray.origin.x) / ray.direction.x

	if tMin > tMax {
		tMin, tMax = tMax, tMin
	}

	tYMin := (object.BoundingBox[0].y - ray.origin.y) / ray.direction.y
	tYMax := (object.BoundingBox[1].y - ray.origin.y) / ray.direction.y

	if tYMin > tYMax {
		tYMin, tYMax = tYMax, tYMin
	}

	if tMin > tYMax || tYMin > tMax {
		return false
	}

	if tYMin > tMin {
		tMin = tYMin
	}

	if tYMax < tMax {
		tMax = tYMax
	}

	tZMin := (object.BoundingBox[0].z - ray.origin.z) / ray.direction.z
	tZMax := (object.BoundingBox[1].z - ray.origin.z) / ray.direction.z

	if tZMin > tZMax {
		tZMin, tZMax = tZMax, tZMin
	}

	if tMin > tZMax || tZMin > tMax {
		return false
	}

	if tZMin > tMin {
		tMin = tZMin
	}

	if tZMax < tMax {
		tMax = tZMax
	}

	return tMin < math.Inf(1) && tMax > 0
}

func (object *object) ConvertToTriangles() []Triangle {
	triangles := []Triangle{}
	triangles = append(triangles, object.triangles...)
	return triangles
}

func DrawRays(object *[]object, Triangles []Triangle, screen *ebiten.Image, camera Camera, FOV float64, light Light, scaling int, samples int) {
	aspectRatio := float64(screenWidth) / float64(screenHeight)
	scale := math.Tan(FOV * 0.5 * math.Pi / 180.0) // Convert FOV to radians

	pixelChan := make(chan Pixel, screenWidth*screenHeight)
	rowsPerWorker := screenHeight / workerCount

	var wg sync.WaitGroup
	wg.Add(workerCount)

	for i := 0; i < workerCount; i++ {
		go func(startY int) {
			defer wg.Done()
			for width := 0; width < screenWidth; width += scaling {
				for height := startY; height < startY+rowsPerWorker && height < screenHeight; height += scaling {
					// Normalize screen coordinates to [-1, 1]
					pixelNDCX := (float64(width) + 0.5) / float64(screenWidth)
					pixelNDCY := (float64(height) + 0.5) / float64(screenHeight)

					// Screen space coordinates [-1, 1]
					pixelScreenX := 2.0*pixelNDCX - 1.0
					pixelScreenY := 1.0 - 2.0*pixelNDCY

					// Apply aspect ratio and FOV scale
					pixelCameraX := pixelScreenX * aspectRatio * scale
					pixelCameraY := pixelScreenY * scale

					// Set ray direction
					rayDirection := Vector{pixelCameraX, pixelCameraY, -1}.Normalize()
					ray := Ray{origin: camera.Position, direction: rayDirection}

					intersection := Intersection{Distance: math.MaxFloat64}
					for _, object := range *object {
						if object.IntersectBoundingBox(ray) {
							for _, triangle := range object.ConvertToTriangles() {
								tempIntersection, intersect := ray.IntersectTriangle(triangle)
								if intersect && tempIntersection.Distance < intersection.Distance {
									intersection = tempIntersection
								}
							}
						}
					}
					// for _, triangle := range Triangles {
					// 	tempIntersection, intersect := ray.IntersectTriangle(triangle)
					// 	if intersect && tempIntersection.Distance < intersection.Distance {
					// 		intersection = tempIntersection
					// 	}
					// }
					if intersection.Distance != math.MaxFloat64 {
						// Calculate the final color with lighting
						Scatter := intersection.Scatter(samples, light, object)
						light := light.CalculateLighting(intersection, Triangles)

						c := color.RGBA{
							G: clampUint8(float64(Scatter.G) + float64(light.G)),
							R: clampUint8(float64(Scatter.R) + float64(light.R)),
							B: clampUint8(float64(Scatter.B) + float64(light.B)),
							A: 255,
						}

						pixelChan <- Pixel{x: width, y: height, color: c}
					}
				}
			}
		}(i * rowsPerWorker)
	}

	go func() {
		wg.Wait()
		close(pixelChan)
	}()

	if scaling == 1 {
		for pixel := range pixelChan {
			screen.Set(pixel.x, pixel.y, pixel.color)
		}
	} else {
		for pixel := range pixelChan {
			vector.DrawFilledRect(screen, float32(pixel.x), float32(pixel.y), float32(scaling), float32(scaling), pixel.color, true)
		}
	}
}

const screenWidth = 1280
const screenHeight = 720
const numLayers = 5

type Button struct {
	x, y, w, h  int
	text        string
	background  color.RGBA
	valueReturn int
}

func (b *Button) Draw(screen *ebiten.Image) {
	vector.DrawFilledRect(screen, float32(b.x), float32(b.y), float32(b.w), float32(b.h), b.background, true)
	ebitenutil.DebugPrintAt(screen, b.text, b.x, b.y)
}

type Slider struct {
	x, y, w, h int
	value      float32
	color      color.RGBA
}

func (s *Slider) Update(mouseX, mouseY int, mousePressed bool) {
	if mousePressed && mouseX >= s.x && mouseX <= s.x+s.w && mouseY >= s.y && mouseY <= s.y+s.h {
		s.value = float32(mouseX-s.x) / float32(s.w)
	}
}

func (s *Slider) Draw(screen *ebiten.Image) {
	vector.DrawFilledRect(screen, float32(s.x), float32(s.y), float32(s.w), float32(s.h), color.Gray{0x80}, true)
	vector.DrawFilledRect(screen, float32(s.x)+float32(s.w)*s.value-2, float32(s.y), 4, float32(s.h), s.color, true)
}

func (g *Game) isKeyReleased(key ebiten.Key) bool {
	return g.prevKeyStates[key] && !g.currKeyStates[key]
}

func (g *Game) updateKeyStates() {
	for k := range g.currKeyStates {
		g.prevKeyStates[k] = g.currKeyStates[k]
	}
	for k := range g.currKeyStates {
		g.currKeyStates[k] = ebiten.IsKeyPressed(k)
	}
}

func drawIntLayer(layer *Layer, x float32, y float32, g *Game) {
	if g.brushType == 0 {
		// Draw a circle
		vector.DrawFilledCircle(layer.image, x, y, g.brushSize, g.color, true)
	} else {
		// Draw a square
		vector.DrawFilledRect(layer.image, x, y, g.brushSize, g.brushSize, g.color, true)
	}
}

const maxMaskSize = 100

type Mask struct {
	mask            [maxMaskSize][maxMaskSize]color.RGBA
	currentMaskSize int
}

func (mask *Mask) createMask(layer *Layer, x int, y int, g *Game) {
	brushSize := g.brushSize
	mask.currentMaskSize = int(brushSize)
	if brushSize > maxMaskSize {
		brushSize = maxMaskSize
	}
	for i := 0; i < int(brushSize); i++ {
		for j := 0; j < int(brushSize); j++ {
			newX := x - int(brushSize)/2 + i
			newY := y - int(brushSize)/2 + j
			if newX >= 0 && newX < screenWidth && newY >= 0 && newY < screenHeight {
				mask.mask[i][j] = layer.image.At(newX, newY).(color.RGBA)
			}
		}
	}
}

type ColorInt16 struct {
	R uint16
	G uint16
	B uint16
	A uint16
}

func (layer *Layer) edgeLayer(x int, y int, game *Game, mask *Mask, threshold int) {
	mask.createMask(layer, x, y, game)
	temp := make([][]color.RGBA, mask.currentMaskSize)
	edgeColor := game.color

	sobelX := [3][3]int{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	}
	sobelY := [3][3]int{
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1},
	}

	for i := range temp {
		temp[i] = make([]color.RGBA, mask.currentMaskSize)
	}

	for h := 0; h < mask.currentMaskSize; h++ {
		for w := 0; w < mask.currentMaskSize; w++ {
			var gxR, gxG, gxB, gyR, gyG, gyB int

			for i := -1; i <= 1; i++ {
				for j := -1; j <= 1; j++ {
					nh, nw := h+i, w+j
					if nh >= 0 && nh < mask.currentMaskSize && nw >= 0 && nw < mask.currentMaskSize {
						gxR += int(mask.mask[nh][nw].R) * sobelX[i+1][j+1]
						gxG += int(mask.mask[nh][nw].G) * sobelX[i+1][j+1]
						gxB += int(mask.mask[nh][nw].B) * sobelX[i+1][j+1]
						gyR += int(mask.mask[nh][nw].R) * sobelY[i+1][j+1]
						gyG += int(mask.mask[nh][nw].G) * sobelY[i+1][j+1]
						gyB += int(mask.mask[nh][nw].B) * sobelY[i+1][j+1]
					}
				}
			}

			// Calculate the gradient magnitude
			gradientR := math.Sqrt(float64(gxR*gxR + gyR*gyR))
			gradientG := math.Sqrt(float64(gxG*gxG + gyG*gyG))
			gradientB := math.Sqrt(float64(gxB*gxB + gyB*gyB))

			// Average the gradient magnitude
			gradient := (gradientR + gradientG + gradientB) / 3

			// Apply the threshold
			if gradient > float64(threshold) {
				temp[h][w] = edgeColor
			} else {
				temp[h][w] = mask.mask[h][w]
			}
		}
		// Copy the result back to the layer's image
		for h := 1; h < mask.currentMaskSize-1; h++ {
			for w := 1; w < mask.currentMaskSize-1; w++ {
				newX := x - mask.currentMaskSize/2 + h
				newY := y - mask.currentMaskSize/2 + w
				if newX >= 0 && newX < screenWidth && newY >= 0 && newY < screenHeight {
					layer.image.Set(newX, newY, temp[h][w])
				}
			}
		}
	}
}

func (layer *Layer) blurLayer(x int, y int, game *Game, mask *Mask) {
	mask.createMask(layer, x, y, game)
	temp := make([][]color.RGBA, mask.currentMaskSize)
	for i := range temp {
		temp[i] = make([]color.RGBA, mask.currentMaskSize)
	}

	kernelSize := int(game.brushSize/25) + 1

	for h := 0; h < mask.currentMaskSize; h++ {
		for w := 0; w < mask.currentMaskSize; w++ {
			average := ColorInt16{}
			count := uint16(0)
			for i := -kernelSize; i <= kernelSize; i++ {
				for j := -kernelSize; j <= kernelSize; j++ {
					nh, nw := h+i, w+j
					if nh >= 0 && nh < mask.currentMaskSize && nw >= 0 && nw < mask.currentMaskSize {
						average.R += uint16(mask.mask[nh][nw].R)
						average.G += uint16(mask.mask[nh][nw].G)
						average.B += uint16(mask.mask[nh][nw].B)
						average.A += uint16(mask.mask[nh][nw].A)
						count++
					}
				}
			}
			average.R /= count
			average.G /= count
			average.B /= count
			average.A /= count
			temp[h][w] = color.RGBA{uint8(average.R), uint8(average.G), uint8(average.B), uint8(average.A)}
		}
	}

	for h := 0; h < mask.currentMaskSize; h++ {
		for w := 0; w < mask.currentMaskSize; w++ {
			newX := x - mask.currentMaskSize/2 + h
			newY := y - mask.currentMaskSize/2 + w
			if newX >= 0 && newX < screenWidth && newY >= 0 && newY < screenHeight {
				layer.image.Set(newX, newY, temp[h][w])
			}
		}
	}
}

func (g *Game) Update() error {
	// Update the current key states
	g.updateKeyStates()

	if g.isKeyReleased(ebiten.KeyTab) {
		g.move = !g.move
		println("Move:", g.move)
	}

	if g.isKeyReleased(ebiten.KeyW) {
		if g.currentLayer < numLayers-1 {
			g.currentLayer++
		}
	}

	if g.isKeyReleased(ebiten.KeyS) {
		if g.currentLayer > 0 {
			g.currentLayer--
		}
	}

	if g.isKeyReleased(ebiten.KeyC) {
		for i := 0; i < numLayers; i++ {
			g.layers.layers[i].image.Clear()
		}
	}

	if g.isKeyReleased(ebiten.KeyQ) {
		g.brushType = (g.brushType + 1) % 2 // Toggle between 0 and 1
	}

	if g.isKeyReleased(ebiten.KeyCapsLock) {
		if g.accumulate {
			g.samples = 4
		} else {
			g.samples = 32 // Default samples
		}
		g.accumulate = !g.accumulate
		println("Accumulate:", g.accumulate)
		println("Samples:", g.samples)
	}

	if g.isKeyReleased(ebiten.KeyR) {
		if g.render {
			g.scaleFactor = 1
		} else {
			g.scaleFactor = 8
		}
		g.render = !g.render
		println("Render:", g.render)
		println("Scale Factor:", g.scaleFactor)
	}

	// increase the brush size by scrolling
	_, scrollY := ebiten.Wheel()
	if scrollY > 0 {
		g.brushSize += 1
	} else if scrollY < 0 {
		g.brushSize -= 1
	}

	mouseX, mouseY := ebiten.CursorPosition()
	mousePressed := ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft)

	for _, slider := range g.sliders {
		slider.Update(mouseX, mouseY, mousePressed)
	}

	g.color = color.RGBA{
		uint8(g.sliders[0].value * 255),
		uint8(g.sliders[1].value * 255),
		uint8(g.sliders[2].value * 255),
		uint8(g.sliders[3].value * 255),
	}

	// Get Button Clicks
	if mousePressed {
		for _, button := range g.Buttons {
			if mouseX >= button.x && mouseX <= button.x+button.w && mouseY >= button.y && mouseY <= button.y+button.h {
				g.currentTool = button.valueReturn
			}
		}
	}

	// Draw on the current layer
	if mousePressed && g.currentTool == 0 {
		drawIntLayer(&g.layers.layers[g.currentLayer], float32(mouseX), float32(mouseY), g)
	} else if mousePressed && g.currentTool == 1 {
		g.layers.layers[g.currentLayer].blurLayer(mouseX, mouseY, g, &g.mask)
	} else if mousePressed && g.currentTool == 2 {
		if g.updateFreq%10 == 0 {
			g.layers.layers[g.currentLayer].edgeLayer(mouseX, mouseY, g, &g.mask, 50)
		}
		g.updateFreq++
	}

	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Draw the layers
	for i := 0; i < numLayers; i++ {
		screen.DrawImage(g.layers.layers[i].image, nil)
	}

	// Draw sliders
	for _, slider := range g.sliders {
		slider.Draw(screen)
	}
	// Draw current color preview
	vector.DrawFilledRect(screen, 50, 470, 200, 200, g.color, true)

	// Draw buttons
	for _, button := range g.Buttons {
		button.Draw(screen)
	}

	// fmt.Printf("FPS: %v", ebiten.ActualFPS())

	fps := ebiten.ActualFPS()

	ebitenutil.DebugPrint(screen, fmt.Sprintf("FPS: %v", fps))
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Current Layer: %v", g.currentLayer), 0, 20)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Brush Size: %v", g.brushSize), 0, 40)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Brush Type: %v", g.brushType), 0, 60)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Current Tool: %v", g.currentTool), 0, 80)

	if g.render {
		if fps < 24 {
			g.scaleFactor += 0.05
		} else {
			g.scaleFactor -= 0.05
		}
	}

	// capture the previous screen

	DrawRays(g.objects, g.triangles, screen, g.camera, g.FOV, g.light, int(g.scaleFactor), g.samples)

	// get the mouse position
	if g.move {
		mouseX, mouseY := ebiten.CursorPosition()
		g.light.Position = Vector{float64(mouseX), float64(mouseY), 200}
		g.camera.Position = Vector{float64(mouseX), float64(mouseY), 550}
	}

	if g.accumulate {
		g.avgScreen = averageImages(screen, g.avgScreen)
		screen.DrawImage(g.avgScreen, nil)
	}
}

func averageImages(img1, img2 *ebiten.Image) *ebiten.Image {
	bounds := img1.Bounds()
	result := ebiten.NewImage(bounds.Dx(), bounds.Dy())
	resultPix := make([]byte, bounds.Dx()*bounds.Dy()*4)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r1, g1, b1, a1 := img1.At(x, y).RGBA()
			r2, g2, b2, a2 := img2.At(x, y).RGBA()

			r := (r1 + r2) / 2
			g := (g1 + g2) / 2
			b := (b1 + b2) / 2
			a := (a1 + a2) / 2

			// result.Set(x, y, color.RGBA{
			// 	uint8(r >> 8),
			// 	uint8(g >> 8),
			// 	uint8(b >> 8),
			// 	uint8(a >> 8),
			// })

			i := (y*bounds.Dx() + x) * 4
			resultPix[i] = uint8(r >> 8)
			resultPix[i+1] = uint8(g >> 8)
			resultPix[i+2] = uint8(b >> 8)
			resultPix[i+3] = uint8(a >> 8)
		}
	}

	result.WritePixels(resultPix)
	return result
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 1280, 720
}

type Layer struct {
	image *ebiten.Image
}

type Layers struct {
	layers [numLayers]Layer
}

type Game struct {
	layers        *Layers
	currentLayer  int
	brushSize     float32
	brushType     int
	prevKeyStates map[ebiten.Key]bool
	currKeyStates map[ebiten.Key]bool
	color         color.RGBA
	sliders       [4]*Slider
	Buttons       []*Button
	currentTool   int
	mask          Mask
	camera        Camera
	FOV           float64
	triangles     []Triangle
	light         Light
	scaleFactor   float64
	objects       *[]object
	render        bool
	move          bool
	avgScreen     *ebiten.Image
	accumulate    bool
	samples       int
	updateFreq    int
}

func main() {

	testVec := Vector{0.1, 0.2, 0.3}

	start := time.Now()
	v := Vector{1, 2, 3}
	for i := 0; i < 10000000000; i++ {
		v = v.Add(testVec)
	}
	fmt.Println(v, time.Since(start), "Vector Classic Calculation")


	v32 := Vector32{1, 2, 3}
	testVec32 := Vector32{0.1, 0.2, 0.3}

	start = time.Now()
	for i := 0; i < 10000000000; i++ {
		v32 = v32.Add(testVec32)
	}
	fmt.Println(v, time.Since(start), "Vector SIMD Calculation")

	numCPU := runtime.NumCPU()
	fmt.Println("Number of CPUs:", numCPU)

	runtime.GOMAXPROCS(workerCount)

	ebiten.SetVsyncEnabled(false)

	ebiten.SetTPS(60)

	layers := Layers{
		layers: [numLayers]Layer{
			{image: ebiten.NewImage(screenWidth, screenHeight)},
			{image: ebiten.NewImage(screenWidth, screenHeight)},
			{image: ebiten.NewImage(screenWidth, screenHeight)},
			{image: ebiten.NewImage(screenWidth, screenHeight)},
			{image: ebiten.NewImage(screenWidth, screenHeight)},
		},
	}
	buttons := []*Button{
		{x: 200, y: 200, w: 100, h: 50, text: "Button 1", background: color.RGBA{255, 0, 0, 0}, valueReturn: 0},
		{x: 200, y: 300, w: 100, h: 50, text: "Button 2", background: color.RGBA{0, 255, 0, 255}, valueReturn: 1},
		{x: 200, y: 400, w: 100, h: 50, text: "Button 3", background: color.RGBA{0, 0, 255, 255}, valueReturn: 2},
	}

	// Set the background color of each layer
	for i := 0; i < numLayers; i++ {
		layers.layers[i].image.Fill(color.Transparent)
	}

	cube1 := CreateCube(Vector{100, 100, 0}, 200, color.RGBA{255, 0, 0, 255}, 0.5)
	cubeObj := CreateObject(cube1)
	cube2 := CreateCube(Vector{200, 0, 200}, 200, color.RGBA{0, 255, 0, 255}, 0.1)
	cubeObj1 := CreateObject(cube2)
	cube3 := CreateCube(Vector{500, 200, 100}, 200, color.RGBA{0, 0, 255, 255}, 0.9)
	cubeObj2 := CreateObject(cube3)
	cube4 := CreateCube(Vector{300, 300, 300}, 200, color.RGBA{255, 255, 255, 255}, 1.0)
	cubeObj3 := CreateObject(cube4)
	cube5 := CreateCube(Vector{500, 100, -200}, 200, color.RGBA{32, 32, 32, 255}, 1.0)
	cubeObj4 := CreateObject(cube5)

	obj, err := LoadOBJ("test.obj")
	if err != nil {
		log.Fatal(err)
	}

	objects := []object{*cubeObj, *cubeObj1, *cubeObj2, *cubeObj3, *cubeObj4, obj}

	t := []Triangle{}
	for _, object := range objects {
		t = append(t, object.ConvertToTriangles()...)
	}

	game := &Game{
		layers:        &layers,
		currentLayer:  0,
		brushSize:     10,
		color:         color.RGBA{255, 0, 0, 255},
		brushType:     0,
		prevKeyStates: make(map[ebiten.Key]bool),
		currKeyStates: make(map[ebiten.Key]bool),
		sliders: [4]*Slider{
			{x: 50, y: 350, w: 200, h: 20, color: color.RGBA{255, 0, 0, 255}},
			{x: 50, y: 380, w: 200, h: 20, color: color.RGBA{0, 255, 0, 255}},
			{x: 50, y: 410, w: 200, h: 20, color: color.RGBA{0, 0, 255, 255}},
			{x: 50, y: 440, w: 200, h: 20, color: color.RGBA{0, 0, 0, 255}}, // Alpha slider
		},
		currentTool: 0,
		Buttons:     buttons,
		mask:        Mask{},
		camera:      Camera{Position: Vector{0, 200, 0}, Direction: Vector{0, 0, -1}},
		FOV:         90.0,
		triangles:   t,
		light:       Light{Position: Vector{0, 400, 200}, Color: color.RGBA{255, 255, 255, 255}, intensity: 1},
		scaleFactor: 16,
		objects:     &objects,
		render:      false,
		move:        true,
		avgScreen:   ebiten.NewImage(screenWidth, screenHeight),
		accumulate:  false,
		samples:     2,
	}

	keys := []ebiten.Key{ebiten.KeyW, ebiten.KeyS, ebiten.KeyQ, ebiten.KeyR, ebiten.KeyTab, ebiten.KeyCapsLock, ebiten.KeyC}
	for _, key := range keys {
		game.prevKeyStates[key] = false
		game.currKeyStates[key] = false
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Ebiten Game")
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}

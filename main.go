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
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"runtime"
	"sync"

	"os"
	"runtime/pprof"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

func startCPUProfile() {
	f, err := os.Create("cpu.prof")
	if err != nil {
		fmt.Println("could not create CPU profile: ", err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		fmt.Println("could not start CPU profile: ", err)
	}
}

func stopCPUProfile() {
	pprof.StopCPUProfile()
}

type Vector struct {
	x, y, z float64
}

type VectorArray []Vector

type Ray struct {
	origin, direction Vector
}

type RayArray []Ray

func (rays *RayArray)RaysDirNormalize() *RayArray{
	for i := range *rays{
		(*rays)[i].direction = (*rays)[i].direction.Normalize()
	}
	return rays
}

func (v Vector) Add(v2 Vector) Vector {
	return Vector{v.x + v2.x, v.y + v2.y, v.z + v2.z}
}

func (v *VectorArray)Add(v2 *VectorArray) *VectorArray{
	for i := range *v{
		(*v)[i].x += (*v2)[i].x
		(*v)[i].y += (*v2)[i].y
		(*v)[i].z += (*v2)[i].z
	}
	return v
}

func (v Vector) Sub(v2 Vector) Vector {
	return Vector{v.x - v2.x, v.y - v2.y, v.z - v2.z}
}

func (v *VectorArray)Sub(v2 *VectorArray) *VectorArray{
	for i := range *v{
		(*v)[i].x -= (*v2)[i].x
		(*v)[i].y -= (*v2)[i].y
		(*v)[i].z -= (*v2)[i].z
	}
	return v
}

func (v Vector) Dot(v2 Vector) float64 {
	return v.x*v2.x + v.y*v2.y + v.z*v2.z
}

func (v *VectorArray)Dot(v2 *VectorArray) *VectorArray{
	for i := range *v{
		(*v)[i].x *= (*v2)[i].x
		(*v)[i].y *= (*v2)[i].y
		(*v)[i].z *= (*v2)[i].z
	}
	return v
}

func (v Vector) Cross(v2 Vector) Vector {
	return Vector{v.y*v2.z - v.z*v2.y, v.z*v2.x - v.x*v2.z, v.x*v2.y - v.y*v2.x}
}

func (v *VectorArray)Cross(v2 *VectorArray) *VectorArray{
	for i := range *v{
		(*v)[i].x = (*v)[i].y*(*v2)[i].z - (*v)[i].z*(*v2)[i].y
		(*v)[i].y = (*v)[i].z*(*v2)[i].x - (*v)[i].x*(*v2)[i].z
		(*v)[i].z = (*v)[i].x*(*v2)[i].y - (*v)[i].y*(*v2)[i].x
	}
	return v
}

func (v *Vector) Scale(s float64) Vector {
	return Vector{v.x * s, v.y * s, v.z * s}
}

func (v *VectorArray)Scale(s float64) *VectorArray{
	for i := range *v{
		(*v)[i].x *= s
		(*v)[i].y *= s
		(*v)[i].z *= s
	}
	return v
}

func (v *Vector) Magnitude() float64 {
	return math.Sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
}

func (v *VectorArray)Magnitude() *VectorArray{
	for i := range *v{
		(*v)[i].x = math.Sqrt((*v)[i].x*(*v)[i].x + (*v)[i].y*(*v)[i].y + (*v)[i].z*(*v)[i].z)
	}
	return v
}

func (v Vector) Normalize() Vector {
	mag := v.Magnitude()
	v.x /= mag
	v.y /= mag
	v.z /= mag
	return v
}

func (v *VectorArray)Normalize() *VectorArray{
	for i := range *v{
		mag := (*v)[i].Magnitude()
		(*v)[i].x /= mag
		(*v)[i].y /= mag
		(*v)[i].z /= mag
	}
	return v
}

type Object interface {
	Intersect(ray Ray) float64
	Normal(v Vector) Vector
}

type Polygon struct {
	vertices    [3]Vector // Triangle
	boundingBox [2]Vector
	color       color.RGBA
}

func NewPolygon(v1, v2, v3 Vector, color color.RGBA) Polygon {
	minX := math.Min(v1.x, math.Min(v2.x, v3.x))
	minY := math.Min(v1.y, math.Min(v2.y, v3.y))
	minZ := math.Min(v1.z, math.Min(v2.z, v3.z))
	maxX := math.Max(v1.x, math.Max(v2.x, v3.x))
	maxY := math.Max(v1.y, math.Max(v2.y, v3.y))
	maxZ := math.Max(v1.z, math.Max(v2.z, v3.z))
	return Polygon{
		vertices:    [3]Vector{v1, v2, v3},
		boundingBox: [2]Vector{Vector{minX, minY, minZ}, Vector{maxX, maxY, maxZ}},
		color:       color,
	}
}

func (p Polygon) Normal(v Vector) Vector {
	edge1 := p.vertices[1].Sub(p.vertices[0])
	edge2 := p.vertices[2].Sub(p.vertices[0])
	return edge1.Cross(edge2).Normalize()
}

func (p Polygon) IsPointInPolygon(point Vector) bool {
	normal := p.Normal(Vector{})
	for i := 0; i < len(p.vertices); i++ {
		v0 := p.vertices[i]
		v1 := p.vertices[(i+1)%len(p.vertices)]
		edge := v1.Sub(v0)
		vp := point.Sub(v0)
		c := edge.Cross(vp)
		if normal.Dot(c) < 0 {
			return false
		}
	}
	return true
}

type Intersection struct {
	distance          float64
	normal            Vector
	color             color.RGBA
	reflection        Vector
	intersectionPoint Point
}

type Point struct {
	x, y, z float64
}

func (p Polygon) Intersect(ray Ray) Intersection {

	// Default intersection result for no intersection
	noIntersection := Intersection{distance: -1.0}

	// start := time.Now() // Start the timer

	// Check if the ray intersects the bounding box
	if !p.IsRayIntersectingBoundingBox(ray) {
		return noIntersection
	}

	// fmt.Println("Bounding Box Intersection Time: ", time.Since(start))
	// start = time.Now()

	normal := p.Normal(Vector{})
	planePoint := p.vertices[0]

	denominator := normal.Dot(ray.direction)
	if math.Abs(denominator) < 1e-6 {
		return noIntersection // Ray is parallel to the polygon plane
	}

	t := planePoint.Sub(ray.origin).Dot(normal) / denominator
	if t < 0 {
		return noIntersection // Polygon is behind the ray
	}

	P := ray.origin.Add(ray.direction.Scale(t))
	if !p.IsPointInPolygon(P) {
		return noIntersection // Intersection point is outside the polygon
	}

	// fmt.Println("Intersection Time: ", time.Since(start))
	// start = time.Now()

	reflection := Vector{}
	if math.Abs(ray.direction.Dot(normal)) > 1e-6 {
		reflection = ray.direction.Sub(normal.Scale(2 * ray.direction.Dot(normal))).Normalize()
	} else {
		// Handle case where ray direction and normal are nearly parallel
		reflection = ray.direction // Fallback to the original direction
	}

	// fmt.Println("Reflection Time: ", time.Since(start))

	return Intersection{
		distance:          t,
		normal:            normal,
		color:             p.color,
		reflection:        reflection,
		intersectionPoint: Point(P),
	}
}

func (p Polygon) IsRayIntersectingBoundingBox(ray Ray) bool {
	// Calculate the inverse of the ray direction for use in the slab method.
	invDir := Vector{1 / ray.direction.x, 1 / ray.direction.y, 1 / ray.direction.z}

	// Calculate tmin and tmax for the x-axis, which represent the intersection
	// distances to the bounding box planes in the x direction.
	tmin := (p.boundingBox[0].x - ray.origin.x) * invDir.x
	tmax := (p.boundingBox[1].x - ray.origin.x) * invDir.x

	// Swap tmin and tmax if necessary to ensure tmin <= tmax.
	if tmin > tmax {
		tmin, tmax = tmax, tmin
	}

	// Calculate tymin and tymax for the y-axis.
	tymin := (p.boundingBox[0].y - ray.origin.y) * invDir.y
	tymax := (p.boundingBox[1].y - ray.origin.y) * invDir.y

	// Swap tymin and tymax if necessary.
	if tymin > tymax {
		tymin, tymax = tymax, tymin
	}

	// Check for overlap in the x and y slabs. If there's no overlap,
	// the ray does not intersect the bounding box.
	if (tmin > tymax) || (tymin > tmax) {
		return false
	}

	// Update tmin and tmax to ensure they represent the intersection
	// distances for both x and y slabs.
	if tymin > tmin {
		tmin = tymin
	}
	if tymax < tmax {
		tmax = tymax
	}

	// Calculate tzmin and tzmax for the z-axis.
	tzmin := (p.boundingBox[0].z - ray.origin.z) * invDir.z
	tzmax := (p.boundingBox[1].z - ray.origin.z) * invDir.z

	// Swap tzmin and tzmax if necessary.
	if tzmin > tzmax {
		tzmin, tzmax = tzmax, tzmin
	}

	// Check for overlap in the x, y, and z slabs. If there's no overlap,
	// the ray does not intersect the bounding box.
	if (tmin > tzmax) || (tzmin > tmax) {
		return false
	}

	// If we pass all checks, the ray intersects the bounding box.
	return true
}

type Mesh struct {
	polygons    []Polygon
	boundingBox [2]Vector
}

func NewMesh(polygons []Polygon) Mesh {
	minX := math.Inf(1)
	minY := math.Inf(1)
	minZ := math.Inf(1)
	maxX := math.Inf(-1)
	maxY := math.Inf(-1)
	maxZ := math.Inf(-1)
	for _, polygon := range polygons {
		for _, vertex := range polygon.vertices {
			minX = math.Min(minX, vertex.x)
			minY = math.Min(minY, vertex.y)
			minZ = math.Min(minZ, vertex.z)
			maxX = math.Max(maxX, vertex.x)
			maxY = math.Max(maxY, vertex.y)
			maxZ = math.Max(maxZ, vertex.z)
		}
	}
	return Mesh{
		polygons:    polygons,
		boundingBox: [2]Vector{Vector{minX, minY, minZ}, Vector{maxX, maxY, maxZ}},
	}
}

func (m Mesh) Intersect(ray Ray) Intersection {
	closestIntersection := Intersection{}
	closestIntersection.distance = -1.0
	for _, polygon := range m.polygons {
		intersection := polygon.Intersect(ray)
		if intersection.distance > 0 && closestIntersection.distance < intersection.distance {
			closestIntersection = intersection
		}
	}
	return closestIntersection
}

func DrawMesh(ray Ray, m Mesh, screen *ebiten.Image, fov float64) {
	// Calculate the field of view (FOV) in radians
	fovX := fov * math.Pi / 180.0 // Convert degrees to radians
	fovY := float64(screenHeight) / float64(screenWidth) * fovX

	// Iterate over the screen pixels
	for y := 0; y < screenHeight; y++ {
		for x := 0; x < screenWidth; x++ {
			// Calculate normalized device coordinates (NDC) in range [-1, 1]
			ndcX := (2.0 * float64(x) / float64(screenWidth)) - 1.0
			ndcY := 1.0 - (2.0 * float64(y) / float64(screenHeight))

			// Calculate direction vector based on FOV and NDC
			ray.direction.x = ndcX * math.Tan(fovX/2.0)
			ray.direction.y = ndcY * math.Tan(fovY/2.0)

			// Normalize the direction vector
			ray.direction = ray.direction.Normalize()

			// Check if the ray intersects the bounding box of the entire mesh
			if !isRayIntersectingMeshBoundingBox(ray, m) {
				continue
			}

			// Intersect the ray with the mesh
			intersection := m.Intersect(ray)

			// Set the color of the pixel on the screen if there's an intersection
			if intersection.distance != -1 {
				screen.Set(x, y, intersection.color)
				// fmt.Println(intersection.color)
			}
		}
	}
}

const numberOfThreads = 4

func DrawMeshMultiProcessing(ray Ray, mashes []Mesh, screen *ebiten.Image, fov float64) {
	// Calculate the field of view (FOV) in radians
	fovX := fov * math.Pi / 180.0 // Convert degrees to radians
	fovY := float64(screenHeight) / float64(screenWidth) * fovX

	BlocksOfScreen := make([][]Ray, numberOfThreads)
	IndicesOfScreen := make([][]int, numberOfThreads)

	for i := range BlocksOfScreen {
		BlocksOfScreen[i] = make([]Ray, 0, screenHeight*screenWidth/numberOfThreads)
		IndicesOfScreen[i] = make([]int, 0, screenHeight*screenWidth/numberOfThreads)
	}

	// Iterate over the screen pixels
	for y := 0; y < screenHeight; y++ {
		for x := 0; x < screenWidth; x++ {
			// Calculate normalized device coordinates (NDC) in range [-1, 1]
			ndcX := (2.0 * float64(x) / float64(screenWidth)) - 1.0
			ndcY := 1.0 - (2.0 * float64(y) / float64(screenHeight))

			// Calculate direction vector based on FOV and NDC
			ray.direction.x = ndcX * math.Tan(fovX/2.0)
			ray.direction.y = ndcY * math.Tan(fovY/2.0)

			// Normalize the direction vector
			ray.direction = ray.direction.Normalize()

			// Split the rays into blocks
			threadIndex := (y*screenWidth + x) % numberOfThreads
			BlocksOfScreen[threadIndex] = append(BlocksOfScreen[threadIndex], ray)
			IndicesOfScreen[threadIndex] = append(IndicesOfScreen[threadIndex], y*screenWidth+x)
		}
	}

	// Channel to communicate pixel updates to the main thread
	pixelUpdates := make(chan PixelUpdate, screenWidth*screenHeight)

	var wg sync.WaitGroup

	// Do raycasting in parallel
	for i := 0; i < numberOfThreads; i++ {
		wg.Add(1)
		go func(block []Ray, indices []int) {
			defer wg.Done()
			for j, ray := range block {
				for _, m := range mashes {
					// Check if the ray intersects the bounding box of the entire mesh
					if !isRayIntersectingMeshBoundingBox(ray, m) {
						continue
					}

					// Intersect the ray with the mesh
					intersection := m.Intersect(ray)

					// Calculate the x and y coordinates
					index := indices[j]
					x := index % (screenWidth)
					y := index / (screenWidth)

					// Send the pixel update to the main thread if there's an intersection
					if intersection.distance != -1 {
						pixelUpdates <- PixelUpdate{x: x, y: y, color: intersection.color}
					}
				}
			}
		}(BlocksOfScreen[i], IndicesOfScreen[i])
	}

	go func() {
		wg.Wait()
		close(pixelUpdates)
	}()

	// Apply pixel updates to the screen in the main thread
	for update := range pixelUpdates {
		screen.Set(update.x, update.y, update.color)
	}
}


func DrawMeshTrueMpProcessing(ray Ray, mashes []Mesh, screen *ebiten.Image, fov float64) {
	// Calculate the field of view (FOV) in radians
	fovX := fov * math.Pi / 180.0 // Convert degrees to radians
	fovY := float64(screenHeight) / float64(screenWidth) * fovX

	BlocksOfScreen := make([]RayArray, numberOfThreads)
	IndicesOfScreen := make([][]int, numberOfThreads)

	for i := range BlocksOfScreen {
		BlocksOfScreen[i] = make(RayArray, 0, screenHeight*screenWidth/numberOfThreads)
		IndicesOfScreen[i] = make([]int, 0, screenHeight*screenWidth/numberOfThreads)
	}

	// Iterate over the screen pixels
	for y := 0; y < screenHeight; y++ {
		for x := 0; x < screenWidth; x++ {
			// Calculate normalized device coordinates (NDC) in range [-1, 1]
			ndcX := (2.0 * float64(x) / float64(screenWidth)) - 1.0
			ndcY := 1.0 - (2.0 * float64(y) / float64(screenHeight))

			// Calculate direction vector based on FOV and NDC
			ray.direction.x = ndcX * math.Tan(fovX/2.0)
			ray.direction.y = ndcY * math.Tan(fovY/2.0)

			// Split the rays into blocks
			threadIndex := (y*screenWidth + x) % numberOfThreads
			BlocksOfScreen[threadIndex] = append(BlocksOfScreen[threadIndex], ray)
			IndicesOfScreen[threadIndex] = append(IndicesOfScreen[threadIndex], y*screenWidth+x)
		}
	}

	for i := range BlocksOfScreen{
		BlocksOfScreen[i].RaysDirNormalize()
	}

	// Normalize the direction vector
	

	// Channel to communicate pixel updates to the main thread
	pixelUpdates := make(chan PixelUpdate, screenWidth*screenHeight)

	var wg sync.WaitGroup

	// Do raycasting in parallel
	for i := 0; i < numberOfThreads; i++ {
		wg.Add(1)
		go func(block []Ray, indices []int) {
			defer wg.Done()
			for j, ray := range block {
				for _, m := range mashes {
					// Check if the ray intersects the bounding box of the entire mesh
					if !isRayIntersectingMeshBoundingBox(ray, m) {
						continue
					}

					// Intersect the ray with the mesh
					intersection := m.Intersect(ray)

					// Calculate the x and y coordinates
					index := indices[j]
					x := index % (screenWidth)
					y := index / (screenWidth)

					// Send the pixel update to the main thread if there's an intersection
					if intersection.distance != -1 {
						pixelUpdates <- PixelUpdate{x: x, y: y, color: intersection.color}
					}
				}
			}
		}(BlocksOfScreen[i], IndicesOfScreen[i])
	}

	go func() {
		wg.Wait()
		close(pixelUpdates)
	}()

	// Apply pixel updates to the screen in the main thread
	for update := range pixelUpdates {
		screen.Set(update.x, update.y, update.color)
	}
}


type PixelUpdate struct {
	x, y  int
	color color.Color
}

func isRayIntersectingMeshBoundingBox(ray Ray, m Mesh) bool {
	// Calculate the inverse of the ray direction for use in the slab method.
	invDir := Vector{1 / ray.direction.x, 1 / ray.direction.y, 1 / ray.direction.z}

	// Calculate tmin and tmax for the x-axis, which represent the intersection
	// distances to the bounding box planes in the x direction.
	tmin := (m.boundingBox[0].x - ray.origin.x) * invDir.x
	tmax := (m.boundingBox[1].x - ray.origin.x) * invDir.x

	// Swap tmin and tmax if necessary to ensure tmin <= tmax.
	if tmin > tmax {
		tmin, tmax = tmax, tmin
	}

	// Calculate tymin and tymax for the y-axis.
	tymin := (m.boundingBox[0].y - ray.origin.y) * invDir.y
	tymax := (m.boundingBox[1].y - ray.origin.y) * invDir.y

	// Swap tymin and tymax if necessary.
	if tymin > tymax {
		tymin, tymax = tymax, tymin
	}

	// Check for overlap in the x and y slabs. If there's no overlap,
	// the ray does not intersect the bounding box.
	if (tmin > tymax) || (tymin > tmax) {
		return false
	}

	// Update tmin and tmax to ensure they represent the intersection
	// distances for both x and y slabs.
	if tymin > tmin {
		tmin = tymin
	}
	if tymax < tmax {
		tmax = tymax
	}

	// Calculate tzmin and tzmax for the z-axis.
	tzmin := (m.boundingBox[0].z - ray.origin.z) * invDir.z
	tzmax := (m.boundingBox[1].z - ray.origin.z) * invDir.z

	// Swap tzmin and tzmax if necessary.
	if tzmin > tzmax {
		tzmin, tzmax = tzmax, tzmin
	}

	// Check for overlap in the x, y, and z slabs. If there's no overlap,
	// the ray does not intersect the bounding box.
	if (tmin > tzmax) || (tzmin > tmax) {
		return false
	}

	// If we pass all checks, the ray intersects the bounding box.
	return true
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
	// Check if the key was previously pressed and is now released
	return g.prevKeyStates[key] && !g.currKeyStates[key]
}

func (g *Game) updateKeyStates() {
	// Update the current key states
	for k := range g.currKeyStates {
		g.currKeyStates[k] = ebiten.IsKeyPressed(k)
	}
}

func (g *Game) storePrevKeyStates() {
	// Store the current key states as the previous key states
	for k, v := range g.currKeyStates {
		g.prevKeyStates[k] = v
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

	if g.isKeyReleased(ebiten.KeyQ) {
		g.brushType = (g.brushType + 1) % 2 // Toggle between 0 and 1
	}

	if g.isKeyReleased(ebiten.KeyR) {
		g.color = color.RGBA{uint8(rand.Intn(255)), uint8(rand.Intn(255)), uint8(rand.Intn(255)), 255}
	}

	// Store the current key states as the previous states for the next update
	g.storePrevKeyStates()

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
		g.layers.layers[g.currentLayer].edgeLayer(mouseX, mouseY, g, &g.mask, 50)
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
	ebitenutil.DebugPrint(screen, fmt.Sprintf("FPS: %v", ebiten.ActualFPS()))
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Current Layer: %v", g.currentLayer), 0, 20)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Brush Size: %v", g.brushSize), 0, 40)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Brush Type: %v", g.brushType), 0, 60)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Current Tool: %v", g.currentTool), 0, 80)

	// DrawMesh(g.ray, g.meshes[0], screen, 60)
	// DrawMesh(g.ray, g.meshes[1], screen, 60)

	// DrawMeshMultiProcessing(g.ray, g.meshes[0], screen, 60)
	DrawMeshMultiProcessing(g.ray, g.meshes, screen, 60)
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
	meshes        []Mesh
	ray           Ray
}

func main() {

	startCPUProfile()
	defer stopCPUProfile()

	numCPU := runtime.NumCPU()
	fmt.Println("Number of CPUs:", numCPU)

	runtime.GOMAXPROCS(numberOfThreads)

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

	p1 := NewPolygon(Vector{0, 0, 0}, Vector{0, 1, 0}, Vector{1, 1, 1}, color.RGBA{255, 0, 0, 255})
	p2 := NewPolygon(Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{1, 1, 1}, color.RGBA{0, 255, 0, 255})
	p3 := NewPolygon(Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{1, 1, 0}, color.RGBA{0, 0, 255, 255})
	p4 := NewPolygon(Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{1, 1, 1}, color.RGBA{255, 0, 0, 255})

	mesh := NewMesh([]Polygon{p1, p2, p3, p4})

	p5 := NewPolygon(Vector{1, 0, 0}, Vector{0, 1, 0}, Vector{1, 1, 10}, color.RGBA{255, 0, 0, 255})
	p6 := NewPolygon(Vector{0, 1, 10}, Vector{1, 0, 0}, Vector{1, 1, 1}, color.RGBA{0, 255, 0, 255})
	p7 := NewPolygon(Vector{5, 1, 0}, Vector{1, 0, 0}, Vector{1, 1, 0}, color.RGBA{0, 0, 255, 255})
	p8 := NewPolygon(Vector{1, 0, 0}, Vector{1, 0, 0}, Vector{1, 1, 1}, color.RGBA{255, 0, 0, 255})

	mesh2 := NewMesh([]Polygon{p5, p6, p7, p8})

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
		meshes:      []Mesh{mesh, mesh2},
		ray:         Ray{origin: Vector{0, 0, -3}, direction: Vector{0, 0, 0.5}},
	}

	keys := []ebiten.Key{ebiten.KeyW, ebiten.KeyS, ebiten.KeyQ, ebiten.KeyR}
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

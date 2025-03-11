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
// TODO [High]: Implement Projection rendering

// TODO: Merge the changes from the previous commit
// TODO: Implement the Projection Rendering

package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"image/draw"

	"github.com/chewxy/math32"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/shirou/gopsutil/cpu"
	"github.com/shirou/gopsutil/mem"

	"github.com/labstack/echo/v4"
)

const screenWidth = 800
const screenHeight = 608
const rowSize = screenHeight / numCPU

var FOV = float32(45)

var ScreenSpaceCoordinates [screenWidth][screenHeight]Vector

const maxDepth = 16
const NumNodes = (1 << (maxDepth + 1)) - 1
const numCPU = 16

const Benchmark = true

var AverageFrameRate float64 = 0.0
var MinFrameRate float64 = math.MaxFloat64
var MaxFrameRate float64 = 0.0
var FPS []float64

type Material struct {
	name            string
	color           ColorFloat32
	specular        float32
	reflection      float32
	directToScatter float32
	Metallic        float32
	Roughness       float32
}

func LoadMTL(filename string) (map[string]Material, error) {
	materials := make(map[string]Material)

	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var currentMaterial string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)

		if len(fields) == 0 || strings.HasPrefix(line, "#") {
			continue // Skip empty lines and comments
		}

		switch fields[0] {
		case "newmtl":
			if len(fields) < 2 {
				continue // Skip malformed material names
			}
			currentMaterial = fields[1]
			materials[currentMaterial] = Material{name: currentMaterial}

		case "Kd": // Diffuse color
			if len(fields) < 4 || currentMaterial == "" {
				continue // Skip if no material is currently being defined
			}
			r, err1 := strconv.ParseFloat(fields[1], 32)
			g, err2 := strconv.ParseFloat(fields[2], 32)
			b, err3 := strconv.ParseFloat(fields[3], 32)
			if err1 != nil || err2 != nil || err3 != nil {
				continue // Skip malformed color definitions
			}
			mat := materials[currentMaterial]
			mat.color = ColorFloat32{
				R: float32(r * 255),
				G: float32(g * 255),
				B: float32(b * 255),
				A: 255,
			}
			materials[currentMaterial] = mat
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return materials, nil
}

func LoadOBJ(filename string) (object, error) {
	var obj object
	materials := make(map[string]Material)

	file, err := os.Open(filename)
	if err != nil {
		return obj, err
	}
	defer file.Close()

	var vertices []Vector
	var currentMaterial string
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)

		if len(fields) == 0 || strings.HasPrefix(line, "#") {
			continue // Skip empty lines and comments
		}

		switch fields[0] {
		case "v":
			if len(fields) < 4 {
				continue // Ensure there are enough fields for vertex coordinates
			}
			x, err1 := strconv.ParseFloat(fields[1], 64)
			y, err2 := strconv.ParseFloat(fields[2], 64)
			z, err3 := strconv.ParseFloat(fields[3], 64)
			if err1 != nil || err2 != nil || err3 != nil {
				continue // Skip malformed vertex lines
			}
			vertices = append(vertices, Vector{float32(x), float32(y), float32(z)})

		case "usemtl":
			if len(fields) < 2 {
				continue // Skip malformed usemtl lines
			}
			currentMaterial = fields[1]

		case "mtllib":
			if len(fields) < 2 {
				continue // Skip malformed mtllib lines
			}
			mtlFilename := fields[1]
			loadedMaterials, err := LoadMTL(mtlFilename)
			if err != nil {
				return obj, err
			}
			// Merge the loaded materials into the materials map
			for name, mat := range loadedMaterials {
				materials[name] = mat
			}

		case "f":
			if len(fields) < 4 {
				continue // Skip lines without enough vertices
			}

			var indices []int
			for i := 1; i < len(fields); i++ {
				parts := strings.Split(fields[i], "/")
				if len(parts) == 0 {
					continue // Skip malformed face definitions
				}
				index, err := strconv.ParseInt(parts[0], 10, 64)
				if err != nil {
					continue // Skip malformed face definitions
				}
				if index < 0 {
					index = int64(len(vertices)) + index + 1
				}
				if index <= 0 || index > int64(len(vertices)) {
					continue // Skip invalid indices
				}
				indices = append(indices, int(index)-1)
			}

			if len(indices) >= 3 {
				for i := 1; i < len(indices)-1; i++ {
					triangle := TriangleSimple{
						v1:              vertices[indices[0]],
						v2:              vertices[indices[i]],
						v3:              vertices[indices[i+1]],
						reflection:      rand.Float32(),
						Roughness:       rand.Float32(),
						specular:        rand.Float32(),
						directToScatter: rand.Float32(),
						Metallic:        rand.Float32(),
					}

					// Apply the current material color if available
					if mat, exists := materials[currentMaterial]; exists {
						triangle.color = mat.color
						triangle.color.A = 255 // Ensure alpha is fully opaque
					} else {
						// triangle.color = color.RGBA{255, 125, 0, 255} // Default color
						triangle.color = ColorFloat32{255, 125, 0, 255} // Default color
					}

					obj.triangles = append(obj.triangles, triangle)
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return obj, err
	}

	obj.CalculateBoundingBox()
	obj.CalculateNormals()

	return obj, nil
}

type Vector struct {
	x, y, z float32
}

func (v Vector) Multiply(v2 Vector) Vector {
	return Vector{v.x * v2.x, v.y * v2.y, v.z * v2.z}
}

func (v Vector) Length() float32 {
	return float32(math.Sqrt(float64(v.x*v.x + v.y*v.y + v.z*v.z)))
}

func (v Vector) Add(v2 Vector) Vector {
	return Vector{v.x + v2.x, v.y + v2.y, v.z + v2.z}
}

func (v Vector) Sub(v2 Vector) Vector {
	return Vector{v.x - v2.x, v.y - v2.y, v.z - v2.z}
}

func (v Vector) Mul(scalar float32) Vector {
	return Vector{v.x * scalar, v.y * scalar, v.z * scalar}
}

func (v Vector) Dot(v2 Vector) float32 {
	return v.x*v2.x + v.y*v2.y + v.z*v2.z
}

func (v Vector) Cross(v2 Vector) Vector {
	return Vector{v.y*v2.z - v.z*v2.y, v.z*v2.x - v.x*v2.z, v.x*v2.y - v.y*v2.x}
}

func (v Vector) Normalize() Vector {
	magnitude := float32(math.Sqrt(float64(v.x*v.x + v.y*v.y + v.z*v.z)))
	if magnitude == 0 {
		return Vector{0, 0, 0}
	}
	return Vector{v.x / magnitude, v.y / magnitude, v.z / magnitude}
}

func (v Vector) RotateX(angle float32) Vector {
	return Vector{
		x: v.x,
		y: v.y*math32.Cos(angle) - v.z*math32.Sin(angle),
		z: v.y*math32.Sin(angle) + v.z*math32.Cos(angle),
	}
}

func (v Vector) RotateY(angle float32) Vector {
	return Vector{
		x: v.x*math32.Cos(angle) + v.z*math32.Sin(angle),
		y: v.y,
		z: -v.x*math32.Sin(angle) + v.z*math32.Cos(angle),
	}
}

func (v Vector) RotateZ(angle float32) Vector {
	return Vector{
		x: v.x*math32.Cos(angle) - v.y*math32.Sin(angle),
		y: v.x*math32.Sin(angle) + v.y*math32.Cos(angle),
		z: v.z,
	}
}

func (v Vector) Rotate(angleX, angleY, angleZ float32) Vector {
	return v.RotateX(angleX).RotateY(angleY).RotateZ(angleZ)
}

func (v Vector) Reflect(normal Vector) Vector {
	return v.Sub(normal.Mul(2 * v.Dot(normal)))
}

type Ray struct {
	origin, direction Vector
}

//	type Triangle struct {
//		v1, v2, v3  Vector
//		color       color.RGBA
//		BoundingBox [2]Vector
//		Normal      Vector
//		reflection  float32
//		specular    float32
//	}

type ColorFloat32 struct {
	R, G, B, A float32
}

func (c ColorFloat32) Average(c1 ColorFloat32) ColorFloat32 {
	return ColorFloat32{
		R: (c.R + c1.R) / 2,
		G: (c.G + c1.G) / 2,
		B: (c.B + c1.B) / 2,
		A: (c.A + c1.A) / 2,
	}
}

func (c ColorFloat32) MulScalar(scalar float32) ColorFloat32 {
	return ColorFloat32{
		R: c.R * scalar,
		G: c.G * scalar,
		B: c.B * scalar,
		A: c.A,
	}
}

func (c ColorFloat32) Mul(c1 ColorFloat32) ColorFloat32 {
	return ColorFloat32{
		R: c.R * c1.R,
		G: c.G * c1.G,
		B: c.B * c1.B,
		A: c.A,
	}
}

func (c ColorFloat32) Add(c2 ColorFloat32) ColorFloat32 {
	return ColorFloat32{c.R + c2.R, c.G + c2.G, c.B + c2.B, c.A + c2.A}
}

type TriangleSimple struct {
	v1, v2, v3 Vector
	// color           color.RGBA
	color           ColorFloat32
	Normal          Vector
	reflection      float32
	directToScatter float32
	specular        float32
	Roughness       float32
	Metallic        float32
	id              uint8
}

type Texture struct {
	texture [128][128]ColorFloat32
	normals [128][128]Vector
	// add material properties like specular and reflection etc .
	reflection      float32
	directToScatter float32
	specular        float32
	Roughness       float32
	Metallic        float32
}

type TriangleBBOX struct {
	V1orBBoxMin, V2orBBoxMax, V3 Vector
	normal                       Vector
	id                           int32
}

type Intersect struct {
	PointOfIntersection Vector
	Distance            float32
	textureX            int16
	textureY            int16
}

func IntersectTriangle(ray Ray, V1 Vector, V2 Vector, V3 Vector) (bool, Intersect) {
	// Möller–Trumbore intersection algorithm
	edge1 := V2.Sub(V1)
	edge2 := V3.Sub(V1)
	h := ray.direction.Cross(edge2)
	a := edge1.Dot(h)
	if a > -0.00001 && a < 0.00001 {
		return false, Intersect{}
	}
	f := 1.0 / a
	s := ray.origin.Sub(V1)
	u := f * s.Dot(h)
	if u < 0.0 || u > 1.0 {
		return false, Intersect{}
	}
	q := s.Cross(edge1)
	v := f * ray.direction.Dot(q)
	if v < 0.0 || u+v > 1.0 {
		return false, Intersect{}
	}
	t := f * edge2.Dot(q)
	if t > 0.00001 {
		// Compute barycentric coordinates
		w := 1.0 - u - v

		// Sample the texture using barycentric coordinates
		texU := int16(w * 127) // Scale to [0,127] range
		texV := int16(v * 127)
		if texU < 0 {
			texU = 0
		} else if texU > 127 {
			texU = 127
		}
		if texV < 0 {
			texV = 0
		} else if texV > 127 {
			texV = 127
		}

		return true, Intersect{
			PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)),
			Distance:            t,
			textureX:            texU,
			textureY:            texV,
		}
	}
	return false, Intersect{}
}

func IntersectBoundingBox(ray Ray, BBoxMin, BBoxMax Vector) (hit bool, dist float32) {
	invDir := Vector{1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z}
	tx1 := (BBoxMin.x - ray.origin.x) * invDir.x
	tx2 := (BBoxMax.x - ray.origin.x) * invDir.x
	tmin := math32.Min(tx1, tx2)
	tmax := math32.Max(tx1, tx2)

	ty1 := (BBoxMin.y - ray.origin.y) * invDir.y
	ty2 := (BBoxMax.y - ray.origin.y) * invDir.y
	tmin = math32.Max(tmin, math32.Min(ty1, ty2))
	tmax = math32.Min(tmax, math32.Max(ty1, ty2))

	tz1 := (BBoxMin.z - ray.origin.z) * invDir.z
	tz2 := (BBoxMax.z - ray.origin.z) * invDir.z
	tmin = math32.Max(tmin, math32.Min(tz1, tz2))
	tmax = math32.Min(tmax, math32.Max(tz1, tz2))

	return tmax >= math32.Max(0.0, tmin), tmin
}

type BVHArray struct {
	triangles [NumNodes]TriangleBBOX
	textures  [128]Texture
}

func (bvh *BVHArray) IntersectBVH(ray Ray) (bool, Intersection) {
	const eps = 1e-7

	// Fixed size stack, avoid allocations
	var stack [maxDepth]int32
	stackPtr := 0
	stack[stackPtr] = 1 // root

	// Cache texture reference
	tex := &bvh.textures[0]

	// Precompute inverse direction with safety checks
	invDir := Vector{
		x: func() float32 {
			if math32.Abs(ray.direction.x) < eps {
				return 1.0 / eps
			}
			return 1.0 / ray.direction.x
		}(),
		y: func() float32 {
			if math32.Abs(ray.direction.y) < eps {
				return 1.0 / eps
			}
			return 1.0 / ray.direction.y
		}(),
		z: func() float32 {
			if math32.Abs(ray.direction.z) < eps {
				return 1.0 / eps
			}
			return 1.0 / ray.direction.z
		}(),
	}

	var (
		closestIntersection Intersection
		bestDistance        = float32(math32.MaxFloat32)
		hasHit              = false
	)

	// Direct pointer access to triangles array
	triPtr := unsafe.Pointer(&bvh.triangles[0])

	for stackPtr >= 0 {
		index := int(stack[stackPtr])
		stackPtr--

		// Get triangle using pointer arithmetic
		tri := (*TriangleBBOX)(unsafe.Pointer(uintptr(triPtr) + uintptr(index)*unsafe.Sizeof(TriangleBBOX{})))

		if tri.id != int32(-1) {
			if hit, intersect := IntersectTriangle(ray, tri.V1orBBoxMin, tri.V2orBBoxMax, tri.V3); hit {
				if intersect.Distance < bestDistance {
					bestDistance = intersect.Distance
					hasHit = true

					// Combine normals and create intersection
					// normal := tri.normal.Add(tex.normals[intersect.textureX][intersect.textureY])
					closestIntersection = Intersection{
						PointOfIntersection: intersect.PointOfIntersection,
						Color:               tex.texture[intersect.textureX][intersect.textureY],
						Normal:              tri.normal,
						Distance:            intersect.Distance,
						reflection:          tex.reflection,
						directToScatter:     tex.directToScatter,
						specular:            tex.specular,
						Roughness:           tex.Roughness,
						Metallic:            tex.Metallic,
					}
				}
			}
			continue
		}

		// Process internal node
		leftIdx := 2 * index
		rightIdx := leftIdx + 1

		// Check bounds to avoid array overflow
		// if rightIdx >= len(bvh.triangles) {
		// 	continue
		// }

		// Test both children's bounding boxes
		// if bvh.triangles[leftIdx].id != int32(0) {
		if leftHit, _ := IntersectBoundingBoxFast(ray,
			invDir,
			bvh.triangles[leftIdx].V2orBBoxMax,
			bvh.triangles[leftIdx].V1orBBoxMin); leftHit {
			stackPtr++
			stack[stackPtr] = int32(leftIdx)
		}
		// }

		// if bvh.triangles[rightIdx].id != int32(0) {
		if rightHit, _ := IntersectBoundingBoxFast(ray,
			invDir,
			bvh.triangles[rightIdx].V2orBBoxMax,
			bvh.triangles[rightIdx].V1orBBoxMin); rightHit {
			stackPtr++
			stack[stackPtr] = int32(rightIdx)
		}
		// }
	}

	return hasHit, closestIntersection
}

// Optimized box intersection test
func IntersectBoundingBoxFast(ray Ray, invDir Vector, BBoxMax, BBoxMin Vector) (bool, float32) {
	tmin := (BBoxMin.x - ray.origin.x) * invDir.x
	tmax := (BBoxMax.x - ray.origin.x) * invDir.x
	if tmin > tmax {
		tmin, tmax = tmax, tmin
	}

	tymin := (BBoxMin.y - ray.origin.y) * invDir.y
	tymax := (BBoxMax.y - ray.origin.y) * invDir.y
	if tymin > tymax {
		tymin, tymax = tymax, tymin
	}

	if tmin > tymax || tymin > tmax {
		return false, 0
	}

	if tymin > tmin {
		tmin = tymin
	}
	if tymax < tmax {
		tmax = tymax
	}

	tzmin := (BBoxMin.z - ray.origin.z) * invDir.z
	tzmax := (BBoxMax.z - ray.origin.z) * invDir.z
	if tzmin > tzmax {
		tzmin, tzmax = tzmax, tzmin
	}

	return tzmax >= tmin && tzmin <= tmax, tmin
}

// Optimized AABB intersection test using pre-calculated inverse ray direction
// func IntersectBoundingBoxFast(rayOrigin Vector, rayDirInv Vector, min, max Vector) (bool, float32) {
// 	t1 := (min.x - rayOrigin.x) * rayDirInv.x
// 	t2 := (max.x - rayOrigin.x) * rayDirInv.x
// 	tmin := math32.Min(t1, t2)
// 	tmax := math32.Max(t1, t2)

// 	t1 = (min.y - rayOrigin.y) * rayDirInv.y
// 	t2 = (max.y - rayOrigin.y) * rayDirInv.y
// 	tmin = math32.Max(tmin, math32.Min(t1, t2))
// 	tmax = math32.Min(tmax, math32.Max(t1, t2))

// 	t1 = (min.z - rayOrigin.z) * rayDirInv.z
// 	t2 = (max.z - rayOrigin.z) * rayDirInv.z
// 	tmin = math32.Max(tmin, math32.Min(t1, t2))
// 	tmax = math32.Min(tmax, math32.Max(t1, t2))

// 	return tmax >= tmin && tmax > 0, tmin
// }

// start with index 1
func (bvh *BVHNode) ConvertToArray(index int, bvhArr *BVHArray) {
	// if bvh == nil {
	// 	return
	// }

	if bvh.active {
		bvhArr.triangles[index] = TriangleBBOX{
			V1orBBoxMin: bvh.Triangles.v1,
			V2orBBoxMax: bvh.Triangles.v2,
			V3:          bvh.Triangles.v3,
			normal:      bvh.Triangles.Normal,
			id:          int32(1),
		}
	} else {
		bvhArr.triangles[index] = TriangleBBOX{
			V1orBBoxMin: bvh.BoundingBox[0],
			V2orBBoxMax: bvh.BoundingBox[1],
			id:          int32(-1),
		}
	}

	if bvh.Left != nil {
		bvh.Left.ConvertToArray(2*index, bvhArr)
	}
	if bvh.Right != nil {
		bvh.Right.ConvertToArray(2*index+1, bvhArr)
	}
}

type SphereSimple struct {
	center Vector
	radius float32
	color  color.RGBA
}

func Distance(v1, v2 Vector, radius float32) float32 {
	// Use vector subtraction and dot product instead of individual calculations
	diff := v1.Sub(v2)
	return diff.Length() - radius
}

// Add normal calculation for spheres
func calculateNormal(point, center Vector) Vector {
	return point.Sub(center).Normalize()
}

type RayMarchingBVH struct {
	BoundingBox [2]Vector
	Sphere      *SphereSimple
	Left, Right *RayMarchingBVH
	Active      bool
}

func calculateSphereSurfaceArea(bbox [2]Vector) float32 {
	dx := bbox[1].x - bbox[0].x
	dy := bbox[1].y - bbox[0].y
	dz := bbox[1].z - bbox[0].z
	return 2 * (dx*dy + dy*dz + dz*dx)
}

func calculateSphereBoundingBox(sphere SphereSimple) [2]Vector {
	return [2]Vector{
		{
			x: sphere.center.x - sphere.radius,
			y: sphere.center.y - sphere.radius,
			z: sphere.center.z - sphere.radius,
		},
		{
			x: sphere.center.x + sphere.radius,
			y: sphere.center.y + sphere.radius,
			z: sphere.center.z + sphere.radius,
		},
	}
}

func BuildBvhForSpheres(spheres []SphereSimple, maxDepth int) *RayMarchingBVH {
	if len(spheres) == 0 {
		return nil
	}

	// Calculate the overall bounding box
	boundingBox := [2]Vector{
		{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32},
		{-math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32},
	}

	for _, sphere := range spheres {
		sphereBBox := calculateSphereBoundingBox(sphere)
		boundingBox[0].x = float32(math.Min(float64(boundingBox[0].x), float64(sphereBBox[0].x)))
		boundingBox[0].y = float32(math.Min(float64(boundingBox[0].y), float64(sphereBBox[0].y)))
		boundingBox[0].z = float32(math.Min(float64(boundingBox[0].z), float64(sphereBBox[0].z)))

		boundingBox[1].x = float32(math.Max(float64(boundingBox[1].x), float64(sphereBBox[1].x)))
		boundingBox[1].y = float32(math.Max(float64(boundingBox[1].y), float64(sphereBBox[1].y)))
		boundingBox[1].z = float32(math.Max(float64(boundingBox[1].z), float64(sphereBBox[1].z)))
	}

	// If the node is a leaf or we've reached the maximum depth
	if len(spheres) <= 1 || maxDepth <= 0 {
		node := &RayMarchingBVH{
			BoundingBox: boundingBox,
			Sphere: &SphereSimple{
				center: spheres[0].center,
				radius: spheres[0].radius,
				color:  spheres[0].color,
			},
			Active: true,
		}
		return node
	}

	// Surface Area Heuristics (SAH) to find the best split
	bestCost := float32(math.MaxFloat32)
	bestSplit := -1
	bestAxis := 0

	for axis := 0; axis < 3; axis++ {
		// Sort spheres along the current axis
		switch axis {
		case 0:
			sort.Slice(spheres, func(i, j int) bool {
				return spheres[i].center.x < spheres[j].center.x
			})
		case 1:
			sort.Slice(spheres, func(i, j int) bool {
				return spheres[i].center.y < spheres[j].center.y
			})
		case 2:
			sort.Slice(spheres, func(i, j int) bool {
				return spheres[i].center.z < spheres[j].center.z
			})
		}

		// Compute surface area for all possible splits
		for i := 1; i < len(spheres); i++ {
			leftBBox := [2]Vector{
				{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32},
				{-math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32},
			}
			rightBBox := [2]Vector{
				{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32},
				{-math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32},
			}

			// Calculate left bounding box
			for j := 0; j < i; j++ {
				sphereBBox := calculateSphereBoundingBox(spheres[j])
				leftBBox[0].x = float32(math.Min(float64(leftBBox[0].x), float64(sphereBBox[0].x)))
				leftBBox[0].y = float32(math.Min(float64(leftBBox[0].y), float64(sphereBBox[0].y)))
				leftBBox[0].z = float32(math.Min(float64(leftBBox[0].z), float64(sphereBBox[0].z)))
				leftBBox[1].x = float32(math.Max(float64(leftBBox[1].x), float64(sphereBBox[1].x)))
				leftBBox[1].y = float32(math.Max(float64(leftBBox[1].y), float64(sphereBBox[1].y)))
				leftBBox[1].z = float32(math.Max(float64(leftBBox[1].z), float64(sphereBBox[1].z)))
			}

			// Calculate right bounding box
			for j := i; j < len(spheres); j++ {
				sphereBBox := calculateSphereBoundingBox(spheres[j])
				rightBBox[0].x = float32(math.Min(float64(rightBBox[0].x), float64(sphereBBox[0].x)))
				rightBBox[0].y = float32(math.Min(float64(rightBBox[0].y), float64(sphereBBox[0].y)))
				rightBBox[0].z = float32(math.Min(float64(rightBBox[0].z), float64(sphereBBox[0].z)))
				rightBBox[1].x = float32(math.Max(float64(rightBBox[1].x), float64(sphereBBox[1].x)))
				rightBBox[1].y = float32(math.Max(float64(rightBBox[1].y), float64(sphereBBox[1].y)))
				rightBBox[1].z = float32(math.Max(float64(rightBBox[1].z), float64(sphereBBox[1].z)))
			}

			// Calculate the SAH cost for this split
			cost := float32(i)*calculateSphereSurfaceArea(leftBBox) + float32(len(spheres)-i)*calculateSphereSurfaceArea(rightBBox)
			if cost < bestCost {
				bestCost = cost
				bestSplit = i
				bestAxis = axis
			}
		}
	}

	// Sort spheres along the best axis before splitting
	switch bestAxis {
	case 0:
		sort.Slice(spheres, func(i, j int) bool {
			return spheres[i].center.x < spheres[j].center.x
		})
	case 1:
		sort.Slice(spheres, func(i, j int) bool {
			return spheres[i].center.y < spheres[j].center.y
		})
	case 2:
		sort.Slice(spheres, func(i, j int) bool {
			return spheres[i].center.z < spheres[j].center.z
		})
	}

	// Create the BVH node with the best split
	node := &RayMarchingBVH{BoundingBox: boundingBox}
	node.Left = BuildBvhForSpheres(spheres[:bestSplit], maxDepth-1)
	node.Right = BuildBvhForSpheres(spheres[bestSplit:], maxDepth-1)

	return node
}

var sphereBVH = RayMarchingBVH{}

func IntersectBVH_RayMarching(bvh RayMarchingBVH, ray Ray) (bool, *SphereSimple) {
	if !BoundingBoxCollision(bvh.BoundingBox, ray) {
		return false, nil
	}

	if bvh.Sphere != nil {
		return true, bvh.Sphere
	}

	hitLeft, left := IntersectBVH_RayMarching(*bvh.Left, ray)
	hitRight, right := IntersectBVH_RayMarching(*bvh.Right, ray)

	if hitLeft && hitRight {
		if Distance(ray.origin, left.center, left.radius) < Distance(ray.origin, right.center, right.radius) {
			return true, left
		}
		return true, right
	}

	if hitLeft {
		return true, left
	}

	if hitRight {
		return true, right
	}

	return false, nil
}

// Improved sphere conversion with pre-allocated slice
func (obj object) ConvertToSquare(count int) []SphereSimple {
	spheres := make([]SphereSimple, 0, count)

	for i := 0; i < count; i += 1 {
		randIndex := rand.Intn(len(obj.triangles))
		R := clampUint8(obj.triangles[randIndex].color.R)
		G := clampUint8(obj.triangles[randIndex].color.G)
		B := clampUint8(obj.triangles[randIndex].color.B)
		spheres = append(spheres, SphereSimple{
			center: obj.triangles[randIndex].v1,
			radius: 2,
			// color:  obj.triangles[randIndex].color,
			color: color.RGBA{R, G, B, 255},
		})
	}
	return spheres
}

func RayMarchBvh(ray Ray, iterations int, light Light) (color.RGBA, float32) {
	const (
		EPSILON      = float32(0.0001)
		MAX_DISTANCE = float32(10000.0)
	)

	var (
		totalDistance float32
		closestSphere *SphereSimple
		currentPoint  Vector
	)

	for i := 0; i < iterations; i++ {
		currentPoint = ray.origin.Add(ray.direction.Mul(totalDistance))
		minDistance := MAX_DISTANCE

		hit, sphere := IntersectBVH_RayMarching(sphereBVH, ray)
		if hit {
			minDistance = Distance(currentPoint, sphere.center, sphere.radius)
			closestSphere = sphere
		}

		totalDistance += minDistance

		// Hit detection with early exit
		if minDistance < EPSILON {
			return calculateShading(currentPoint, *closestSphere, totalDistance, MAX_DISTANCE, light), totalDistance
		}

		// Miss detection with early exit
		if minDistance > MAX_DISTANCE || totalDistance > MAX_DISTANCE {
			return color.RGBA{0, 0, 0, 0}, totalDistance
		}
	}

	return calculateShading(currentPoint, *closestSphere, totalDistance, MAX_DISTANCE, light), totalDistance
}

func RayMarching(ray Ray, spheres []SphereSimple, iterations int, light Light) (color.RGBA, float32) {
	const (
		EPSILON      = float32(0.0001)
		MAX_DISTANCE = float32(10000.0)
	)

	var (
		totalDistance float32
		sphereColor   = color.RGBA{0, 0, 0, 255}
		closestSphere SphereSimple
		currentPoint  Vector
	)

	// Early exit if no spheres
	if len(spheres) == 0 {
		return sphereColor, totalDistance
	}

	for i := 0; i < iterations; i++ {
		currentPoint = ray.origin.Add(ray.direction.Mul(totalDistance))
		minDistance := MAX_DISTANCE

		// Find closest sphere
		for _, sphere := range spheres {
			if dist := Distance(currentPoint, sphere.center, sphere.radius); dist < minDistance {
				minDistance = dist
				closestSphere = sphere
			}
		}

		totalDistance += minDistance

		// Hit detection with early exit
		if minDistance < EPSILON {
			return calculateShading(currentPoint, closestSphere, totalDistance, MAX_DISTANCE, light), totalDistance
		}

		// Miss detection with early exit
		if minDistance > MAX_DISTANCE || totalDistance > MAX_DISTANCE {
			return color.RGBA{0, 0, 0, 0}, totalDistance
		}
	}

	return calculateShading(currentPoint, closestSphere, totalDistance, MAX_DISTANCE, light), totalDistance
}

// Helper function for color shading calculations
func calculateShading(point Vector, sphere SphereSimple, totalDistance, maxDistance float32, light Light) color.RGBA {

	// Calculate normal at intersection point
	normal := calculateNormal(point, sphere.center)

	// Calculate light direction
	lightDir := light.Position.Sub(point).Normalize()

	// Ambient component
	ambientStrength := float32(0.1)
	ambient := float32(sphere.color.R) * ambientStrength

	// Diffuse component
	diff := max(normal.Dot(lightDir), 0.0)
	diffuse := diff * float32(sphere.color.R)

	// Specular component
	specularStrength := float32(0.5)
	viewDir := point.Mul(-1).Normalize()
	reflectDir := lightDir.Mul(-1).Reflect(normal)
	spec := math32.Pow(max(viewDir.Dot(reflectDir), 0.0), 32)
	specular := specularStrength * spec

	// Distance attenuation
	// attenuation := maxDistance / totalDistance

	// Combine components
	final := min((ambient + diffuse + specular), 255)

	return color.RGBA{
		R: uint8(final / 255 * float32(sphere.color.R)),
		G: uint8(final / 255 * float32(sphere.color.G)),
		B: uint8(final / 255 * float32(sphere.color.B)),
		A: 255,
	}
}

func (t *TriangleSimple) CalculateNormal() {
	edge1 := t.v2.Sub(t.v1)
	edge2 := t.v3.Sub(t.v1)
	t.Normal = edge1.Cross(edge2).Normalize()
}

// func BoundingBoxCollision(BoundingBox [2]Vector, ray Ray) bool {
// 	// Precompute the inverse direction
// 	invDirX := 1.0 / ray.direction.x
// 	invDirY := 1.0 / ray.direction.y
// 	invDirZ := 1.0 / ray.direction.z

// 	// Compute the tmin and tmax for each axis directly
// 	tx1 := (BoundingBox[0].x - ray.origin.x) * invDirX
// 	tx2 := (BoundingBox[1].x - ray.origin.x) * invDirX
// 	tmin := min(tx1, tx2)
// 	tmax := max(tx1, tx2)

// 	ty1 := (BoundingBox[0].y - ray.origin.y) * invDirY
// 	ty2 := (BoundingBox[1].y - ray.origin.y) * invDirY
// 	tmin = max(tmin, min(ty1, ty2))
// 	tmax = min(tmax, max(ty1, ty2))

// 	tz1 := (BoundingBox[0].z - ray.origin.z) * invDirZ
// 	tz2 := (BoundingBox[1].z - ray.origin.z) * invDirZ
// 	tmin = max(tmin, min(tz1, tz2))
// 	tmax = min(tmax, max(tz1, tz2))
// 	return tmax >= max(0.0, tmin)
// }

func BoundingBoxCollision(BoundingBox [2]Vector, ray Ray) bool {
	// Handle zero components in ray direction to avoid division by zero
	const epsilon = 1e-7

	invDirX := float32(0)
	invDirY := float32(0)
	invDirZ := float32(0)

	// Precompute inverse directions with safety checks
	if math32.Abs(ray.direction.x) > epsilon {
		invDirX = 1.0 / ray.direction.x
	}
	if math32.Abs(ray.direction.y) > epsilon {
		invDirY = 1.0 / ray.direction.y
	}
	if math32.Abs(ray.direction.z) > epsilon {
		invDirZ = 1.0 / ray.direction.z
	}

	// Store sign of inverse directions to optimize min/max operations
	signX := invDirX < 0
	signY := invDirY < 0
	signZ := invDirZ < 0

	// Use sign to select bounds directly, avoiding branches
	bounds := BoundingBox
	var tmin, tmax float32

	if signX {
		tmin = (bounds[1].x - ray.origin.x) * invDirX
		tmax = (bounds[0].x - ray.origin.x) * invDirX
	} else {
		tmin = (bounds[0].x - ray.origin.x) * invDirX
		tmax = (bounds[1].x - ray.origin.x) * invDirX
	}

	if signY {
		tymin := (bounds[1].y - ray.origin.y) * invDirY
		tymax := (bounds[0].y - ray.origin.y) * invDirY

		// Early exit
		if tmin > tymax || tymin > tmax {
			return false
		}

		tmin = math32.Max(tmin, tymin)
		tmax = math32.Min(tmax, tymax)
	} else {
		tymin := (bounds[0].y - ray.origin.y) * invDirY
		tymax := (bounds[1].y - ray.origin.y) * invDirY

		// Early exit
		if tmin > tymax || tymin > tmax {
			return false
		}

		tmin = math32.Max(tmin, tymin)
		tmax = math32.Min(tmax, tymax)
	}

	if signZ {
		tzmin := (bounds[1].z - ray.origin.z) * invDirZ
		tzmax := (bounds[0].z - ray.origin.z) * invDirZ

		// Early exit
		if tmin > tzmax || tzmin > tmax {
			return false
		}

		tmin = math32.Max(tmin, tzmin)
		tmax = math32.Min(tmax, tzmax)
	} else {
		tzmin := (bounds[0].z - ray.origin.z) * invDirZ
		tzmax := (bounds[1].z - ray.origin.z) * invDirZ

		// Early exit
		if tmin > tzmax || tzmin > tmax {
			return false
		}

		tmin = math32.Max(tmin, tzmin)
		tmax = math32.Min(tmax, tzmax)
	}

	return tmax >= math32.Max(0.0, tmin)
}

func BoundingBoxCollisionEntryExitPoint(BBMin Vector, BBMax Vector, ray Ray) (hit bool, entry Vector, exit Vector) {
	// Handle zero components in ray direction
	invDirX := float32(0)
	invDirY := float32(0)
	invDirZ := float32(0)

	if ray.direction.x != 0 {
		invDirX = 1.0 / ray.direction.x
	}
	if ray.direction.y != 0 {
		invDirY = 1.0 / ray.direction.y
	}
	if ray.direction.z != 0 {
		invDirZ = 1.0 / ray.direction.z
	}

	// Compute intersection with x-aligned slabs
	tx1 := (BBMin.x - ray.origin.x) * invDirX
	tx2 := (BBMax.x - ray.origin.x) * invDirX
	tmin := min(tx1, tx2)
	tmax := max(tx1, tx2)

	// Compute intersection with y-aligned slabs
	ty1 := (BBMin.y - ray.origin.y) * invDirY
	ty2 := (BBMax.y - ray.origin.y) * invDirY
	tymin := min(ty1, ty2)
	tymax := max(ty1, ty2)

	// Early exit
	if tmin > tymax || tymin > tmax {
		return false, Vector{}, Vector{}
	}

	if tymin > tmin {
		tmin = tymin
	}
	if tymax < tmax {
		tmax = tymax
	}

	// Compute intersection with z-aligned slabs
	tz1 := (BBMin.z - ray.origin.z) * invDirZ
	tz2 := (BBMax.z - ray.origin.z) * invDirZ
	tzmin := min(tz1, tz2)
	tzmax := max(tz1, tz2)

	// Early exit
	if tmin > tzmax || tzmin > tmax {
		return false, Vector{}, Vector{}
	}

	if tzmin > tmin {
		tmin = tzmin
	}
	if tzmax < tmax {
		tmax = tzmax
	}

	// Check if intersection is behind the ray origin
	if tmax < 0 {
		return false, Vector{}, Vector{}
	}

	// Compute entry and exit points
	entry = Vector{
		x: ray.origin.x + tmin*ray.direction.x,
		y: ray.origin.y + tmin*ray.direction.y,
		z: ray.origin.z + tmin*ray.direction.z,
	}
	exit = Vector{
		x: ray.origin.x + tmax*ray.direction.x,
		y: ray.origin.y + tmax*ray.direction.y,
		z: ray.origin.z + tmax*ray.direction.z,
	}

	return true, entry, exit
}

func BoundingBoxCollisionDistance(BoundingBox [2]Vector, ray Ray) (bool, float32) {
	// Precompute the inverse direction
	invDirX := 1.0 / ray.direction.x
	invDirY := 1.0 / ray.direction.y
	invDirZ := 1.0 / ray.direction.z

	// Compute the tmin and tmax for each axis directly
	tx1 := (BoundingBox[0].x - ray.origin.x) * invDirX
	tx2 := (BoundingBox[1].x - ray.origin.x) * invDirX
	tmin := min(tx1, tx2)
	tmax := max(tx1, tx2)

	ty1 := (BoundingBox[0].y - ray.origin.y) * invDirY
	ty2 := (BoundingBox[1].y - ray.origin.y) * invDirY
	tmin = max(tmin, min(ty1, ty2))
	tmax = min(tmax, max(ty1, ty2))

	tz1 := (BoundingBox[0].z - ray.origin.z) * invDirZ
	tz2 := (BoundingBox[1].z - ray.origin.z) * invDirZ
	tmin = max(tmin, min(tz1, tz2))
	tmax = min(tmax, max(tz1, tz2))

	// Final intersection check
	if tmax >= max(0.0, tmin) {
		return true, tmin
	}

	return false, 0.0 // Return 0 distance if no intersection
}

func BoundingBoxCollisionVector(BBMin Vector, BBMax Vector, ray Ray) (bool, float32) {
	// Precompute the inverse direction
	invDirX := 1.0 / ray.direction.x
	invDirY := 1.0 / ray.direction.y
	invDirZ := 1.0 / ray.direction.z

	// Compute the tmin and tmax for each axis directly
	tx1 := (BBMax.x - ray.origin.x) * invDirX
	tx2 := (BBMin.x - ray.origin.x) * invDirX
	tmin := min(tx1, tx2)
	tmax := max(tx1, tx2)

	ty1 := (BBMax.y - ray.origin.y) * invDirY
	ty2 := (BBMin.y - ray.origin.y) * invDirY
	tmin = max(tmin, min(ty1, ty2))
	tmax = min(tmax, max(ty1, ty2))

	tz1 := (BBMax.z - ray.origin.z) * invDirZ
	tz2 := (BBMin.z - ray.origin.z) * invDirZ
	tmin = max(tmin, min(tz1, tz2))
	tmax = min(tmax, max(tz1, tz2))

	// Final intersection check
	if tmax >= max(0.0, tmin) {
		return true, tmin
	}

	return false, 0.0 // Return 0 distance if no intersection
}

// BoundingBoxCollisionPair checks if a ray intersects with two bounding boxes and returns
// hit status and distance for both boxes
func BoundingBoxCollisionPair(box1Min, box1Max, box2Min, box2Max Vector, ray Ray) (bool, bool, float32, float32) {
	// Precompute the inverse direction (once for both boxes)
	invDirX := 1.0 / ray.direction.x
	invDirY := 1.0 / ray.direction.y
	invDirZ := 1.0 / ray.direction.z

	// Box 1 intersection
	tx1_1 := (box1Min.x - ray.origin.x) * invDirX
	tx2_1 := (box1Max.x - ray.origin.x) * invDirX
	tmin_1 := min(tx1_1, tx2_1)
	tmax_1 := max(tx1_1, tx2_1)

	ty1_1 := (box1Min.y - ray.origin.y) * invDirY
	ty2_1 := (box1Max.y - ray.origin.y) * invDirY
	tmin_1 = max(tmin_1, min(ty1_1, ty2_1))
	tmax_1 = min(tmax_1, max(ty1_1, ty2_1))

	tz1_1 := (box1Min.z - ray.origin.z) * invDirZ
	tz2_1 := (box1Max.z - ray.origin.z) * invDirZ
	tmin_1 = max(tmin_1, min(tz1_1, tz2_1))
	tmax_1 = min(tmax_1, max(tz1_1, tz2_1))

	// Box 2 intersection
	tx1_2 := (box2Min.x - ray.origin.x) * invDirX
	tx2_2 := (box2Max.x - ray.origin.x) * invDirX
	tmin_2 := min(tx1_2, tx2_2)
	tmax_2 := max(tx1_2, tx2_2)

	ty1_2 := (box2Min.y - ray.origin.y) * invDirY
	ty2_2 := (box2Max.y - ray.origin.y) * invDirY
	tmin_2 = max(tmin_2, min(ty1_2, ty2_2))
	tmax_2 = min(tmax_2, max(ty1_2, ty2_2))

	tz1_2 := (box2Min.z - ray.origin.z) * invDirZ
	tz2_2 := (box2Max.z - ray.origin.z) * invDirZ
	tmin_2 = max(tmin_2, min(tz1_2, tz2_2))
	tmax_2 = min(tmax_2, max(tz1_2, tz2_2))

	// Check intersections
	hit1 := tmax_1 >= max(0.0, tmin_1)
	hit2 := tmax_2 >= max(0.0, tmin_2)

	// Return hit status and distances
	return hit1, hit2, tmin_1, tmin_2
}

func (triangle *TriangleSimple) Rotate(xAngle, yAngle, zAngle float32) {
	// Rotation matrices
	rotationMatrixX := [3][3]float32{
		{1, 0, 0},
		{0, math32.Cos(xAngle), -math32.Sin(xAngle)},
		{0, math32.Sin(xAngle), math32.Cos(xAngle)},
	}

	rotationMatrixY := [3][3]float32{
		{math32.Cos(yAngle), 0, math32.Sin(yAngle)},
		{0, 1, 0},
		{-math32.Sin(yAngle), 0, math32.Cos(yAngle)},
	}

	rotationMatrixZ := [3][3]float32{
		{math32.Cos(zAngle), -math32.Sin(zAngle), 0},
		{math32.Sin(zAngle), math32.Cos(zAngle), 0},
		{0, 0, 1},
	}

	// Apply the rotation matrices to each vertex
	triangle.v1 = rotateVector(triangle.v1, rotationMatrixX, rotationMatrixY, rotationMatrixZ)
	triangle.v2 = rotateVector(triangle.v2, rotationMatrixX, rotationMatrixY, rotationMatrixZ)
	triangle.v3 = rotateVector(triangle.v3, rotationMatrixX, rotationMatrixY, rotationMatrixZ)

	// Recalculate the bounding box
	triangle.CalculateBoundingBox()
}

func rotateVector(v Vector, rotationMatrixX, rotationMatrixY, rotationMatrixZ [3][3]float32) Vector {
	v = applyRotationMatrix(v, rotationMatrixX)
	v = applyRotationMatrix(v, rotationMatrixY)
	v = applyRotationMatrix(v, rotationMatrixZ)
	return v
}

func applyRotationMatrix(v Vector, matrix [3][3]float32) Vector {
	return Vector{
		x: matrix[0][0]*v.x + matrix[0][1]*v.y + matrix[0][2]*v.z,
		y: matrix[1][0]*v.x + matrix[1][1]*v.y + matrix[1][2]*v.z,
		z: matrix[2][0]*v.x + matrix[2][1]*v.y + matrix[2][2]*v.z,
	}
}

func CreateCube(center Vector, size float32, color ColorFloat32, refection float32, specular float32) []TriangleSimple {
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

	return []TriangleSimple{
		NewTriangle(vertices[0], vertices[1], vertices[2], color, refection, specular), // Front face
		NewTriangle(vertices[0], vertices[2], vertices[3], color, refection, specular),

		NewTriangle(vertices[4], vertices[5], vertices[6], color, refection, specular), // Back face
		NewTriangle(vertices[4], vertices[6], vertices[7], color, refection, specular),

		NewTriangle(vertices[0], vertices[1], vertices[5], color, refection, specular), // Bottom face
		NewTriangle(vertices[0], vertices[5], vertices[4], color, refection, specular),

		NewTriangle(vertices[2], vertices[3], vertices[7], color, refection, specular), // Top face
		NewTriangle(vertices[2], vertices[7], vertices[6], color, refection, specular),

		NewTriangle(vertices[1], vertices[2], vertices[6], color, refection, specular), // Right face
		NewTriangle(vertices[1], vertices[6], vertices[5], color, refection, specular),

		NewTriangle(vertices[0], vertices[3], vertices[7], color, refection, specular), // Left face
		NewTriangle(vertices[0], vertices[7], vertices[4], color, refection, specular),
	}
}

func CreatePlane(center Vector, normal Vector, width, height float32, color ColorFloat32, reflection float32, specular float32) []TriangleSimple {
	// Calculate the tangent vectors
	var tangent, bitangent Vector
	if math32.Abs(normal.x) > math32.Abs(normal.y) {
		tangent = Vector{normal.z, 0, -normal.x}.Normalize()
	} else {
		tangent = Vector{0, -normal.z, normal.y}.Normalize()
	}
	bitangent = normal.Cross(tangent)

	// Calculate the corner vertices
	halfWidth := width / 2
	halfHeight := height / 2
	v1 := center.Add(tangent.Mul(-halfWidth)).Add(bitangent.Mul(-halfHeight))
	v2 := center.Add(tangent.Mul(halfWidth)).Add(bitangent.Mul(-halfHeight))
	v3 := center.Add(tangent.Mul(halfWidth)).Add(bitangent.Mul(halfHeight))
	v4 := center.Add(tangent.Mul(-halfWidth)).Add(bitangent.Mul(halfHeight))

	return []TriangleSimple{
		NewTriangle(v1, v2, v3, color, reflection, specular),
		NewTriangle(v1, v3, v4, color, reflection, specular),
	}
}

func CreateSphere(center Vector, radius float32, color ColorFloat32, reflection float32, specular float32) []TriangleSimple {
	var triangles []TriangleSimple
	latitudeBands := 20
	longitudeBands := 20

	for lat := 0; lat < latitudeBands; lat++ {
		for long := 0; long < longitudeBands; long++ {
			lat0 := math.Pi * float64(-0.5+float32(lat)/float32(latitudeBands))
			z0 := math32.Sin(float32(lat0)) * radius
			zr0 := math32.Cos(float32(lat0)) * radius

			lat1 := math.Pi * float64(-0.5+float32(lat+1)/float32(latitudeBands))
			z1 := math32.Sin(float32(lat1)) * radius
			zr1 := math32.Cos(float32(lat1)) * radius

			lng0 := 2 * math.Pi * float64(float32(long)/float32(longitudeBands))
			x0 := math32.Cos(float32(lng0)) * zr0
			y0 := math32.Sin(float32(lng0)) * zr0

			lng1 := 2 * math.Pi * float64(float32(long+1)/float32(longitudeBands))
			x1 := math32.Cos(float32(lng1)) * zr0
			y1 := math32.Sin(float32(lng1)) * zr0

			lng2 := 2 * math.Pi * float64(float32(long)/float32(longitudeBands))
			x2 := math32.Cos(float32(lng2)) * zr1
			y2 := math32.Sin(float32(lng2)) * zr1

			lng3 := 2 * math.Pi * float64(float32(long+1)/float32(longitudeBands))
			x3 := math32.Cos(float32(lng3)) * zr1
			y3 := math32.Sin(float32(lng3)) * zr1

			triangles = append(triangles, NewTriangle(Vector{x0 + center.x, y0 + center.y, z0 + center.z}, Vector{x1 + center.x, y1 + center.y, z0 + center.z}, Vector{x2 + center.x, y2 + center.y, z1 + center.z}, color, reflection, specular))
			triangles = append(triangles, NewTriangle(Vector{x1 + center.x, y1 + center.y, z0 + center.z}, Vector{x3 + center.x, y3 + center.y, z1 + center.z}, Vector{x2 + center.x, y2 + center.y, z1 + center.z}, color, reflection, specular))
		}
	}

	return triangles
}

// func (triangle *Triangle) CalculateBoundingBox() {
// 	// Compute the minimum and maximum coordinates using float32 functions
// 	minX := math32.Min(triangle.v1.x, math32.Min(triangle.v2.x, triangle.v3.x))
// 	minY := math32.Min(triangle.v1.y, math32.Min(triangle.v2.y, triangle.v3.y))
// 	minZ := math32.Min(triangle.v1.z, math32.Min(triangle.v2.z, triangle.v3.z))
// 	maxX := math32.Max(triangle.v1.x, math32.Max(triangle.v2.x, triangle.v3.x))
// 	maxY := math32.Max(triangle.v1.y, math32.Max(triangle.v2.y, triangle.v3.y))
// 	maxZ := math32.Max(triangle.v1.z, math32.Max(triangle.v2.z, triangle.v3.z))

// 	// Set the BoundingBox with computed min and max values
// 	triangle.BoundingBox[0] = Vector{minX, minY, minZ}
// 	triangle.BoundingBox[1] = Vector{maxX, maxY, maxZ}
// }

func (triangle TriangleSimple) CalculateBoundingBox() (minBox Vector, maxBox Vector) {
	// Compute the minimum and maximum coordinates using float32 functions
	minX := math32.Min(triangle.v1.x, math32.Min(triangle.v2.x, triangle.v3.x))
	minY := math32.Min(triangle.v1.y, math32.Min(triangle.v2.y, triangle.v3.y))
	minZ := math32.Min(triangle.v1.z, math32.Min(triangle.v2.z, triangle.v3.z))
	maxX := math32.Max(triangle.v1.x, math32.Max(triangle.v2.x, triangle.v3.x))
	maxY := math32.Max(triangle.v1.y, math32.Max(triangle.v2.y, triangle.v3.y))
	maxZ := math32.Max(triangle.v1.z, math32.Max(triangle.v2.z, triangle.v3.z))

	// Set the BoundingBox with computed min and max values
	return Vector{minX, minY, minZ}, Vector{maxX, maxY, maxZ}
}

func NewTriangle(v1, v2, v3 Vector, color ColorFloat32, reflection float32, specular float32) TriangleSimple {
	triangle := TriangleSimple{v1: v1, v2: v2, v3: v3, color: color, reflection: reflection, specular: specular, directToScatter: 0.5}
	triangle.CalculateBoundingBox()
	triangle.CalculateNormal()
	return triangle
}

// func (triangle *Triangle) IntersectBoundingBox(ray Ray) bool {
// 	// Precompute the inverse direction
// 	invDirX := 1.0 / ray.direction.x
// 	invDirY := 1.0 / ray.direction.y
// 	invDirZ := 1.0 / ray.direction.z

// 	// Compute the tmin and tmax for each axis directly
// 	tx1 := (triangle.BoundingBox[0].x - ray.origin.x) * invDirX
// 	tx2 := (triangle.BoundingBox[1].x - ray.origin.x) * invDirX
// 	tmin := min(tx1, tx2)
// 	tmax := max(tx1, tx2)

// 	ty1 := (triangle.BoundingBox[0].y - ray.origin.y) * invDirY
// 	ty2 := (triangle.BoundingBox[1].y - ray.origin.y) * invDirY
// 	tmin = max(tmin, min(ty1, ty2))
// 	tmax = min(tmax, max(ty1, ty2))

// 	tz1 := (triangle.BoundingBox[0].z - ray.origin.z) * invDirZ
// 	tz2 := (triangle.BoundingBox[1].z - ray.origin.z) * invDirZ
// 	tmin = max(tmin, min(tz1, tz2))
// 	tmax = min(tmax, max(tz1, tz2))

// 	// Final intersection check
// 	return tmax >= max(0.0, tmin)
// }

type Intersection struct {
	PointOfIntersection Vector
	Color               ColorFloat32
	Normal              Vector
	Direction           Vector
	Distance            float32
	reflection          float32
	directToScatter     float32
	specular            float32
	Roughness           float32
	Metallic            float32
}

type IntersectionLean struct {
	PointOfIntersection Vector
	Color               ColorFloat32
	Normal              Vector
	// Direction           Vector
	// Distance            float32
	reflection      float32
	directToScatter float32
	specular        float32
	Roughness       float32
	Metallic        float32
}

// type IntersectionAdvance struct {
// 	PointOfIntersection Vector
// 	Color               ColorFloat32
// 	Normal              Vector
// 	Direction           Vector
// 	Distance            float32
// 	reflection          float32
// 	directToScatter     float32
// 	specular            float32
// 	Roughness           float32
// 	Metallic            float32
// 	TextureColor		color.RGBA
// }

type Light struct {
	Position  Vector
	Color     [3]float32
	intensity float32
}

// func (light *Light) CalculateLighting(intersection Intersection, bvh *BVHNode) color.RGBA {
// 	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
// 	shadowRay := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: lightDir}

// 	// Check if the point is in shadow
// 	inShadow := false
// 	if _, intersect := shadowRay.IntersectBVH(bvh); intersect {
// 		inShadow = true
// 	}

// 	// Ambient light contribution
// 	ambientFactor := 0.05 // Adjust ambient factor as needed
// 	ambientColor := color.RGBA{
// 		uint8(float64(light.Color.R) * ambientFactor),
// 		uint8(float64(light.Color.G) * ambientFactor),
// 		uint8(float64(light.Color.B) * ambientFactor),
// 		light.Color.A,
// 	}

// 	if inShadow {
// 		// If in shadow, return ambient color
// 		return ambientColor
// 	}

// 	// Calculate diffuse lighting
// 	lightIntensity := light.intensity * math32.Max(0.0, lightDir.Dot(intersection.Normal))
// 	finalColor := color.RGBA{
// 		clampUint8(float32(ambientColor.R) + lightIntensity*float32(intersection.Color.R)),
// 		clampUint8(float32(ambientColor.G) + lightIntensity*float32(intersection.Color.G)),
// 		clampUint8(float32(ambientColor.B) + lightIntensity*float32(intersection.Color.B)),
// 		ambientColor.A,
// 	}

// 	return finalColor
// }

// Helper function to clamp a float64 value to uint8 range
func clampUint8(value float32) uint8 {
	if value < 0 {
		return 0
	}
	if value > 255 {
		return 255
	}
	return uint8(value)
}

// var Old time.Duration
// var OldCount int64
// var New time.Duration
// var NewCount int64

// Intersect BVH average time: 497ns
// func (ray *Ray) IntersectBVH(nodeBVH *BVHNode) (Intersection, bool) {
// 	if nodeBVH.Triangles != nil {
// 		return IntersectTriangles(*ray, *nodeBVH.Triangles)
// 	}

// 	leftHit := nodeBVH.Left != nil && BoundingBoxCollision(nodeBVH.Left.BoundingBox, ray)
// 	rightHit := nodeBVH.Right != nil && BoundingBoxCollision(nodeBVH.Right.BoundingBox, ray)

// 	if leftHit && rightHit {
// 		leftIntersection, leftIntersect := ray.IntersectBVH(nodeBVH.Left)
// 		rightIntersection, rightIntersect := ray.IntersectBVH(nodeBVH.Right)

// 		if leftIntersect && rightIntersect {
// 			if leftIntersection.Distance < rightIntersection.Distance {
// 				return leftIntersection, true
// 			}
// 			return rightIntersection, true
// 		} else if leftIntersect {
// 			return leftIntersection, true
// 		} else if rightIntersect {
// 			return rightIntersection, true
// 		}
// 	} else if leftHit {
// 		return ray.IntersectBVH(nodeBVH.Left)
// 	} else if rightHit {
// 		return ray.IntersectBVH(nodeBVH.Right)
// 	}
// 	return Intersection{}, false
// }

// Intersect BVH average time:  458ns
func (ray Ray) IntersectBVH(nodeBVH *BVHNode) (Intersection, bool) {
	// Preallocate a stack large enough for the BVH depth
	stack := make([]*BVHNode, maxDepth)
	stackIndex := 0
	stack[stackIndex] = nodeBVH
	var closestIntersection Intersection
	hit := false

	for stackIndex >= 0 {
		// Pop the top item from the stack
		currentNode := stack[stackIndex]
		stackIndex--

		// If the node contains triangles, check for intersections
		if currentNode.active {
			// intersection, intersects := IntersectTrianglesSimple(*ray, *currentNode.Triangles)
			intersection, intersects := ray.IntersectTriangleSimple(currentNode.Triangles)
			if intersects {
				if !hit || intersection.Distance < closestIntersection.Distance {
					closestIntersection = intersection
					hit = true
				}
			}
			continue
		}
		// Check for bounding box intersections for left and right children
		var leftHit, rightHit bool
		var leftDist, rightDist float32

		if currentNode.Left != nil {
			leftHit, leftDist = BoundingBoxCollisionDistance(currentNode.Left.BoundingBox, ray)
		}
		if currentNode.Right != nil {
			rightHit, rightDist = BoundingBoxCollisionDistance(currentNode.Right.BoundingBox, ray)
		}

		// Prioritize traversal based on hit distance (closer node first)
		if leftHit && rightHit {
			if leftDist < rightDist {
				// Left is closer, traverse left first
				stackIndex++
				stack[stackIndex] = currentNode.Right
				stackIndex++
				stack[stackIndex] = currentNode.Left
			} else {
				// Right is closer, traverse right first
				stackIndex++
				stack[stackIndex] = currentNode.Left
				stackIndex++
				stack[stackIndex] = currentNode.Right
			}
		} else if leftHit {
			// Only left child is hit
			stackIndex++
			stack[stackIndex] = currentNode.Left
		} else if rightHit {
			// Only right child is hit
			stackIndex++
			stack[stackIndex] = currentNode.Right
		}
	}

	return closestIntersection, hit
}

func (ray Ray) IntersectBVH_Texture(nodeBVH *BVHNode, textureMap *[128]Texture) (Intersection, bool) {
	// Preallocate a stack large enough for the BVH depth
	stack := make([]*BVHNode, maxDepth)
	stackIndex := 0
	stack[stackIndex] = nodeBVH
	var closestIntersection Intersection
	hit := false

	// var time1 time.Duration
	// count := 1
	// var time2 time.Duration

	for stackIndex >= 0 {
		// Pop the top item from the stack
		currentNode := stack[stackIndex]
		stackIndex--

		// If the node contains triangles, check for intersections
		if currentNode.active {
			// intersection, intersects := IntersectTrianglesSimple(*ray, *currentNode.Triangles)
			// start := time.Now()
			intersection, intersects := ray.IntersectTriangleTexture(currentNode.Triangles, textureMap)
			// time1 += time.Since(start)
			// start = time.Now()
			// _, _ = ray.IntersectTriangleTextureGeneral(currentNode.Triangles.v1, currentNode.Triangles.v2, currentNode.Triangles.v3, currentNode.Triangles.Normal, textureMap, currentNode.Triangles.id)
			// time2 += time.Since(start)
			// Time1:  620  Time2:  470
			// Time1:  380  Time2:  280
			// Time1:  610  Time2:  330
			if intersects {
				if !hit || intersection.Distance < closestIntersection.Distance {
					closestIntersection = intersection
					hit = true
				}
			}
			continue
		}
		// Check for bounding box intersections for left and right children
		var leftHit, rightHit bool
		var leftDist, rightDist float32

		if currentNode.Left != nil {
			leftHit, leftDist = BoundingBoxCollisionDistance(currentNode.Left.BoundingBox, ray)
		}
		if currentNode.Right != nil {
			rightHit, rightDist = BoundingBoxCollisionDistance(currentNode.Right.BoundingBox, ray)
		}

		// Prioritize traversal based on hit distance (closer node first)
		if leftHit && rightHit {
			if leftDist < rightDist {
				// Left is closer, traverse left first
				stackIndex++
				stack[stackIndex] = currentNode.Right
				stackIndex++
				stack[stackIndex] = currentNode.Left
			} else {
				// Right is closer, traverse right first
				stackIndex++
				stack[stackIndex] = currentNode.Left
				stackIndex++
				stack[stackIndex] = currentNode.Right
			}
		} else if leftHit {
			// Only left child is hit
			stackIndex++
			stack[stackIndex] = currentNode.Left
		} else if rightHit {
			// Only right child is hit
			stackIndex++
			stack[stackIndex] = currentNode.Right
		}
	}
	// println("Time1: ", time1/time.Duration(count), " Time2: ", time2/time.Duration(count))
	return closestIntersection, hit
}

func (ray Ray) IntersectBVHLean_Texture(nodeBVH *BVHLeanNode, textureMap *[128]Texture) (Intersection, bool) {
	// Preallocate a stack large enough for the BVH depth
	stack := make([]*BVHLeanNode, maxDepth)
	stackIndex := 0
	stack[stackIndex] = nodeBVH
	var closestIntersection Intersection
	hit := false

	for stackIndex >= 0 {
		currentNode := stack[stackIndex]
		stackIndex--

		// If the node contains triangles, check for intersections
		if currentNode.active {
			intersection, intersects := ray.IntersectTriangleTextureGeneral(currentNode.TriangleBBOX.V1orBBoxMin, currentNode.TriangleBBOX.V2orBBoxMax, currentNode.TriangleBBOX.V3, currentNode.TriangleBBOX.normal, textureMap, currentNode.TriangleBBOX.id)
			if intersects {
				if !hit || intersection.Distance < closestIntersection.Distance {
					closestIntersection = intersection
					hit = true
				}
			}
			continue
		} else {
			// Check for bounding box intersections for left and right children
			var leftHit, rightHit bool
			var leftDist, rightDist float32

			if currentNode.Left != nil {
				// Check if the left child is Bounding Box
				if currentNode.Left.active == false {
					leftHit, leftDist = BoundingBoxCollisionVector(currentNode.Left.TriangleBBOX.V1orBBoxMin, currentNode.Left.TriangleBBOX.V2orBBoxMax, ray)
				} else {
					leftHit = true
					leftDist = math32.MaxFloat32
				}
			}
			if currentNode.Right != nil {
				// Check if the right child is Bounding Box
				if currentNode.Right.active == false {
					rightHit, rightDist = BoundingBoxCollisionVector(currentNode.Right.TriangleBBOX.V1orBBoxMin, currentNode.Right.TriangleBBOX.V2orBBoxMax, ray)
				} else {
					rightHit = true
					rightDist = math32.MaxFloat32
				}
			}

			// Prioritize traversal based on hit distance (closer node first)
			if leftHit && rightHit {
				if leftDist < rightDist {
					// Left is closer, traverse left first
					stackIndex++
					stack[stackIndex] = currentNode.Right
					stackIndex++
					stack[stackIndex] = currentNode.Left
				} else {
					// Right is closer, traverse right first
					stackIndex++
					stack[stackIndex] = currentNode.Left
					stackIndex++
					stack[stackIndex] = currentNode.Right
				}
			} else if leftHit {
				// Only left child is hit
				stackIndex++
				stack[stackIndex] = currentNode.Left
			} else if rightHit {
				// Only right child is hit
				stackIndex++
				stack[stackIndex] = currentNode.Right
			}
		}
	}
	return closestIntersection, hit
}

func (ray Ray) IntersectBVHLean_TextureLean(nodeBVH *BVHLeanNode, textureMap *[128]Texture) (IntersectionLean, bool) {
	// Preallocate a stack large enough for the BVH depth
	stack := make([]*BVHLeanNode, maxDepth)
	stackIndex := 0
	stack[stackIndex] = nodeBVH
	var closestIntersection IntersectionLean
	closestDist := float32(0)
	hit := false

	for stackIndex >= 0 {
		currentNode := stack[stackIndex]
		stackIndex--

		// If the node contains triangles, check for intersections
		if currentNode.active {
			intersection, intersects, dist := ray.IntersectTriangleTextureGeneralLean(currentNode.TriangleBBOX.V1orBBoxMin, currentNode.TriangleBBOX.V2orBBoxMax, currentNode.TriangleBBOX.V3, currentNode.TriangleBBOX.normal, textureMap, currentNode.TriangleBBOX.id)
			if intersects {
				if !hit || dist < closestDist {
					closestIntersection = intersection
					hit = true
					closestDist = dist
				}
			}
			continue
		} else {
			// Check for bounding box intersections for left and right children
			var leftHit, rightHit bool
			var leftDist, rightDist float32

			if currentNode.Left != nil {
				// Check if the left child is Bounding Box
				if currentNode.Left.active == false {
					leftHit, leftDist = BoundingBoxCollisionVector(currentNode.Left.TriangleBBOX.V1orBBoxMin, currentNode.Left.TriangleBBOX.V2orBBoxMax, ray)
				} else {
					leftHit = true
					leftDist = math32.MaxFloat32
				}
			}
			if currentNode.Right != nil {
				// Check if the right child is Bounding Box
				if currentNode.Right.active == false {
					rightHit, rightDist = BoundingBoxCollisionVector(currentNode.Right.TriangleBBOX.V1orBBoxMin, currentNode.Right.TriangleBBOX.V2orBBoxMax, ray)
				} else {
					rightHit = true
					rightDist = math32.MaxFloat32
				}
			}

			// Prioritize traversal based on hit distance (closer node first)
			if leftHit && rightHit {
				if leftDist < rightDist {
					// Left is closer, traverse left first
					stackIndex++
					stack[stackIndex] = currentNode.Right
					stackIndex++
					stack[stackIndex] = currentNode.Left
				} else {
					// Right is closer, traverse right first
					stackIndex++
					stack[stackIndex] = currentNode.Left
					stackIndex++
					stack[stackIndex] = currentNode.Right
				}
			} else if leftHit {
				// Only left child is hit
				stackIndex++
				stack[stackIndex] = currentNode.Left
			} else if rightHit {
				// Only right child is hit
				stackIndex++
				stack[stackIndex] = currentNode.Right
			}
		}
	}
	return closestIntersection, hit
}

func (ray Ray) IntersectBVHLean_TextureLeanOptim(nodeBVH *BVHLeanNode, textureMap *[128]Texture) (IntersectionLean, bool) {
	// Preallocate a stack large enough for the BVH depth
	stack := make([]*BVHLeanNode, maxDepth)
	stackIndex := 0
	stack[stackIndex] = nodeBVH
	var closestIntersection IntersectionLean
	closestDist := float32(math.MaxFloat32)
	hit := false

	for stackIndex >= 0 {
		currentNode := stack[stackIndex]
		stackIndex--

		// If the node contains triangles, check for intersections
		if currentNode.active {
			intersection, intersects, dist := ray.IntersectTriangleTextureGeneralLean(
				currentNode.TriangleBBOX.V1orBBoxMin,
				currentNode.TriangleBBOX.V2orBBoxMax,
				currentNode.TriangleBBOX.V3,
				currentNode.TriangleBBOX.normal,
				textureMap,
				currentNode.TriangleBBOX.id)

			if intersects && dist < closestDist {
				closestIntersection = intersection
				hit = true
				closestDist = dist
			}
			continue
		}

		// Both children exist, use the optimized function
		if currentNode.Left != nil && currentNode.Right != nil {
			// Get bounding box info for both children
			leftBBoxMin := currentNode.Left.TriangleBBOX.V1orBBoxMin
			leftBBoxMax := currentNode.Left.TriangleBBOX.V2orBBoxMax
			rightBBoxMin := currentNode.Right.TriangleBBOX.V1orBBoxMin
			rightBBoxMax := currentNode.Right.TriangleBBOX.V2orBBoxMax

			// Check both boxes at once, considering if they're leaf nodes
			leftHit := currentNode.Left.active   // Default true if it's a leaf node
			rightHit := currentNode.Right.active // Default true if it's a leaf node
			leftDist := float32(math.MaxFloat32)
			rightDist := float32(math.MaxFloat32)

			// Only check BB collision if it's not a leaf node
			if !currentNode.Left.active && !currentNode.Right.active {
				// Both are boxes, use optimized function
				leftHit, rightHit, leftDist, rightDist = BoundingBoxCollisionPair(
					leftBBoxMin, leftBBoxMax,
					rightBBoxMin, rightBBoxMax,
					ray)
			} else if !currentNode.Left.active {
				// Only left is a box
				leftHit, leftDist = BoundingBoxCollisionVector(leftBBoxMin, leftBBoxMax, ray)
			} else if !currentNode.Right.active {
				// Only right is a box
				rightHit, rightDist = BoundingBoxCollisionVector(rightBBoxMin, rightBBoxMax, ray)
			}

			// Prioritize traversal based on hit status and distance
			if leftHit && rightHit {
				if leftDist < rightDist {
					// Left is closer, traverse left first
					stackIndex++
					stack[stackIndex] = currentNode.Right
					stackIndex++
					stack[stackIndex] = currentNode.Left
				} else {
					// Right is closer, traverse right first
					stackIndex++
					stack[stackIndex] = currentNode.Left
					stackIndex++
					stack[stackIndex] = currentNode.Right
				}
			} else if leftHit {
				stackIndex++
				stack[stackIndex] = currentNode.Left
			} else if rightHit {
				stackIndex++
				stack[stackIndex] = currentNode.Right
			}
		} else if currentNode.Left != nil {
			// Only left child exists
			if !currentNode.Left.active {
				leftHit, _ := BoundingBoxCollisionVector(
					currentNode.Left.TriangleBBOX.V1orBBoxMin,
					currentNode.Left.TriangleBBOX.V2orBBoxMax,
					ray)
				if leftHit {
					stackIndex++
					stack[stackIndex] = currentNode.Left
				}
			} else {
				// It's a leaf node, always traverse
				stackIndex++
				stack[stackIndex] = currentNode.Left
			}
		} else if currentNode.Right != nil {
			// Only right child exists
			if !currentNode.Right.active {
				rightHit, _ := BoundingBoxCollisionVector(
					currentNode.Right.TriangleBBOX.V1orBBoxMin,
					currentNode.Right.TriangleBBOX.V2orBBoxMax,
					ray)
				if rightHit {
					stackIndex++
					stack[stackIndex] = currentNode.Right
				}
			} else {
				// It's a leaf node, always traverse
				stackIndex++
				stack[stackIndex] = currentNode.Right
			}
		}
	}

	return closestIntersection, hit
}

func (ray Ray) IntersectBVHLean_TextureWithNode(nodeBVH *BVHLeanNode, textureMap *[128]Texture) (Intersection, bool, *BVHLeanNode) {
	// Preallocate a stack large enough for the BVH depth
	stack := make([]*BVHLeanNode, maxDepth)
	stackIndex := 0
	stack[stackIndex] = nodeBVH
	var closestIntersection Intersection
	var HitTrianglePtr *BVHLeanNode
	hit := false

	for stackIndex >= 0 {
		currentNode := stack[stackIndex]
		stackIndex--

		// If the node contains triangles, check for intersections
		if currentNode.active {
			intersection, intersects := ray.IntersectTriangleTextureGeneral(currentNode.TriangleBBOX.V1orBBoxMin, currentNode.TriangleBBOX.V2orBBoxMax, currentNode.TriangleBBOX.V3, currentNode.TriangleBBOX.normal, textureMap, currentNode.TriangleBBOX.id)
			if intersects {
				if !hit || intersection.Distance < closestIntersection.Distance {
					closestIntersection = intersection
					HitTrianglePtr = currentNode
					hit = true
				}
			}
			continue
		} else {
			// Check for bounding box intersections for left and right children
			var leftHit, rightHit bool
			var leftDist, rightDist float32

			if currentNode.Left != nil {
				// Check if the left child is Bounding Box
				if currentNode.Left.active == false {
					leftHit, leftDist = BoundingBoxCollisionVector(currentNode.Left.TriangleBBOX.V1orBBoxMin, currentNode.Left.TriangleBBOX.V2orBBoxMax, ray)
				} else {
					leftHit = true
					leftDist = math32.MaxFloat32
				}
			}
			if currentNode.Right != nil {
				// Check if the right child is Bounding Box
				if currentNode.Right.active == false {
					rightHit, rightDist = BoundingBoxCollisionVector(currentNode.Right.TriangleBBOX.V1orBBoxMin, currentNode.Right.TriangleBBOX.V2orBBoxMax, ray)
				} else {
					rightHit = true
					rightDist = math32.MaxFloat32
				}
			}

			// Prioritize traversal based on hit distance (closer node first)
			if leftHit && rightHit {
				if leftDist < rightDist {
					// Left is closer, traverse left first
					stackIndex++
					stack[stackIndex] = currentNode.Right
					stackIndex++
					stack[stackIndex] = currentNode.Left
				} else {
					// Right is closer, traverse right first
					stackIndex++
					stack[stackIndex] = currentNode.Left
					stackIndex++
					stack[stackIndex] = currentNode.Right
				}
			} else if leftHit {
				// Only left child is hit
				stackIndex++
				stack[stackIndex] = currentNode.Left
			} else if rightHit {
				// Only right child is hit
				stackIndex++
				stack[stackIndex] = currentNode.Right
			}
		}
	}
	return closestIntersection, hit, HitTrianglePtr
}

// func (ray *Ray) IntersectTriangle(triangle Triangle) (Intersection, bool) {
// 	// Check if the ray intersects the bounding box of the triangle first
// 	if !triangle.IntersectBoundingBox(*ray) {
// 		return Intersection{}, false
// 	}

// 	// Möller–Trumbore intersection algorithm
// 	edge1 := triangle.v2.Sub(triangle.v1)
// 	edge2 := triangle.v3.Sub(triangle.v1)
// 	h := ray.direction.Cross(edge2)
// 	a := edge1.Dot(h)
// 	if a > -0.00001 && a < 0.00001 {
// 		return Intersection{}, false
// 	}
// 	f := 1.0 / a
// 	s := ray.origin.Sub(triangle.v1)
// 	u := f * s.Dot(h)
// 	if u < 0.0 || u > 1.0 {
// 		return Intersection{}, false
// 	}
// 	q := s.Cross(edge1)
// 	v := f * ray.direction.Dot(q)
// 	if v < 0.0 || u+v > 1.0 {
// 		return Intersection{}, false
// 	}
// 	t := f * edge2.Dot(q)
// 	if t > 0.00001 {
// 		return Intersection{PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)), Color: triangle.color, Normal: triangle.Normal, Direction: ray.direction, Distance: t, reflection: triangle.reflection}, true
// 	}
// 	return Intersection{}, false
// }

func (ray *Ray) IntersectTriangleSimple(triangle TriangleSimple) (Intersection, bool) {
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
		return Intersection{
			PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)),
			Color:               triangle.color,
			Normal:              triangle.Normal,
			Direction:           ray.direction,
			Distance:            t,
			reflection:          triangle.reflection,
			directToScatter:     triangle.directToScatter,
			Roughness:           triangle.Roughness,
			Metallic:            triangle.Metallic}, true
	}
	return Intersection{}, false
}

// func (ray Ray) IntersectTriangleTexture(triangle TriangleSimple, textureMap *[128]Texture) (Intersection, bool) {
//     // Möller–Trumbore intersection algorithm
//     edge1 := triangle.v2.Sub(triangle.v1)
//     edge2 := triangle.v3.Sub(triangle.v1)
//     h := ray.direction.Cross(edge2)
//     a := edge1.Dot(h)
//     if a > -0.00001 && a < 0.00001 {
//         return Intersection{}, false
//     }
//     f := 1.0 / a
//     s := ray.origin.Sub(triangle.v1)
//     u := f * s.Dot(h)
//     if u < 0.0 || u > 1.0 {
//         return Intersection{}, false
//     }
//     q := s.Cross(edge1)
//     v := f * ray.direction.Dot(q)
//     if v < 0.0 || u+v > 1.0 {
//         return Intersection{}, false
//     }
//     t := f * edge2.Dot(q)
//     if t <= 0.00001 {
//         return Intersection{}, false
//     }

//     // Compute barycentric coordinates
//     w := 1.0 - u - v

//     // Sample the texture using barycentric coordinates
//     texU := int(w * 127) // Scale to [0,127] range
//     texV := int(v * 127)
//     if texU < 0 {
//         texU = 0
//     } else if texU > 127 {
//         texU = 127
//     }
//     if texV < 0 {
//         texV = 0
//     } else if texV > 127 {
//         texV = 127
//     }

//     // Calculate tangent space basis vectors
//     deltaUV1 := Vector{1.0, 0.0, 0.0} // UV coordinates of vertex 2 - vertex 1
//     deltaUV2 := Vector{0.0, 1.0, 0.0} // UV coordinates of vertex 3 - vertex 1

//     // Calculate tangent and bitangent
//     f = 1.0 / (deltaUV1.x*deltaUV2.y - deltaUV2.x*deltaUV1.y)
//     tangent := Vector{
//         f * (deltaUV2.y*edge1.x - deltaUV1.y*edge2.x),
//         f * (deltaUV2.y*edge1.y - deltaUV1.y*edge2.y),
//         f * (deltaUV2.y*edge1.z - deltaUV1.y*edge2.z),
//     }
//     bitangent := Vector{
//         f * (-deltaUV2.x*edge1.x + deltaUV1.x*edge2.x),
//         f * (-deltaUV2.x*edge1.y + deltaUV1.x*edge2.y),
//         f * (-deltaUV2.x*edge1.z + deltaUV1.x*edge2.z),
//     }

//     // Normalize basis vectors
//     tangent = tangent.Normalize()
//     bitangent = bitangent.Normalize()
//     normal := triangle.Normal.Normalize()

//     // Get normal from texture
//     textureNormal := textureMap[uint8(1)].normals[texU][texV]

//     // Transform normal from tangent space to world space
//     worldNormal := Vector{
//         tangent.x*textureNormal.x + bitangent.x*textureNormal.y + normal.x*textureNormal.z,
//         tangent.y*textureNormal.x + bitangent.y*textureNormal.y + normal.y*textureNormal.z,
//         tangent.z*textureNormal.x + bitangent.z*textureNormal.y + normal.z*textureNormal.z,
//     }

//     // Normalize the final normal
//     worldNormal = worldNormal.Normalize()

//     // Return intersection data
//     return Intersection{
//         PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)),
//         Color:               textureMap[uint8(1)].texture[texU][texV], // Texture color
//         Normal:              worldNormal,                              // Normal perturbation
//         Direction:           ray.direction,
//         Distance:            t,
//         reflection:          textureMap[uint8(1)].reflection,
//         specular:            textureMap[uint8(1)].specular,
//         Roughness:           textureMap[uint8(1)].Roughness,
//         directToScatter:     textureMap[uint8(1)].directToScatter,
//         Metallic:            textureMap[uint8(1)].Metallic,
//     }, true
// }

// func (ray Ray) IntersectTriangleTexture(triangle TriangleSimple, textureMap *[128]Texture) (Intersection, bool) {
// 	// Möller–Trumbore intersection algorithm
// 	edge1 := triangle.v2.Sub(triangle.v1)
// 	edge2 := triangle.v3.Sub(triangle.v1)
// 	h := ray.direction.Cross(edge2)
// 	a := edge1.Dot(h)
// 	if a > -0.00001 && a < 0.00001 {
// 		return Intersection{}, false
// 	}
// 	f := 1.0 / a
// 	s := ray.origin.Sub(triangle.v1)
// 	u := f * s.Dot(h)
// 	if u < 0.0 || u > 1.0 {
// 		return Intersection{}, false
// 	}
// 	q := s.Cross(edge1)
// 	v := f * ray.direction.Dot(q)
// 	if v < 0.0 || u+v > 1.0 {
// 		return Intersection{}, false
// 	}
// 	t := f * edge2.Dot(q)
// 	if t <= 0.00001 {
// 		return Intersection{}, false
// 	}

// 	// Compute barycentric coordinates
// 	w := 1.0 - u - v

// 	// Sample the texture using barycentric coordinates
// 	texU := int(w * 127) // Scale to [0,127] range
// 	texV := int(v * 127)
// 	if texU < 0 {
// 		texU = 0
// 	} else if texU > 127 {
// 		texU = 127
// 	}
// 	if texV < 0 {
// 		texV = 0
// 	} else if texV > 127 {
// 		texV = 127
// 	}

// 	// fmt.Println(texU, texV)
// 	// fmt.Println(textureMap)

// 	// fmt.Println(Material.texture[texU][texV])

// 	normal := triangle.Normal.Add(triangle.Normal.Multiply(textureMap[uint8(1)].normals[texU][texV])).Normalize()
// 	// normal = normal.Normalize()

// 	// Return intersection data
// 	return Intersection{
// 		PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)),
// 		Color:               textureMap[uint8(1)].texture[texU][texV], // Texture color
// 		Normal:              normal,                                   // Normal perturbation
// 		Direction:           ray.direction,
// 		Distance:            t,
// 		reflection:          textureMap[uint8(1)].reflection,
// 		specular:            textureMap[uint8(1)].specular,
// 		Roughness:           textureMap[uint8(1)].Roughness,
// 		directToScatter:     textureMap[uint8(1)].directToScatter,
// 		Metallic:            textureMap[uint8(1)].Metallic,
// 	}, true
// }

func (ray Ray) IntersectTriangleTexture(triangle TriangleSimple, textureMap *[128]Texture) (Intersection, bool) {
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
	if t <= 0.00001 {
		return Intersection{}, false
	}

	// Compute barycentric coordinates
	w := 1.0 - u - v

	// Sample the texture using barycentric coordinates
	texU := int(w * 127)
	texV := int(v * 127)
	if texU < 0 {
		texU = 0
	} else if texU > 127 {
		texU = 127
	}
	if texV < 0 {
		texV = 0
	} else if texV > 127 {
		texV = 127
	}

	// Get the normal from the normal map (already in -1 to 1 range)
	normalMap := textureMap[uint8(1)].normals[texU][texV]

	// Blend the normals using a proper normal blending technique
	// This uses a hemisphere-based blending approach that preserves detail
	baseNormal := triangle.Normal
	perturbedNormal := normalMap

	// Ensure the perturbed normal is in the same hemisphere as the base normal
	if baseNormal.Dot(perturbedNormal) < 0 {
		perturbedNormal = perturbedNormal.Mul(-1)
	}

	// Blend the normals with emphasis on maintaining the base normal's orientation
	normal := Vector{
		x: (baseNormal.x + perturbedNormal.x) * 0.5,
		y: (baseNormal.y + perturbedNormal.y) * 0.5,
		z: (baseNormal.z + perturbedNormal.z) * 0.5,
	}.Normalize()

	return Intersection{
		PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)).Add(normal.Mul(0.01)),
		Color:               textureMap[uint8(1)].texture[texU][texV],
		Normal:              normal,
		Direction:           ray.direction,
		Distance:            t,
		reflection:          textureMap[uint8(1)].reflection,
		specular:            textureMap[uint8(1)].specular,
		Roughness:           textureMap[uint8(1)].Roughness,
		directToScatter:     textureMap[uint8(1)].directToScatter,
		Metallic:            textureMap[uint8(1)].Metallic,
	}, true
}

func (ray Ray) IntersectTriangleTextureGeneral(v1 Vector, v2 Vector, v3 Vector, baseNormal Vector, textureMap *[128]Texture, id int32) (Intersection, bool) {
	// Möller–Trumbore intersection algorithm
	edge1 := v2.Sub(v1)
	edge2 := v3.Sub(v1)
	h := ray.direction.Cross(edge2)
	a := edge1.Dot(h)
	if a > -0.00001 && a < 0.00001 {
		return Intersection{}, false
	}
	f := 1.0 / a
	s := ray.origin.Sub(v1)
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
	if t <= 0.00001 {
		return Intersection{}, false
	}

	// Compute barycentric coordinates
	w := 1.0 - u - v

	// Sample the texture using barycentric coordinates
	texU := int(w * 127)
	texV := int(v * 127)
	if texU < 0 {
		texU = 0
	} else if texU > 127 {
		texU = 127
	}
	if texV < 0 {
		texV = 0
	} else if texV > 127 {
		texV = 127
	}

	// index := uint8(1)

	// Blend the normals using a proper normal blending technique
	// This uses a hemisphere-based blending approach that preserves detail
	perturbedNormal := textureMap[id].normals[texU][texV]

	// Ensure the perturbed normal is in the same hemisphere as the base normal
	if baseNormal.Dot(perturbedNormal) < 0 {
		perturbedNormal = perturbedNormal.Mul(-1)
	}

	// Blend the normals with emphasis on maintaining the base normal's orientation
	normal := Vector{
		x: (baseNormal.x + perturbedNormal.x) * 0.5,
		y: (baseNormal.y + perturbedNormal.y) * 0.5,
		z: (baseNormal.z + perturbedNormal.z) * 0.5,
	}.Normalize()

	return Intersection{
		PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)).Add(normal.Mul(0.01)),
		Color:               textureMap[id].texture[texU][texV],
		Normal:              normal,
		Direction:           ray.direction,
		Distance:            t,
		reflection:          textureMap[id].reflection,
		specular:            textureMap[id].specular,
		Roughness:           textureMap[id].Roughness,
		directToScatter:     textureMap[id].directToScatter,
		Metallic:            textureMap[id].Metallic,
	}, true
}

func (ray Ray) IntersectTriangleTextureGeneralLean(v1 Vector, v2 Vector, v3 Vector, baseNormal Vector, textureMap *[128]Texture, id int32) (IntersectionLean, bool, float32) {
	// Möller–Trumbore intersection algorithm
	edge1 := v2.Sub(v1)
	edge2 := v3.Sub(v1)
	h := ray.direction.Cross(edge2)
	a := edge1.Dot(h)
	if a > -0.00001 && a < 0.00001 {
		return IntersectionLean{}, false, 0
	}
	f := 1.0 / a
	s := ray.origin.Sub(v1)
	u := f * s.Dot(h)
	if u < 0.0 || u > 1.0 {
		return IntersectionLean{}, false, 0
	}
	q := s.Cross(edge1)
	v := f * ray.direction.Dot(q)
	if v < 0.0 || u+v > 1.0 {
		return IntersectionLean{}, false, 0
	}
	t := f * edge2.Dot(q)
	if t <= 0.00001 {
		return IntersectionLean{}, false, 0
	}

	// Compute barycentric coordinates
	w := 1.0 - u - v

	// Sample the texture using barycentric coordinates
	texU := int(w * 127)
	texV := int(v * 127)
	if texU < 0 {
		texU = 0
	} else if texU > 127 {
		texU = 127
	}
	if texV < 0 {
		texV = 0
	} else if texV > 127 {
		texV = 127
	}

	// index := uint8(1)

	// Blend the normals using a proper normal blending technique
	// This uses a hemisphere-based blending approach that preserves detail
	perturbedNormal := textureMap[id].normals[texU][texV]

	// Ensure the perturbed normal is in the same hemisphere as the base normal
	if baseNormal.Dot(perturbedNormal) < 0 {
		perturbedNormal = perturbedNormal.Mul(-1)
	}

	// Blend the normals with emphasis on maintaining the base normal's orientation
	normal := Vector{
		x: (baseNormal.x + perturbedNormal.x) * 0.5,
		y: (baseNormal.y + perturbedNormal.y) * 0.5,
		z: (baseNormal.z + perturbedNormal.z) * 0.5,
	}.Normalize()

	return IntersectionLean{
		PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)).Add(normal.Mul(0.01)),
		Color:               textureMap[id].texture[texU][texV],
		Normal:              normal,
		reflection:          textureMap[id].reflection,
		specular:            textureMap[id].specular,
		Roughness:           textureMap[id].Roughness,
		directToScatter:     textureMap[id].directToScatter,
		Metallic:            textureMap[id].Metallic,
	}, true, t
}

// func (ray Ray) IntersectTriangleTextureGeneral(v1 Vector, v2 Vector, v3 Vector, baseNormal Vector, textureMap *[128]Texture, id uint8) (Intersection, bool) {
//     // Möller–Trumbore intersection algorithm
//     edge1 := v2.Sub(v1)
//     edge2 := v3.Sub(v1)
//     h := ray.direction.Cross(edge2)
//     a := edge1.Dot(h)
//     if a > -0.00001 && a < 0.00001 {
//         return Intersection{}, false
//     }
//     f := 1.0 / a
//     s := ray.origin.Sub(v1)
//     u := f * s.Dot(h)
//     if u < 0.0 || u > 1.0 {
//         return Intersection{}, false
//     }
//     q := s.Cross(edge1)
//     v := f * ray.direction.Dot(q)
//     if v < 0.0 || u+v > 1.0 {
//         return Intersection{}, false
//     }
//     t := f * edge2.Dot(q)
//     if t <= 0.00001 {
//         return Intersection{}, false
//     }

//     // Compute barycentric coordinates
//     w := 1.0 - u - v

//     // Sample the texture using barycentric coordinates
//     texU := int(w * 127)
//     texV := int(v * 127)
//     if texU < 0 {
//         texU = 0
//     } else if texU > 127 {
//         texU = 127
//     }
//     if texV < 0 {
//         texV = 0
//     } else if texV > 127 {
//         texV = 127
//     }

//     index := uint8(1)

//     // Get the normal from the normal map
//     normalMap := textureMap[index].normals[texU][texV]

//     // Calculate displacement amount based on normal map intensity
//     displacementAmount := normalMap.Length() * 0.5 // Adjust multiplier to control displacement strength

//     // Displace the intersection point along the base normal
//     hitPoint := ray.origin.Add(ray.direction.Mul(t))
//     displacedPoint := hitPoint.Add(baseNormal.Mul(displacementAmount))

//     // Recalculate the final t value for the displaced point
//     newT := ray.origin.Sub(displacedPoint).Length()

//     // Calculate the perturbed normal by blending base normal with normal map
//     perturbedNormal := textureMap[index].normals[texU][texV]
//     if baseNormal.Dot(perturbedNormal) < 0 {
//         perturbedNormal = perturbedNormal.Mul(-1)
//     }

//     normal := Vector{
//         x: (baseNormal.x + perturbedNormal.x) * 0.5,
//         y: (baseNormal.y + perturbedNormal.y) * 0.5,
//         z: (baseNormal.z + perturbedNormal.z) * 0.5,
//     }.Normalize()

//     return Intersection{
//         PointOfIntersection: displacedPoint,
//         Color:               textureMap[index].texture[texU][texV],
//         Normal:              normal,
//         Direction:           ray.direction,
//         Distance:            newT,
//         reflection:          textureMap[index].reflection,
//         specular:            textureMap[index].specular,
//         Roughness:           textureMap[index].Roughness,
//         directToScatter:     textureMap[index].directToScatter,
//         Metallic:            textureMap[index].Metallic,
//     }, true
// }

// func IntersectTriangles(ray Ray, triangles []Triangle) (Intersection, bool) {
// 	// Initialize the closest intersection and hit status
// 	closestIntersection := Intersection{Distance: math32.MaxFloat32}
// 	hasIntersection := false

// 	// Iterate over each triangle for the given ray
// 	for _, triangle := range triangles {
// 		// Check if the ray intersects the bounding box of the triangle first
// 		if !triangle.IntersectBoundingBox(ray) {
// 			continue
// 		}

// 		// Möller–Trumbore intersection algorithm
// 		edge1 := triangle.v2.Sub(triangle.v1)
// 		edge2 := triangle.v3.Sub(triangle.v1)
// 		h := ray.direction.Cross(edge2)
// 		a := edge1.Dot(h)
// 		if a > -0.00001 && a < 0.00001 {
// 			continue
// 		}
// 		f := 1.0 / a
// 		s := ray.origin.Sub(triangle.v1)
// 		u := f * s.Dot(h)
// 		if u < 0.0 || u > 1.0 {
// 			continue
// 		}
// 		q := s.Cross(edge1)
// 		v := f * ray.direction.Dot(q)
// 		if v < 0.0 || u+v > 1.0 {
// 			continue
// 		}
// 		t := f * edge2.Dot(q)
// 		if t > 0.00001 {
// 			tempIntersection := Intersection{
// 				PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)),
// 				Color:               triangle.color,
// 				Normal:              triangle.Normal,
// 				Direction:           ray.direction,
// 				Distance:            t,
// 				reflection:          triangle.reflection,
// 				specular:            triangle.specular,
// 			}

// 			// Update the closest intersection if the new one is closer
// 			if t < closestIntersection.Distance {
// 				closestIntersection = tempIntersection
// 				hasIntersection = true
// 			}
// 		}
// 	}

// 	return closestIntersection, hasIntersection
// }

func IntersectTrianglesSimple(ray Ray, triangles []TriangleSimple) (Intersection, bool) {
	// Initialize the closest intersection and hit status
	closestIntersection := Intersection{Distance: math32.MaxFloat32}
	hasIntersection := false

	// Iterate over each triangle for the given ray
	for _, triangle := range triangles {
		// Möller–Trumbore intersection algorithm
		edge1 := triangle.v2.Sub(triangle.v1)
		edge2 := triangle.v3.Sub(triangle.v1)
		h := ray.direction.Cross(edge2)
		a := edge1.Dot(h)
		if a > -0.00001 && a < 0.00001 {
			continue
		}
		f := 1.0 / a
		s := ray.origin.Sub(triangle.v1)
		u := f * s.Dot(h)
		if u < 0.0 || u > 1.0 {
			continue
		}
		q := s.Cross(edge1)
		v := f * ray.direction.Dot(q)
		if v < 0.0 || u+v > 1.0 {
			continue
		}
		t := f * edge2.Dot(q)
		if t > 0.00001 {
			tempIntersection := Intersection{
				PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)),
				Color:               triangle.color,
				Normal:              triangle.Normal,
				Direction:           ray.direction,
				Distance:            t,
				reflection:          triangle.reflection,
				specular:            triangle.specular,
				directToScatter:     triangle.directToScatter,
			}

			// Update the closest intersection if the new one is closer
			if t < closestIntersection.Distance {
				closestIntersection = tempIntersection
				hasIntersection = true
			}
		}
	}

	return closestIntersection, hasIntersection
}

type Camera struct {
	Position     Vector
	xAxis, yAxis float32
}

func TraceRay(ray Ray, depth int, light Light, samples int) ColorFloat32 {
	if depth == 0 {
		return ColorFloat32{}
	}

	intersection, intersect := ray.IntersectBVH(BVH)
	if !intersect {
		return ColorFloat32{}
	}

	// Scatter calculation
	var scatteredRed, scatteredGreen, scatteredBlue float32
	uVec := Vector{1.0, 0.0, 0.0}
	if math32.Abs(intersection.Normal.x) > 0.1 {
		uVec = Vector{0.0, 1.0, 0.0}
	}
	uVec = uVec.Cross(intersection.Normal).Normalize()
	vVec := intersection.Normal.Cross(uVec)

	for i := 0; i < samples; i++ {
		u := rand.Float64()
		v := rand.Float32()
		r := float32(math.Sqrt(u))
		theta := 2 * math32.Pi * v

		directionLocal := uVec.Mul(r * math32.Cos(theta)).Add(vVec.Mul(r * math32.Sin(theta))).Add(intersection.Normal.Mul(float32(math.Sqrt(1 - u))))

		scatterRay := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: directionLocal.Normalize()}

		if bvhIntersection, scatterIntersect := scatterRay.IntersectBVH(BVH); scatterIntersect && bvhIntersection.Distance != math32.MaxFloat32 {
			scatteredRed += float32(bvhIntersection.Color.R)
			scatteredGreen += float32(bvhIntersection.Color.G)
			scatteredBlue += float32(bvhIntersection.Color.B)
		}
	}

	if samples > 0 {
		s := float32(samples)
		scatteredRed /= s
		scatteredGreen /= s
		scatteredBlue /= s
	}

	ratioScatterToDirect := 1 - intersection.reflection
	scatteredColor := ColorFloat32{
		R: scatteredRed * ratioScatterToDirect,
		G: scatteredGreen * ratioScatterToDirect,
		B: scatteredBlue * ratioScatterToDirect,
		A: float32(intersection.Color.A),
	}

	// Reflection and specular calculations
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	reflectDir := lightDir.Mul(-1).Reflect(intersection.Normal)

	reflectRayOrigin := intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001))

	reflectRay := Ray{origin: reflectRayOrigin, direction: reflectDir}

	tempIntersection, _ := reflectRay.IntersectBVH(BVH)

	directReflectionColor := ColorFloat32{
		R: tempIntersection.Color.R * intersection.reflection,
		G: tempIntersection.Color.G * intersection.reflection,
		B: tempIntersection.Color.B * intersection.reflection,
		A: intersection.Color.A,
	}

	shadowRay := Ray{
		origin:    reflectRayOrigin,
		direction: lightDir,
	}
	_, inShadow := shadowRay.IntersectBVH(BVH)

	viewDir := ray.origin.Sub(intersection.PointOfIntersection).Normalize()
	specularFactor := math32.Pow(math32.Max(0.0, viewDir.Dot(reflectDir)), intersection.specular)
	specularIntensity := light.intensity * specularFactor

	var lightIntensity = float32(0.005)
	if !inShadow {
		lightIntensity = light.intensity * math32.Max(0.0, lightDir.Dot(intersection.Normal))
	}

	finalColor := ColorFloat32{
		R: ((directReflectionColor.R + scatteredColor.R) * (1 - intersection.directToScatter)) + ((intersection.Color.R) * intersection.directToScatter) + (specularIntensity*(light.Color[0]))*lightIntensity*light.Color[0],
		G: ((directReflectionColor.G + scatteredColor.G) * (1 - intersection.directToScatter)) + ((intersection.Color.G) * intersection.directToScatter) + (specularIntensity*(light.Color[1]))*lightIntensity*light.Color[1],
		B: ((directReflectionColor.B + scatteredColor.B) * (1 - intersection.directToScatter)) + ((intersection.Color.B) * intersection.directToScatter) + (specularIntensity*(light.Color[2]))*lightIntensity*light.Color[2],
		A: float32(intersection.Color.A),
	}

	bounceRay := Ray{origin: reflectRayOrigin, direction: reflectDir}
	bouncedColor := TraceRay(bounceRay, depth-1, light, samples)

	Color := ColorFloat32{
		R: (finalColor.R*intersection.directToScatter + (float32(bouncedColor.R) * (1 - intersection.directToScatter))),
		G: (finalColor.G*intersection.directToScatter + (float32(bouncedColor.G) * (1 - intersection.directToScatter))),
		B: (finalColor.B*intersection.directToScatter + (float32(bouncedColor.B) * (1 - intersection.directToScatter))),
		A: finalColor.A,
	}

	// Color := color.RGBA{
	// 	R: clampUint8((finalColor.R + float32(bouncedColor.R)) / 2),
	// 	G: clampUint8((finalColor.G + float32(bouncedColor.G)) / 2),
	// 	B: clampUint8((finalColor.B + float32(bouncedColor.B)) / 2),
	// 	A: uint8(finalColor.A),
	// }

	return Color
}

func FresnelSchlick(cosTheta, F0 float32) float32 {
	return F0 + (1.0-F0)*math32.Pow(1.0-cosTheta, 5)
}

func GGXDistribution(NdotH, roughness float32) float32 {
	alpha := roughness * roughness
	alpha2 := alpha * alpha
	NdotH2 := NdotH * NdotH
	denom := NdotH2*(alpha2-1.0) + 1.0
	return alpha2 / (math32.Pi * denom * denom)
}

func TraceRayV3(ray Ray, depth int, light Light, samples int) ColorFloat32 {
	if depth <= 0 {
		return ColorFloat32{}
	}

	intersection, intersect := ray.IntersectBVH(BVH)
	if !intersect {
		return ColorFloat32{}
	}

	viewDir := ray.origin.Sub(intersection.PointOfIntersection).Normalize()
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	halfwayDir := lightDir.Add(viewDir).Normalize()

	// Calculate important dot products
	NdotL := math32.Max(0.0, intersection.Normal.Dot(lightDir))
	NdotV := math32.Max(0.0, intersection.Normal.Dot(viewDir))
	NdotH := math32.Max(0.0, intersection.Normal.Dot(halfwayDir))

	// Calculate Fresnel term
	F0 := float32(intersection.Metallic) // Base reflectivity for non-metals
	fresnel := FresnelSchlick(NdotV, F0)

	// Calculate roughness-based distribution
	distribution := GGXDistribution(NdotH, intersection.Roughness)

	// Scatter calculation using hemisphere sampling
	var scatteredColor ColorFloat32
	rayOriginOffset := intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001))

	for i := 0; i < samples; i++ {
		scatterDirection := SampleHemisphere(intersection.Normal)
		scatterDirection = scatterDirection.Perturb(intersection.Normal, intersection.Roughness)

		scatterRay := Ray{
			origin:    rayOriginOffset,
			direction: scatterDirection.Normalize(),
		}

		if bvhIntersection, scatterIntersect := scatterRay.IntersectBVH(BVH); scatterIntersect && bvhIntersection.Distance != math32.MaxFloat32 {
			scatteredColor.R += bvhIntersection.Color.R
			scatteredColor.G += bvhIntersection.Color.G
			scatteredColor.B += bvhIntersection.Color.B
		}
	}

	if samples > 0 {
		s := float32(samples)
		scatteredColor = ColorFloat32{
			R: scatteredColor.R / s,
			G: scatteredColor.G / s,
			B: scatteredColor.B / s,
		}
	}

	// Calculate reflection direction using Fresnel
	reflectDir := lightDir.Mul(-1).Reflect(intersection.Normal)
	reflectRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	tempIntersection, _ := reflectRay.IntersectBVH(BVH)

	// Apply Fresnel to reflection color
	directReflectionColor := ColorFloat32{
		R: tempIntersection.Color.R * fresnel,
		G: tempIntersection.Color.G * fresnel,
		B: tempIntersection.Color.B * fresnel,
		A: intersection.Color.A,
	}

	// Shadow calculation
	shadowRay := Ray{
		origin:    rayOriginOffset,
		direction: lightDir,
	}
	_, inShadow := shadowRay.IntersectBVH(BVH)

	// Calculate specular using GGX distribution
	var lightIntensity float32 = 0.005
	if !inShadow {
		lightIntensity = light.intensity * NdotL
	}

	specularIntensity := distribution * fresnel * lightIntensity * intersection.specular

	specularColor := ColorFloat32{
		R: specularIntensity * light.Color[0],
		G: specularIntensity * light.Color[1],
		B: specularIntensity * light.Color[2],
	}

	// Calculate diffuse contribution
	diffuseFactor := (1.0 - fresnel) * (1.0 / math32.Pi)
	diffuseColor := ColorFloat32{
		R: intersection.Color.R * diffuseFactor * NdotL * lightIntensity,
		G: intersection.Color.G * diffuseFactor * NdotL * lightIntensity,
		B: intersection.Color.B * diffuseFactor * NdotL * lightIntensity,
	}

	// Combine direct and indirect lighting
	finalColor := ColorFloat32{
		R: diffuseColor.R + specularColor.R + (directReflectionColor.R * intersection.directToScatter) + (scatteredColor.R * (1 - intersection.directToScatter)),
		G: diffuseColor.G + specularColor.G + (directReflectionColor.G * intersection.directToScatter) + (scatteredColor.G * (1 - intersection.directToScatter)),
		B: diffuseColor.B + specularColor.B + (directReflectionColor.B * intersection.directToScatter) + (scatteredColor.B * (1 - intersection.directToScatter)),
		A: intersection.Color.A,
	}

	// Calculate bounced contribution
	bounceRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	bouncedColor := TraceRayV3(bounceRay, depth-1, light, samples)

	// Final color composition with energy conservation
	return ColorFloat32{
		R: light.Color[0] * (finalColor.R*(1.0-fresnel) + bouncedColor.R*fresnel),
		G: light.Color[1] * (finalColor.G*(1.0-fresnel) + bouncedColor.G*fresnel),
		B: light.Color[2] * (finalColor.B*(1.0-fresnel) + bouncedColor.B*fresnel),
		A: finalColor.A,
	}
}

func TraceRayV3Advance(ray Ray, depth int, light Light, samples int) (c ColorFloat32, distance float32, normal Vector) {
	if depth <= 0 {
		return ColorFloat32{}, 0, Vector{}
	}

	intersection, intersect := ray.IntersectBVH(BVH)
	if !intersect {
		return ColorFloat32{}, 0, Vector{}
	}

	viewDir := ray.origin.Sub(intersection.PointOfIntersection).Normalize()
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	halfwayDir := lightDir.Add(viewDir).Normalize()

	// Calculate important dot products
	NdotL := math32.Max(0.0, intersection.Normal.Dot(lightDir))
	NdotV := math32.Max(0.0, intersection.Normal.Dot(viewDir))
	NdotH := math32.Max(0.0, intersection.Normal.Dot(halfwayDir))

	// Calculate Fresnel term
	F0 := intersection.Metallic // Base reflectivity for non-metals
	fresnel := FresnelSchlick(NdotV, F0)

	// Calculate roughness-based distribution
	distribution := GGXDistribution(NdotH, intersection.Roughness)

	// Scatter calculation using hemisphere sampling
	var scatteredColor ColorFloat32
	rayOriginOffset := intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001))

	for i := 0; i < samples; i++ {
		scatterDirection := SampleHemisphere(intersection.Normal)
		scatterDirection = scatterDirection.Perturb(intersection.Normal, intersection.Roughness)

		scatterRay := Ray{
			origin:    rayOriginOffset,
			direction: scatterDirection.Normalize(),
		}

		if bvhIntersection, scatterIntersect := scatterRay.IntersectBVH(BVH); scatterIntersect && bvhIntersection.Distance != math32.MaxFloat32 {
			scatteredColor.R += bvhIntersection.Color.R
			scatteredColor.G += bvhIntersection.Color.G
			scatteredColor.B += bvhIntersection.Color.B
		}
	}

	if samples > 0 {
		s := float32(samples)
		scatteredColor = ColorFloat32{
			R: scatteredColor.R / s,
			G: scatteredColor.G / s,
			B: scatteredColor.B / s,
		}
	}

	// Calculate reflection direction using Fresnel
	reflectDir := lightDir.Mul(-1).Reflect(intersection.Normal)
	reflectRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	tempIntersection, _ := reflectRay.IntersectBVH(BVH)

	// Apply Fresnel to reflection color
	directReflectionColor := ColorFloat32{
		R: tempIntersection.Color.R * fresnel,
		G: tempIntersection.Color.G * fresnel,
		B: tempIntersection.Color.B * fresnel,
		A: intersection.Color.A,
	}

	// Shadow calculation
	shadowRay := Ray{
		origin:    rayOriginOffset,
		direction: lightDir,
	}
	_, inShadow := shadowRay.IntersectBVH(BVH)

	// Calculate specular using GGX distribution
	var lightIntensity float32 = 0.005
	if !inShadow {
		lightIntensity = light.intensity * NdotL
	}

	specularIntensity := distribution * fresnel * lightIntensity * intersection.specular

	specularColor := ColorFloat32{
		R: specularIntensity * light.Color[0],
		G: specularIntensity * light.Color[1],
		B: specularIntensity * light.Color[2],
	}

	// Calculate diffuse contribution
	diffuseFactor := (1.0 - fresnel) * (1.0 / math32.Pi)
	diffuseColor := ColorFloat32{
		R: intersection.Color.R * diffuseFactor * NdotL * lightIntensity,
		G: intersection.Color.G * diffuseFactor * NdotL * lightIntensity,
		B: intersection.Color.B * diffuseFactor * NdotL * lightIntensity,
	}

	// Combine direct and indirect lighting
	finalColor := ColorFloat32{
		R: light.Color[0] * (diffuseColor.R + specularColor.R + (directReflectionColor.R * intersection.directToScatter) + (scatteredColor.R * (1 - intersection.directToScatter))),
		G: light.Color[1] * (diffuseColor.G + specularColor.G + (directReflectionColor.G * intersection.directToScatter) + (scatteredColor.G * (1 - intersection.directToScatter))),
		B: light.Color[2] * (diffuseColor.B + specularColor.B + (directReflectionColor.B * intersection.directToScatter) + (scatteredColor.B * (1 - intersection.directToScatter))),
		A: intersection.Color.A,
	}

	// Calculate bounced contribution
	bounceRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	bouncedColor := TraceRayV3(bounceRay, depth-1, light, samples)

	// Final color composition with energy conservation
	return ColorFloat32{
		R: finalColor.R*(1.0-fresnel) + bouncedColor.R*fresnel,
		G: finalColor.G*(1.0-fresnel) + bouncedColor.G*fresnel,
		B: finalColor.B*(1.0-fresnel) + bouncedColor.B*fresnel,
		A: finalColor.A,
	}, intersection.Distance, intersection.Normal
}

func TraceRayV3AdvanceTexture(ray Ray, depth int, light Light, samples int, textureMap *[128]Texture, BVH *BVHNode) (c ColorFloat32, normal Vector) {
	if depth <= 0 {
		return ColorFloat32{}, Vector{}
	}

	intersection, intersect := ray.IntersectBVH_Texture(BVH, textureMap)
	if !intersect {
		return ColorFloat32{}, Vector{}
	}

	viewDir := ray.origin.Sub(intersection.PointOfIntersection).Normalize()
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	halfwayDir := lightDir.Add(viewDir).Normalize()

	// Calculate important dot products
	NdotL := math32.Max(0.0, intersection.Normal.Dot(lightDir))
	NdotV := math32.Max(0.0, intersection.Normal.Dot(viewDir))
	NdotH := math32.Max(0.0, intersection.Normal.Dot(halfwayDir))

	// Calculate Fresnel term
	F0 := intersection.Metallic // Base reflectivity for non-metals
	fresnel := FresnelSchlick(NdotV, F0)

	// Calculate roughness-based distribution
	distribution := GGXDistribution(NdotH, intersection.Roughness)

	// Scatter calculation using hemisphere sampling
	var scatteredColor ColorFloat32
	rayOriginOffset := intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.01))

	for i := 0; i < samples; i++ {
		scatterDirection := SampleHemisphere(intersection.Normal)
		scatterDirection = scatterDirection.Perturb(intersection.Normal, intersection.Roughness)

		scatterRay := Ray{
			origin:    rayOriginOffset,
			direction: scatterDirection.Normalize(),
		}

		if bvhIntersection, scatterIntersect := scatterRay.IntersectBVH_Texture(BVH, textureMap); scatterIntersect && bvhIntersection.Distance != math32.MaxFloat32 {
			scatteredColor.R += bvhIntersection.Color.R
			scatteredColor.G += bvhIntersection.Color.G
			scatteredColor.B += bvhIntersection.Color.B
		}
	}

	if samples > 0 {
		s := float32(samples)
		scatteredColor = ColorFloat32{
			R: scatteredColor.R / s,
			G: scatteredColor.G / s,
			B: scatteredColor.B / s,
		}
	}

	// Calculate reflection direction using Fresnel
	reflectDir := lightDir.Mul(-1).Reflect(intersection.Normal)
	reflectRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	tempIntersection, _ := reflectRay.IntersectBVH_Texture(BVH, textureMap)

	// Apply Fresnel to reflection color
	directReflectionColor := ColorFloat32{
		R: tempIntersection.Color.R * fresnel,
		G: tempIntersection.Color.G * fresnel,
		B: tempIntersection.Color.B * fresnel,
		A: intersection.Color.A,
	}

	// Shadow calculation
	shadowRay := Ray{
		origin:    rayOriginOffset,
		direction: lightDir,
	}
	_, inShadow := shadowRay.IntersectBVH(BVH)

	// Calculate specular using GGX distribution
	var lightIntensity float32 = 0.005
	if !inShadow {
		lightIntensity = light.intensity * NdotL
	}

	specularIntensity := distribution * fresnel * lightIntensity * intersection.specular

	specularColor := ColorFloat32{
		R: specularIntensity * light.Color[0],
		G: specularIntensity * light.Color[1],
		B: specularIntensity * light.Color[2],
	}

	// Calculate diffuse contribution
	diffuseFactor := (1.0 - fresnel) * (1.0 / math32.Pi)
	diffuseColor := ColorFloat32{
		R: intersection.Color.R * diffuseFactor * NdotL * lightIntensity,
		G: intersection.Color.G * diffuseFactor * NdotL * lightIntensity,
		B: intersection.Color.B * diffuseFactor * NdotL * lightIntensity,
	}

	// Combine direct and indirect lighting
	finalColor := ColorFloat32{
		R: light.Color[0] * (diffuseColor.R + specularColor.R + (directReflectionColor.R * intersection.directToScatter) + (scatteredColor.R * (1 - intersection.directToScatter))),
		G: light.Color[1] * (diffuseColor.G + specularColor.G + (directReflectionColor.G * intersection.directToScatter) + (scatteredColor.G * (1 - intersection.directToScatter))),
		B: light.Color[2] * (diffuseColor.B + specularColor.B + (directReflectionColor.B * intersection.directToScatter) + (scatteredColor.B * (1 - intersection.directToScatter))),
		A: intersection.Color.A,
	}

	// Calculate bounced contribution
	bounceRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	bouncedColor, _ := TraceRayV3AdvanceTexture(bounceRay, depth-1, light, samples, textureMap, BVH)

	// Final color composition with energy conservation
	return ColorFloat32{
		R: finalColor.R*(1.0-fresnel) + bouncedColor.R*fresnel,
		G: finalColor.G*(1.0-fresnel) + bouncedColor.G*fresnel,
		B: finalColor.B*(1.0-fresnel) + bouncedColor.B*fresnel,
		A: finalColor.A,
	}, intersection.Normal
}

func TraceRayV4AdvanceTexture(ray Ray, depth int, light Light, samples int, textureMap *[128]Texture, BVH *BVHLeanNode) (c ColorFloat32, normal Vector) {
	if depth <= 0 {
		return ColorFloat32{}, Vector{}
	}

	intersection, intersect := ray.IntersectBVHLean_Texture(BVH, textureMap)
	if !intersect {
		return ColorFloat32{}, Vector{}
	}

	viewDir := ray.origin.Sub(intersection.PointOfIntersection).Normalize()
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	halfwayDir := lightDir.Add(viewDir).Normalize()

	// Calculate important dot products
	NdotL := math32.Max(0.0, intersection.Normal.Dot(lightDir))
	NdotV := math32.Max(0.0, intersection.Normal.Dot(viewDir))
	NdotH := math32.Max(0.0, intersection.Normal.Dot(halfwayDir))

	// Calculate Fresnel term
	F0 := intersection.Metallic // Base reflectivity for non-metals
	fresnel := FresnelSchlick(NdotV, F0)

	// Calculate roughness-based distribution
	distribution := GGXDistribution(NdotH, intersection.Roughness)

	// Scatter calculation using hemisphere sampling
	var scatteredColor ColorFloat32
	rayOriginOffset := intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.01))

	for i := 0; i < samples; i++ {
		scatterDirection := SampleHemisphere(intersection.Normal)
		scatterDirection = scatterDirection.Perturb(intersection.Normal, intersection.Roughness)

		scatterRay := Ray{
			origin:    rayOriginOffset,
			direction: scatterDirection.Normalize(),
		}

		if bvhIntersection, scatterIntersect := scatterRay.IntersectBVHLean_Texture(BVH, textureMap); scatterIntersect {
			scatteredColor.R += bvhIntersection.Color.R
			scatteredColor.G += bvhIntersection.Color.G
			scatteredColor.B += bvhIntersection.Color.B
		}
	}

	if samples > 0 {
		s := float32(samples)
		scatteredColor = ColorFloat32{
			R: scatteredColor.R / s,
			G: scatteredColor.G / s,
			B: scatteredColor.B / s,
		}
	}

	// Calculate reflection direction using Fresnel
	reflectDir := lightDir.Mul(-1).Reflect(intersection.Normal)
	reflectRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	tempIntersection, _ := reflectRay.IntersectBVHLean_Texture(BVH, textureMap)

	// Apply Fresnel to reflection color
	directReflectionColor := ColorFloat32{
		R: tempIntersection.Color.R * fresnel,
		G: tempIntersection.Color.G * fresnel,
		B: tempIntersection.Color.B * fresnel,
		A: intersection.Color.A,
	}

	// Shadow calculation
	shadowRay := Ray{
		origin:    rayOriginOffset,
		direction: lightDir,
	}
	_, inShadow := shadowRay.IntersectBVHLean_Texture(BVH, textureMap)

	// Calculate specular using GGX distribution
	var lightIntensity float32 = 0.005
	if !inShadow {
		lightIntensity = light.intensity * NdotL
	}

	specularIntensity := distribution * fresnel * lightIntensity * intersection.specular

	specularColor := ColorFloat32{
		R: specularIntensity * light.Color[0],
		G: specularIntensity * light.Color[1],
		B: specularIntensity * light.Color[2],
	}

	// Calculate diffuse contribution
	diffuseFactor := (1.0 - fresnel) * (1.0 / math32.Pi)
	diffuseColor := ColorFloat32{
		R: intersection.Color.R * diffuseFactor * NdotL * lightIntensity,
		G: intersection.Color.G * diffuseFactor * NdotL * lightIntensity,
		B: intersection.Color.B * diffuseFactor * NdotL * lightIntensity,
	}

	// Combine direct and indirect lighting
	finalColor := ColorFloat32{
		R: light.Color[0] * (diffuseColor.R + specularColor.R + (directReflectionColor.R * intersection.directToScatter) + (scatteredColor.R * (1 - intersection.directToScatter))),
		G: light.Color[1] * (diffuseColor.G + specularColor.G + (directReflectionColor.G * intersection.directToScatter) + (scatteredColor.G * (1 - intersection.directToScatter))),
		B: light.Color[2] * (diffuseColor.B + specularColor.B + (directReflectionColor.B * intersection.directToScatter) + (scatteredColor.B * (1 - intersection.directToScatter))),
		A: intersection.Color.A,
	}

	// Calculate bounced contribution
	bounceRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	bouncedColor, _ := TraceRayV4AdvanceTexture(bounceRay, depth-1, light, samples, textureMap, BVH)

	// Final color composition with energy conservation
	invFresnel := 1.0 - fresnel
	return ColorFloat32{
		R: finalColor.R*invFresnel + bouncedColor.R*fresnel,
		G: finalColor.G*invFresnel + bouncedColor.G*fresnel,
		B: finalColor.B*invFresnel + bouncedColor.B*fresnel,
		A: finalColor.A,
	}, intersection.Normal
}

func TraceRayV4AdvanceTextureLean(ray Ray, depth int, light Light, samples int, textureMap *[128]Texture, BVH *BVHLeanNode) (c ColorFloat32) {
	if depth <= 0 {
		return ColorFloat32{}
	}

	intersection, intersect := ray.IntersectBVHLean_TextureLean(BVH, textureMap)
	if !intersect {
		return ColorFloat32{}
	}

	viewDir := ray.origin.Sub(intersection.PointOfIntersection).Normalize()
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	halfwayDir := lightDir.Add(viewDir).Normalize()

	// Calculate important dot products
	NdotL := math32.Max(0.0, intersection.Normal.Dot(lightDir))
	NdotV := math32.Max(0.0, intersection.Normal.Dot(viewDir))
	NdotH := math32.Max(0.0, intersection.Normal.Dot(halfwayDir))

	// Calculate Fresnel term
	// F0 :=  // Base reflectivity for non-metals
	fresnel := FresnelSchlick(NdotV, intersection.Metallic)

	// Calculate roughness-based distribution
	distribution := GGXDistribution(NdotH, intersection.Roughness)

	// Scatter calculation using hemisphere sampling
	var scatteredColor ColorFloat32
	rayOriginOffset := intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.01))

	for i := 0; i < samples; i++ {
		scatterDirection := SampleHemisphere(intersection.Normal)
		scatterDirection = scatterDirection.Perturb(intersection.Normal, intersection.Roughness)

		scatterRay := Ray{
			origin:    rayOriginOffset,
			direction: scatterDirection.Normalize(),
		}

		if bvhIntersection, scatterIntersect := scatterRay.IntersectBVHLean_TextureLean(BVH, textureMap); scatterIntersect {
			scatteredColor.R += bvhIntersection.Color.R
			scatteredColor.G += bvhIntersection.Color.G
			scatteredColor.B += bvhIntersection.Color.B
		}
	}

	if samples > 0 {
		s := float32(samples)
		scatteredColor = ColorFloat32{
			R: scatteredColor.R / s,
			G: scatteredColor.G / s,
			B: scatteredColor.B / s,
		}
	}

	// Calculate reflection direction using Fresnel
	reflectDir := lightDir.Mul(-1).Reflect(intersection.Normal)
	reflectRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	tempIntersection, _ := reflectRay.IntersectBVHLean_TextureLean(BVH, textureMap)

	// Apply Fresnel to reflection color
	directReflectionColor := ColorFloat32{
		R: tempIntersection.Color.R * fresnel,
		G: tempIntersection.Color.G * fresnel,
		B: tempIntersection.Color.B * fresnel,
		A: intersection.Color.A,
	}

	// Shadow calculation
	shadowRay := Ray{
		origin:    rayOriginOffset,
		direction: lightDir,
	}
	_, inShadow := shadowRay.IntersectBVHLean_TextureLean(BVH, textureMap)

	// Calculate specular using GGX distribution
	var lightIntensity float32 = 0.005
	if !inShadow {
		lightIntensity = light.intensity * NdotL
	}

	specularIntensity := distribution * fresnel * lightIntensity * intersection.specular

	specularColor := ColorFloat32{
		R: specularIntensity * light.Color[0],
		G: specularIntensity * light.Color[1],
		B: specularIntensity * light.Color[2],
	}

	// Calculate diffuse contribution
	diffuseFactor := (1.0 - fresnel) * (1.0 / math32.Pi)
	diffuseColor := ColorFloat32{
		R: intersection.Color.R * diffuseFactor * NdotL * lightIntensity,
		G: intersection.Color.G * diffuseFactor * NdotL * lightIntensity,
		B: intersection.Color.B * diffuseFactor * NdotL * lightIntensity,
	}

	// Combine direct and indirect lighting
	finalColor := ColorFloat32{
		R: light.Color[0] * (diffuseColor.R + specularColor.R + (directReflectionColor.R * intersection.directToScatter) + (scatteredColor.R * (1 - intersection.directToScatter))),
		G: light.Color[1] * (diffuseColor.G + specularColor.G + (directReflectionColor.G * intersection.directToScatter) + (scatteredColor.G * (1 - intersection.directToScatter))),
		B: light.Color[2] * (diffuseColor.B + specularColor.B + (directReflectionColor.B * intersection.directToScatter) + (scatteredColor.B * (1 - intersection.directToScatter))),
		A: intersection.Color.A,
	}

	// Calculate bounced contribution
	bounceRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	bouncedColor := TraceRayV4AdvanceTextureLean(bounceRay, depth-1, light, samples, textureMap, BVH)

	// Final color composition with energy conservation
	invFresnel := 1.0 - fresnel
	return ColorFloat32{
		R: finalColor.R*invFresnel + bouncedColor.R*fresnel,
		G: finalColor.G*invFresnel + bouncedColor.G*fresnel,
		B: finalColor.B*invFresnel + bouncedColor.B*fresnel,
		A: finalColor.A,
	}
}

func TraceRayV4AdvanceTextureLeanOptim(ray Ray, depth int, light Light, samples int, textureMap *[128]Texture, BVH *BVHLeanNode) (c ColorFloat32) {
	if depth <= 0 {
		return ColorFloat32{}
	}

	intersection, intersect := ray.IntersectBVHLean_TextureLeanOptim(BVH, textureMap)
	if !intersect {
		return ColorFloat32{}
	}

	viewDir := ray.origin.Sub(intersection.PointOfIntersection).Normalize()
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	halfwayDir := lightDir.Add(viewDir).Normalize()

	// Calculate important dot products
	NdotL := math32.Max(0.0, intersection.Normal.Dot(lightDir))
	NdotV := math32.Max(0.0, intersection.Normal.Dot(viewDir))
	NdotH := math32.Max(0.0, intersection.Normal.Dot(halfwayDir))

	// Calculate Fresnel term
	// F0 :=  // Base reflectivity for non-metals
	fresnel := FresnelSchlick(NdotV, intersection.Metallic)

	// Calculate roughness-based distribution
	distribution := GGXDistribution(NdotH, intersection.Roughness)

	// Scatter calculation using hemisphere sampling
	var scatteredColor ColorFloat32
	rayOriginOffset := intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.01))

	for i := 0; i < samples; i++ {
		scatterDirection := SampleHemisphere(intersection.Normal)
		scatterDirection = scatterDirection.Perturb(intersection.Normal, intersection.Roughness)

		scatterRay := Ray{
			origin:    rayOriginOffset,
			direction: scatterDirection.Normalize(),
		}

		if bvhIntersection, scatterIntersect := scatterRay.IntersectBVHLean_TextureLeanOptim(BVH, textureMap); scatterIntersect {
			scatteredColor.R += bvhIntersection.Color.R
			scatteredColor.G += bvhIntersection.Color.G
			scatteredColor.B += bvhIntersection.Color.B
		}
	}

	if samples > 0 {
		s := float32(samples)
		scatteredColor = ColorFloat32{
			R: scatteredColor.R / s,
			G: scatteredColor.G / s,
			B: scatteredColor.B / s,
		}
	}

	// Calculate reflection direction using Fresnel
	reflectDir := lightDir.Mul(-1).Reflect(intersection.Normal)
	reflectRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	tempIntersection, _ := reflectRay.IntersectBVHLean_TextureLeanOptim(BVH, textureMap)

	// Apply Fresnel to reflection color
	directReflectionColor := ColorFloat32{
		R: tempIntersection.Color.R * fresnel,
		G: tempIntersection.Color.G * fresnel,
		B: tempIntersection.Color.B * fresnel,
		A: intersection.Color.A,
	}

	// Shadow calculation
	shadowRay := Ray{
		origin:    rayOriginOffset,
		direction: lightDir,
	}
	_, inShadow := shadowRay.IntersectBVHLean_TextureLeanOptim(BVH, textureMap)

	// Calculate specular using GGX distribution
	var lightIntensity float32 = 0.005
	if !inShadow {
		lightIntensity = light.intensity * NdotL
	}

	specularIntensity := distribution * fresnel * lightIntensity * intersection.specular

	specularColor := ColorFloat32{
		R: specularIntensity * light.Color[0],
		G: specularIntensity * light.Color[1],
		B: specularIntensity * light.Color[2],
	}

	// Calculate diffuse contribution
	diffuseFactor := (1.0 - fresnel) * (1.0 / math32.Pi)
	diffuseColor := ColorFloat32{
		R: intersection.Color.R * diffuseFactor * NdotL * lightIntensity,
		G: intersection.Color.G * diffuseFactor * NdotL * lightIntensity,
		B: intersection.Color.B * diffuseFactor * NdotL * lightIntensity,
	}

	// Combine direct and indirect lighting
	finalColor := ColorFloat32{
		R: light.Color[0] * (diffuseColor.R + specularColor.R + (directReflectionColor.R * intersection.directToScatter) + (scatteredColor.R * (1 - intersection.directToScatter))),
		G: light.Color[1] * (diffuseColor.G + specularColor.G + (directReflectionColor.G * intersection.directToScatter) + (scatteredColor.G * (1 - intersection.directToScatter))),
		B: light.Color[2] * (diffuseColor.B + specularColor.B + (directReflectionColor.B * intersection.directToScatter) + (scatteredColor.B * (1 - intersection.directToScatter))),
		A: intersection.Color.A,
	}

	// Calculate bounced contribution
	bounceRay := Ray{origin: rayOriginOffset, direction: reflectDir}
	bouncedColor := TraceRayV4AdvanceTextureLeanOptim(bounceRay, depth-1, light, samples, textureMap, BVH)

	// Final color composition with energy conservation
	invFresnel := 1.0 - fresnel
	return ColorFloat32{
		R: finalColor.R*invFresnel + bouncedColor.R*fresnel,
		G: finalColor.G*invFresnel + bouncedColor.G*fresnel,
		B: finalColor.B*invFresnel + bouncedColor.B*fresnel,
		A: finalColor.A,
	}
}

func (v Vector) LengthSquared() float32 {
	return v.x*v.x + v.y*v.y + v.z*v.z
}

func RandomInUnitSphere() Vector {
	for {
		p := Vector{
			rand.Float32()*2.0 - 1.0,
			rand.Float32()*2.0 - 1.0,
			rand.Float32()*2.0 - 1.0,
		}
		if p.LengthSquared() < 1.0 {
			return p.Normalize()
		}
	}
}

func (v Vector) Perturb(normal Vector, roughness float32) Vector {
	if roughness <= 0 {
		return v
	}

	// Generate a random perturbation vector
	randomVec := RandomInUnitSphere()

	// Scale the perturbation by roughness
	perturbation := randomVec.Mul(roughness)

	// Add the perturbation to the original vector
	result := v.Add(perturbation)

	// Ensure the perturbed vector is in the correct hemisphere
	if result.Dot(normal) < 0 {
		result = result.Mul(-1)
	}

	return result.Normalize()
}

func SampleHemisphere(normal Vector) Vector {
	u := rand.Float64()
	v := rand.Float32()

	r := float32(math.Sqrt(float64(1.0 - u*u)))
	theta := 2 * math32.Pi * v

	// test fast approximation of cos(theta) and sin(theta)
	// x := r * math32.Cos(theta)
	x := r * math32.Cos(theta)
	// y := r * math32.Sin(theta)
	y := r * math32.Sin(theta)
	z := u

	// Create tangent and bitangent vectors
	tangent := Vector{1.0, 0.0, 0.0}
	if math32.Abs(normal.x) > 0.1 {
		tangent = Vector{0.0, 1.0, 0.0}
	}
	tangent = tangent.Cross(normal).Normalize()
	bitangent := normal.Cross(tangent)

	// Transform sample to world space
	sampleDir := tangent.Mul(x).Add(bitangent.Mul(y)).Add(normal.Mul(float32(z)))

	return sampleDir.Normalize()
}

func TraceRayV2(ray Ray, depth int, light Light, samples int) ColorFloat32 {
	if depth <= 0 {
		return ColorFloat32{}
	}

	intersection, intersect := ray.IntersectBVH(BVH)
	if !intersect {
		return ColorFloat32{}
	}

	hitPoint := intersection.PointOfIntersection
	normal := intersection.Normal.Normalize()
	viewDir := ray.direction.Mul(-1).Normalize()

	// Accumulate color contributions
	var accumulatedColor ColorFloat32

	// --- Direct Illumination ---
	lightDir := light.Position.Sub(hitPoint).Normalize()
	reflectDir := lightDir.Mul(-1).Reflect(normal)

	// Apply roughness to reflection direction
	reflectDir = reflectDir.Perturb(normal, intersection.Roughness)

	// Shadow check
	shadowRay := Ray{
		origin:    hitPoint.Add(normal.Mul(0.001)),
		direction: lightDir,
	}
	_, inShadow := shadowRay.IntersectBVH(BVH)

	var lightIntensity float32
	if !inShadow {
		lightIntensity = light.intensity * math32.Max(0.0, normal.Dot(lightDir))
	}

	// Diffuse shading
	diffuse := intersection.Color.MulScalar(lightIntensity * (1.0 - intersection.reflection))

	// Specular shading (adjusted for roughness)
	specularExponent := intersection.specular * (1.0 - intersection.Roughness)
	specularFactor := math32.Pow(math32.Max(0.0, viewDir.Dot(reflectDir)), specularExponent)
	specular := ColorFloat32{
		R: light.Color[0],
		G: light.Color[1],
		B: light.Color[2],
		A: 1.0,
	}.MulScalar(specularFactor * light.intensity * intersection.reflection)

	// --- Indirect Illumination (Scattering) ---
	var scatteredColor ColorFloat32
	if samples > 0 {
		for i := 0; i < samples; i++ {
			// Sample hemisphere around the normal
			scatterDir := SampleHemisphere(normal)

			// Apply roughness to scattered direction
			scatterDir = scatterDir.Perturb(normal, intersection.Roughness)

			scatterRay := Ray{
				origin:    hitPoint.Add(normal.Mul(0.001)),
				direction: scatterDir,
			}

			bouncedColor := TraceRayV2(scatterRay, depth, light, 0)
			scatteredColor = scatteredColor.Add(bouncedColor)
		}
		scatteredColor = scatteredColor.MulScalar(1.0 / float32(samples))
	}

	// --- Reflection ---
	reflectDir = viewDir.Reflect(normal)
	reflectDir = reflectDir.Perturb(normal, intersection.Roughness)
	reflectRay := Ray{
		origin:    hitPoint.Add(normal.Mul(0.001)),
		direction: reflectDir,
	}
	reflectedColor := TraceRayV2(reflectRay, depth-1, light, samples)

	// --- Combine All Components ---
	accumulatedColor = diffuse
	accumulatedColor = accumulatedColor.Add(specular)
	accumulatedColor = accumulatedColor.Add(reflectedColor.MulScalar(intersection.reflection))
	accumulatedColor = accumulatedColor.Add(scatteredColor.MulScalar(intersection.directToScatter))

	return accumulatedColor
}

type object struct {
	triangles   []TriangleSimple
	BoundingBox [2]Vector
}

func ConvertObjectsToBVH(objects []object, maxDepth int) *BVHNode {
	Triangles := []TriangleSimple{}
	for _, object := range objects {
		Triangles = append(Triangles, object.triangles...)
	}
	return buildBVHNode(Triangles, 0, maxDepth)
}

type BVHNode struct {
	Left, Right *BVHNode
	BoundingBox [2]Vector
	Triangles   TriangleSimple
	active      bool
}

type BVHLeanNode struct {
	Left, Right  *BVHLeanNode
	TriangleBBOX TriangleBBOX
	active       bool
}

func (node *BVHLeanNode) FindIntersectionAndSetIt(id int32, ray Ray, textureMap *[128]Texture) {
	_, hit, n := ray.IntersectBVHLean_TextureWithNode(node, textureMap)
	if hit {
		// set Triangle id the id parameter using unsafe pointer
		idPtr := (*int32)(unsafe.Pointer(&n.TriangleBBOX.id))
		// fmt.Println("Before:", n.TriangleBBOX.id)
		*idPtr = id
		// fmt.Println("After:", n.TriangleBBOX.id)
	}
	return
}

// ConvertToLeanBVH converts a standard BVH to a lean BVH structure recursively
func (node *BVHNode) ConvertToLeanBVH() *BVHLeanNode {
	// Base case: nil node
	if node == nil {
		return nil
	}

	// Create new lean node
	leanNode := &BVHLeanNode{
		active: node.active,
	}

	// Handle leaf nodes (triangles)
	if node.active {
		leanNode.TriangleBBOX = TriangleBBOX{
			V1orBBoxMin: node.Triangles.v1,
			V2orBBoxMax: node.Triangles.v2,
			V3:          node.Triangles.v3,
			normal:      node.Triangles.Normal,
			id:          int32(1),
		}
	} else {
		// Handle internal nodes (bounding boxes)
		leanNode.TriangleBBOX = TriangleBBOX{
			V1orBBoxMin: node.BoundingBox[0],
			V2orBBoxMax: node.BoundingBox[1],
			V3:          Vector{},
			normal:      Vector{},
			id:          -1, // Use -1 to indicate internal node
		}

		if node.Left != nil {
			leanNode.Left = node.Left.ConvertToLeanBVH()
		}
		if node.Right != nil {
			leanNode.Right = node.Right.ConvertToLeanBVH()
		}

	}

	return leanNode
}

// Usage example:
func ConvertBVHToLean(rootNode *BVHNode) *BVHLeanNode {
	if rootNode == nil {
		return nil
	}
	return rootNode.ConvertToLeanBVH()
}

func (object *object) BuildBVH(maxDepth int) *BVHNode {
	return buildBVHNode(object.triangles, 0, maxDepth)
}

func (node *BVHNode) PointInBoundingBox(point Vector) (bool, TriangleSimple) {
	if point.x >= node.BoundingBox[0].x && point.x <= node.BoundingBox[1].x &&
		point.y >= node.BoundingBox[0].y && point.y <= node.BoundingBox[1].y &&
		point.z >= node.BoundingBox[0].z && point.z <= node.BoundingBox[1].z {

		if node.Left == nil && node.Right == nil {
			return true, node.Triangles
		}

		if node.Left != nil {
			hit, triangle := node.Left.PointInBoundingBox(point)
			if hit {
				return true, triangle
			}
		}

		if node.Right != nil {
			hit, triangle := node.Right.PointInBoundingBox(point)
			if hit {
				return true, triangle
			}
		}
	}
	return false, TriangleSimple{}
}

func calculateSurfaceArea(bbox [2]Vector) float32 {
	dx := bbox[1].x - bbox[0].x
	dy := bbox[1].y - bbox[0].y
	dz := bbox[1].z - bbox[0].z
	return 2 * (dx*dy + dy*dz + dz*dx)
}

// Call recursively and set triangle properties
func (node *BVHNode) SetPropertiesWithID(id uint8, reflection, specular, directToScatter, roughness, metallic float32) {
	if node.Left == nil && node.Right == nil && node.active && node.Triangles.id == id {
		node.Triangles.reflection = reflection
		node.Triangles.specular = specular
		node.Triangles.directToScatter = directToScatter
		node.Triangles.Roughness = roughness
		node.Triangles.Metallic = metallic
		return
	}

	if node.Left != nil {
		node.Left.SetPropertiesWithID(id, reflection, specular, directToScatter, roughness, metallic)
	}
	if node.Right != nil {
		node.Right.SetPropertiesWithID(id, reflection, specular, directToScatter, roughness, metallic)
	}
}

func buildBVHNode(triangles []TriangleSimple, depth int, maxDepth int) *BVHNode {
	if len(triangles) == 0 {
		return nil
	}

	// Calculate the bounding box of the node
	boundingBox := [2]Vector{
		{math32.MaxFloat32, math32.MaxFloat32, math32.MaxFloat32},
		{-math32.MaxFloat32, -math32.MaxFloat32, -math32.MaxFloat32},
	}

	for _, triangle := range triangles {
		minBox, maxBox := triangle.CalculateBoundingBox()
		boundingBox[0].x = math32.Min(boundingBox[0].x, minBox.x)
		boundingBox[0].y = math32.Min(boundingBox[0].y, minBox.y)
		boundingBox[0].z = math32.Min(boundingBox[0].z, minBox.z)

		boundingBox[1].x = math32.Max(boundingBox[1].x, maxBox.x)
		boundingBox[1].y = math32.Max(boundingBox[1].y, maxBox.y)
		boundingBox[1].z = math32.Max(boundingBox[1].z, maxBox.z)
	}

	// If the node is a leaf or we've reached the maximum depth
	if len(triangles) <= 1 || depth >= maxDepth {
		// Allocate the slice with the exact capacity needed
		// trianglesSimple := make([]TriangleSimple, len(triangles))
		// for i, triangle := range triangles {

		node := &BVHNode{
			BoundingBox: boundingBox,
			Triangles: TriangleSimple{
				v1: triangles[0].v1,
				v2: triangles[0].v2,
				v3: triangles[0].v3,
				color: ColorFloat32{
					R: triangles[0].color.R,
					G: triangles[0].color.G,
					B: triangles[0].color.B,
					A: triangles[0].color.A,
				},
				Normal:          triangles[0].Normal,
				reflection:      triangles[0].reflection,
				specular:        triangles[0].specular,
				directToScatter: 0.5,
				Roughness:       triangles[0].Roughness,
				Metallic:        triangles[0].Metallic,
				id:              uint8(1),
			},
			active: true,
		}
		return node
	}

	// Surface Area Heuristics (SAH) to find the best split
	bestCost := float32(math32.MaxFloat32)
	bestSplit := -1
	bestAxis := 0

	for axis := 0; axis < 3; axis++ {
		// Sort the triangles along the current axis
		switch axis {
		case 0:
			sort.Slice(triangles, func(i, j int) bool {
				iMinBox, _ := triangles[i].CalculateBoundingBox()
				jMinBox, _ := triangles[j].CalculateBoundingBox()
				return iMinBox.x < jMinBox.x
			})
		case 1:
			sort.Slice(triangles, func(i, j int) bool {
				iMinBox, _ := triangles[i].CalculateBoundingBox()
				jMinBox, _ := triangles[j].CalculateBoundingBox()
				return iMinBox.y < jMinBox.y
			})
		case 2:
			sort.Slice(triangles, func(i, j int) bool {
				iMinBox, _ := triangles[i].CalculateBoundingBox()
				jMinBox, _ := triangles[j].CalculateBoundingBox()
				return iMinBox.z < jMinBox.z
			})
		}

		// Compute surface area for all possible splits
		for i := 1; i < len(triangles); i++ {
			leftBBox := [2]Vector{
				{math32.MaxFloat32, math32.MaxFloat32, math32.MaxFloat32},
				{-math32.MaxFloat32, -math32.MaxFloat32, -math32.MaxFloat32},
			}
			rightBBox := [2]Vector{
				{math32.MaxFloat32, math32.MaxFloat32, math32.MaxFloat32},
				{-math32.MaxFloat32, -math32.MaxFloat32, -math32.MaxFloat32},
			}

			for j := 0; j < i; j++ {
				jMinBox, jMaxBox := triangles[j].CalculateBoundingBox()
				leftBBox[0].x = math32.Min(leftBBox[0].x, jMinBox.x)
				leftBBox[0].y = math32.Min(leftBBox[0].y, jMinBox.y)
				leftBBox[0].z = math32.Min(leftBBox[0].z, jMinBox.z)
				leftBBox[1].x = math32.Max(leftBBox[1].x, jMaxBox.x)
				leftBBox[1].y = math32.Max(leftBBox[1].y, jMaxBox.y)
				leftBBox[1].z = math32.Max(leftBBox[1].z, jMaxBox.z)
			}

			for j := i; j < len(triangles); j++ {
				jMinBox, jMaxBox := triangles[j].CalculateBoundingBox()
				rightBBox[0].x = math32.Min(rightBBox[0].x, jMinBox.x)
				rightBBox[0].y = math32.Min(rightBBox[0].y, jMinBox.y)
				rightBBox[0].z = math32.Min(rightBBox[0].z, jMinBox.z)
				rightBBox[1].x = math32.Max(rightBBox[1].x, jMaxBox.x)
				rightBBox[1].y = math32.Max(rightBBox[1].y, jMaxBox.y)
				rightBBox[1].z = math32.Max(rightBBox[1].z, jMaxBox.z)
			}

			// Calculate the SAH cost for this split
			cost := float32(i)*calculateSurfaceArea(leftBBox) + float32(len(triangles)-i)*calculateSurfaceArea(rightBBox)
			if cost < bestCost {
				bestCost = cost
				bestSplit = i
				bestAxis = axis
			}
		}
	}

	// Sort triangles along the best axis before splitting
	switch bestAxis {
	case 0:
		sort.Slice(triangles, func(i, j int) bool {
			jMinBox, _ := triangles[j].CalculateBoundingBox()
			iMinBox, _ := triangles[i].CalculateBoundingBox()
			return iMinBox.x < jMinBox.x
			// return triangles[i].BoundingBox[0].x < triangles[j].BoundingBox[0].x
		})
	case 1:
		sort.Slice(triangles, func(i, j int) bool {
			jMinBox, _ := triangles[j].CalculateBoundingBox()
			iMinBox, _ := triangles[i].CalculateBoundingBox()
			// return triangles[i].BoundingBox[0].y < triangles[j].BoundingBox[0].y
			return iMinBox.y < jMinBox.y
		})
	case 2:
		sort.Slice(triangles, func(i, j int) bool {
			jMinBox, _ := triangles[j].CalculateBoundingBox()
			iMinBox, _ := triangles[i].CalculateBoundingBox()
			// return triangles[i].BoundingBox[0].z < triangles[j].BoundingBox[0].z
			return iMinBox.z < jMinBox.z
		})
	}

	// Create the BVH node with the best split
	node := &BVHNode{BoundingBox: boundingBox, active: false}
	node.Left = buildBVHNode(triangles[:bestSplit], depth+1, maxDepth)
	node.Right = buildBVHNode(triangles[bestSplit:], depth+1, maxDepth)

	return node
}

func (object *object) Move(v Vector) {
	for i := range object.triangles {
		object.triangles[i].v1 = object.triangles[i].v1.Add(v)
		object.triangles[i].v2 = object.triangles[i].v2.Add(v)
		object.triangles[i].v3 = object.triangles[i].v3.Add(v)
		object.triangles[i].CalculateBoundingBox()
	}
	object.CalculateBoundingBox()
}

func (object *object) Rotate(xAngle float32, yAngle float32, zAngle float32) {
	for i := range object.triangles {
		object.triangles[i].Rotate(xAngle, yAngle, zAngle)
		object.triangles[i].CalculateBoundingBox()
	}
	object.CalculateBoundingBox()
}

func (object *object) Scale(scalar float32) {
	for i := range object.triangles {
		object.triangles[i].v1 = object.triangles[i].v1.Mul(scalar)
		object.triangles[i].v2 = object.triangles[i].v2.Mul(scalar)
		object.triangles[i].v3 = object.triangles[i].v3.Mul(scalar)
		object.triangles[i].CalculateBoundingBox()
	}
	object.CalculateBoundingBox()
}

func CreateObject(triangles []TriangleSimple) *object {
	object := &object{
		triangles: triangles,
		BoundingBox: [2]Vector{
			{math32.MaxFloat32, math32.MaxFloat32, math32.MaxFloat32},
			{-math32.MaxFloat32, -math32.MaxFloat32, -math32.MaxFloat32},
		},
	}
	object.CalculateBoundingBox()
	return object
}

func (object *object) CalculateNormals() {
	for i := range object.triangles {
		object.triangles[i].CalculateNormal()
	}
}

func (object *object) CalculateBoundingBox() {
	for _, triangle := range object.triangles {
		// Update minimum coordinates (BoundingBox[0])
		minBox, maxBox := triangle.CalculateBoundingBox()
		object.BoundingBox[0].x = math32.Min(object.BoundingBox[0].x, minBox.x)
		object.BoundingBox[0].y = math32.Min(object.BoundingBox[0].y, minBox.y)
		object.BoundingBox[0].z = math32.Min(object.BoundingBox[0].z, minBox.z)

		// Update maximum coordinates (BoundingBox[1])
		object.BoundingBox[1].x = math32.Max(object.BoundingBox[1].x, maxBox.x)
		object.BoundingBox[1].y = math32.Max(object.BoundingBox[1].y, maxBox.y)
		object.BoundingBox[1].z = math32.Max(object.BoundingBox[1].z, maxBox.z)
	}
}

func GenerateRandomSpheres(numSpheres int) []object {
	spheres := make([]object, numSpheres)
	for i := 0; i < numSpheres; i++ {
		radius := rand.Float32()*50 + 10
		color := ColorFloat32{float32(rand.Intn(255)), float32(rand.Intn(255)), float32(rand.Intn(255)), 255}
		reflection := rand.Float32()
		position := Vector{rand.Float32()*400 - 200, rand.Float32()*400 - 200, rand.Float32()*400 - 200}
		specular := rand.Float32()
		sphere := CreateSphere(position, radius, color, reflection, specular)
		spheres[i] = *CreateObject(sphere)
	}
	return spheres
}

func GenerateRandomCubes(numCubes int) []object {
	cubes := make([]object, numCubes)
	for i := 0; i < numCubes; i++ {
		size := rand.Float32()*50 + 10
		color := ColorFloat32{float32(rand.Intn(255)), float32(rand.Intn(255)), float32(rand.Intn(255)), 255}
		reflection := rand.Float32()
		specular := rand.Float32()
		position := Vector{rand.Float32()*400 - 200, rand.Float32()*400 - 200, rand.Float32()*400 - 200}
		cube := CreateCube(position, size, color, reflection, specular)
		obj := CreateObject(cube)
		obj.Rotate(rand.Float32()*math.Pi, rand.Float32()*math.Pi, rand.Float32()*math.Pi)
		cubes[i] = *obj
	}
	return cubes
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

	return tMin < math32.Inf(1) && tMax > 0
}

func (object *object) ConvertToTriangles() []TriangleSimple {
	triangles := []TriangleSimple{}
	triangles = append(triangles, object.triangles...)
	return triangles
}

// PositionOnSphere calculates the 3D position on a unit sphere given two angles.
func PositionOnSphere(theta, phi float32) Vector {
	x := math32.Sin(phi) * math32.Cos(theta)
	y := math32.Sin(phi) * math32.Sin(theta)
	z := math32.Cos(phi)
	return Vector{x: x, y: y, z: z}
}

var FOVRadians = FOV * math32.Pi / 180

func PrecomputeScreenSpaceCoordinatesSphere(camera Camera) {
	// Calculate corners
	topLeft := PositionOnSphere(camera.xAxis, camera.yAxis)
	topRight := PositionOnSphere(camera.xAxis+FOVRadians, camera.yAxis)
	bottomLeft := PositionOnSphere(camera.xAxis, camera.yAxis+FOVRadians)

	// Calculate steps
	xStep := Vector{
		x: (topRight.x - topLeft.x) / float32(screenWidth-1),
		y: (topRight.y - topLeft.y) / float32(screenWidth-1),
		z: (topRight.z - topLeft.z) / float32(screenWidth-1),
	}
	yStep := Vector{
		x: (bottomLeft.x - topLeft.x) / float32(screenHeight-1),
		y: (bottomLeft.y - topLeft.y) / float32(screenHeight-1),
		z: (bottomLeft.z - topLeft.z) / float32(screenHeight-1),
	}

	// Interpolate
	for width := 0; width < screenWidth; width++ {
		for height := 0; height < screenHeight; height++ {
			ScreenSpaceCoordinates[width][height] = Vector{
				x: topLeft.x + float32(width)*xStep.x + float32(height)*yStep.x,
				y: topLeft.y + float32(width)*xStep.y + float32(height)*yStep.y,
				z: topLeft.z + float32(width)*xStep.z + float32(height)*yStep.z,
			}
		}
	}
}

// func PrecomputeScreenSpaceCoordinatesSphereOptimalized(camera Camera) {
// 	// Calculate corners
// 	topLeft := PositionOnSphere(camera.xAxis, camera.yAxis)
// 	topRight := PositionOnSphere(camera.xAxis+FOVRadians, camera.yAxis)
// 	bottomLeft := PositionOnSphere(camera.xAxis, camera.yAxis+FOVRadians)

// 	// Calculate steps
// 	xStep := Vector{
// 		x: (topRight.x - topLeft.x) / float32(screenWidth-1),
// 		y: (topRight.y - topLeft.y) / float32(screenWidth-1),
// 		z: (topRight.z - topLeft.z) / float32(screenWidth-1),
// 	}
// 	yStep := Vector{
// 		x: (bottomLeft.x - topLeft.x) / float32(screenHeight-1),
// 		y: (bottomLeft.y - topLeft.y) / float32(screenHeight-1),
// 		z: (bottomLeft.z - topLeft.z) / float32(screenHeight-1),
// 	}

// 	// Interpolate
// 	for width := 0; width < screenWidth; width++ {
// 		for height := 0; height < screenHeight; height += 4 {
// 			x := topLeft.x + float32(width)*xStep.x
// 			y := topLeft.y + float32(width)*xStep.y
// 			z := topLeft.z + float32(width)*xStep.z
// 			ScreenSpaceCoordinates[width][height] = Vector{
// 				x: x + float32(height)*yStep.x,
// 				y: y + float32(height)*yStep.y,
// 				z: z + float32(height)*yStep.z,
// 			}
// 			ScreenSpaceCoordinates[width][height+1] = Vector{
// 				x: x + float32(height+1)*yStep.x,
// 				y: y + float32(height+1)*yStep.y,
// 				z: z + float32(height+1)*yStep.z,
// 			}
// 			ScreenSpaceCoordinates[width][height+2] = Vector{
// 				x: x + float32(height+2)*yStep.x,
// 				y: y + float32(height+2)*yStep.y,
// 				z: z + float32(height+2)*yStep.z,
// 			}
// 			ScreenSpaceCoordinates[width][height+3] = Vector{
// 				x: x + float32(height+3)*yStep.x,
// 				y: y + float32(height+3)*yStep.y,
// 				z: z + float32(height+3)*yStep.z,
// 			}
// 		}
// 	}
// }

func DrawRays(camera Camera, light Light, scaling int, samples int, depth int, subImages []*ebiten.Image) {
	var wg sync.WaitGroup

	// Create a pool of worker goroutines, each handling a portion of the image
	for i := 0; i < numCPU; i++ {
		wg.Add(1)
		go func(startY int, endIndex int, subImage *ebiten.Image) {
			defer wg.Done()
			yRow := 0
			width, height := subImage.Bounds().Dx(), subImage.Bounds().Dy()
			imageSize := width * height * 4
			pixelBuffer := make([]uint8, imageSize)
			for y := startY; y < endIndex; y += scaling {
				xColumn := 0
				for x := 0; x < screenWidth; x += scaling {
					rayDir := ScreenSpaceCoordinates[x][y]
					c := TraceRay(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples)

					// Write the pixel color to the pixel buffer
					index := ((yRow*width + xColumn) * 4)
					pixelBuffer[index] = clampUint8(c.R)
					pixelBuffer[index+1] = clampUint8(c.G)
					pixelBuffer[index+2] = clampUint8(c.B)
					pixelBuffer[index+3] = clampUint8(c.A)
					xColumn++
					// // Set the pixel color in the sub-image
					// subImage.Set(x/scaling, yRow, c)
				}
				yRow++
			}
			subImage.WritePixels(pixelBuffer)
		}(i*rowSize, (i+1)*rowSize, subImages[i])
	}
	if performanceOptions.Selected == 0 {
		// Wait for all workers to finish
		wg.Wait()
	}
}

func DrawRaysBlock(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImage, performance bool) {
	var wg sync.WaitGroup
	for i := 0; i < len(blocks); i++ {
		wg.Add(1)
		go func(blockIndex int) {
			defer wg.Done()
			block := blocks[blockIndex]
			for y := block.startY; y < block.endY; y += 1 {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x += 1 {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c := TraceRay(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples)

					// Write the pixel color to the pixel buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.pixelBuffer[index] = clampUint8(c.R)
					block.pixelBuffer[index+1] = clampUint8(c.G)
					block.pixelBuffer[index+2] = clampUint8(c.B)
					block.pixelBuffer[index+3] = clampUint8(c.A)
				}
			}
			block.image.WritePixels(block.pixelBuffer)
		}(i)
	}

	if !performance {
		wg.Wait()
	}
}

const (
	logMode = 2
	linMode = 3
)

// Helper function to clamp values between 0 and 255
func clampToUint8(value float32) uint8 {
	return uint8(math32.Min(math32.Max(value, 0), 255))
}

func ColorGradeLogarithmic(colors []float32, maxRed, maxGreen, maxBlue, gamma float32) []uint8 {
	// Ensure we do not divide by zero.
	if maxRed < 1 {
		maxRed = 1
	}
	if maxGreen < 1 {
		maxGreen = 1
	}
	if maxBlue < 1 {
		maxBlue = 1
	}

	// Precompute constant values
	logMaxRed := math32.Log(maxRed + 1)
	logMaxGreen := math32.Log(maxGreen + 1)
	logMaxBlue := math32.Log(maxBlue + 1)
	// Combine the two gamma corrections:
	//   pow( pow(x, gamma), 1/(gamma-1) ) == pow(x, gamma/(gamma-1) )
	gammaExp := gamma / (gamma - 1)

	// Allocate output slice.
	out := make([]uint8, len(colors))

	// Process each pixel (assuming RGBA order)
	for i := 0; i < len(colors); i += 4 {
		// Apply logarithmic tone mapping
		red := math32.Log(colors[i]*maxRed+1) / logMaxRed
		green := math32.Log(colors[i+1]*maxGreen+1) / logMaxGreen
		blue := math32.Log(colors[i+2]*maxBlue+1) / logMaxBlue

		// Apply combined gamma correction
		red = math32.Pow(red, gammaExp)
		green = math32.Pow(green, gammaExp)
		blue = math32.Pow(blue, gammaExp)

		// Scale to 0-255 and clamp
		out[i] = clampToUint8(red)
		out[i+1] = clampToUint8(green)
		out[i+2] = clampToUint8(blue)
		out[i+3] = 255 // Alpha channel
	}

	return out
}

func ColorGradeLinear(colors []float32, maxRed, maxGreen, maxBlue, gamma float32) []uint8 {
	// Ensure we do not divide by zero.
	if maxRed < 1 {
		maxRed = 1
	}
	if maxGreen < 1 {
		maxGreen = 1
	}
	if maxBlue < 1 {
		maxBlue = 1
	}

	// Precompute the gamma exponent for efficiency.
	gammaExp := 1 / (gamma - 1)
	out := make([]uint8, len(colors))

	// Process each pixel (assuming RGBA order)
	for i := 0; i < len(colors); i += 4 {
		red := colors[i] / maxRed
		green := colors[i+1] / maxGreen
		blue := colors[i+2] / maxBlue

		// Apply gamma correction
		red = math32.Pow(red, gammaExp)
		green = math32.Pow(green, gammaExp)
		blue = math32.Pow(blue, gammaExp)

		// Scale to 0-255 and clamp
		out[i] = clampToUint8(red)
		out[i+1] = clampToUint8(green)
		out[i+2] = clampToUint8(blue)
		out[i+3] = 255 // Alpha channel
	}

	return out
}

func DrawRaysBlockAdvanceTexture(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImageAdvance, gama float32, textureMap *[128]Texture, performance bool) {
	var wg sync.WaitGroup

	// Process each block
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			for y := block.startY; y < block.endY; y++ {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x++ {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c, normal := TraceRayV3AdvanceTexture(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples, textureMap, BVH)

					// Write the pixel color to the float buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.colorRGB_Float32[index] = c.R
					block.colorRGB_Float32[index+1] = c.G
					block.colorRGB_Float32[index+2] = c.B
					block.colorRGB_Float32[index+3] = c.A

					// block.distanceBuffer[index] = distance
					// block.distanceBuffer[index+1] = distance
					// block.distanceBuffer[index+2] = distance
					// block.distanceBuffer[index+3] = 255

					// if distance != math.MaxFloat32 {
					// 	block.maxDistance = math32.Max(block.maxDistance, distance)
					// 	block.minDistance = math32.Min(block.minDistance, distance)
					// }

					// Normalize the normal vector
					normal = normal.Normalize()

					// Convert the normal to the range [0 - 255]
					block.normalsBuffer[index] = clampToUint8((normal.x + 1) * 127.5)
					block.normalsBuffer[index+1] = clampToUint8((normal.y + 1) * 127.5)
					block.normalsBuffer[index+2] = clampToUint8((normal.z + 1) * 127.5)
					block.normalsBuffer[index+3] = 255

					// Track the maximum values
					block.maxColor.R = math32.Max(block.maxColor.R, c.R)
					block.maxColor.G = math32.Max(block.maxColor.G, c.G)
					block.maxColor.B = math32.Max(block.maxColor.B, c.B)
				}
			}
		}(block)
	}

	if !performance {
		wg.Wait()
	}

	// Compute global maximum color values
	maxColor := ColorFloat32{0, 0, 0, 0}
	// maxDistance := float32(0)
	// minDistance := float32(math32.MaxFloat32)
	for _, block := range blocks {
		maxColor.R = math32.Max(maxColor.R, block.maxColor.R)
		maxColor.G = math32.Max(maxColor.G, block.maxColor.G)
		maxColor.B = math32.Max(maxColor.B, block.maxColor.B)
		// maxDistance = math32.Max(maxDistance, block.maxDistance)
		// minDistance = math32.Min(minDistance, block.minDistance)
	}

	// normalize the distance buffer
	// for _, block := range blocks {
	// 	wg.Add(1)
	// 	go func(block BlocksImageAdvance) {
	// 		defer wg.Done()

	// 		// Handle edge case when all distances are equal
	// 		normalizer := maxDistance - minDistance
	// 		if normalizer == 0 {
	// 			normalizer = 1
	// 		}

	// 		for i := 0; i < len(block.distanceBuffer); i += 4 {
	// 			normalizedValue := (block.distanceBuffer[i] - minDistance) / normalizer
	// 			value := clampToUint8(normalizedValue)

	// 			// Set RGBA values
	// 			baseIndex := (i / 4) * 4
	// 			block.distanceBufferProcessed[baseIndex] = value
	// 			block.distanceBufferProcessed[baseIndex+1] = value
	// 			block.distanceBufferProcessed[baseIndex+2] = value
	// 			block.distanceBufferProcessed[baseIndex+3] = 255
	// 		}
	// 	}(block)
	// }

	// wg.Wait()

	// Apply color grading and write pixels
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			wg.Done()
			if renderVersion.Selected == logMode {
				block.image.WritePixels(ColorGradeLogarithmic(block.colorRGB_Float32, maxColor.R, maxColor.G, maxColor.B, gama+1*gama+1*gama+1))
			} else {
				block.image.WritePixels(ColorGradeLinear(block.colorRGB_Float32, maxColor.R, maxColor.G, maxColor.B, gama+1*gama+1*gama+1))
			}
			// block.distanceImage.WritePixels(block.distanceBufferProcessed)
			block.normalImage.WritePixels(block.normalsBuffer)
		}(block)

	}
	if !performance {
		wg.Wait()
	}
}

func DrawRaysBlockV2(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImage, performance bool) {
	var wg sync.WaitGroup
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImage) {
			defer wg.Done()
			for y := block.startY; y < block.endY; y += 1 {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x += 1 {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c := TraceRayV2(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples)

					// Write the pixel color to the pixel buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.pixelBuffer[index] = clampUint8(c.R)
					block.pixelBuffer[index+1] = clampUint8(c.G)
					block.pixelBuffer[index+2] = clampUint8(c.B)
					block.pixelBuffer[index+3] = clampUint8(c.A)
				}
			}
			block.image.WritePixels(block.pixelBuffer)
		}(block)
	}

	if !performance {
		wg.Wait()
	}
}

func DrawRaysBlockV2M(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImage, performance bool) {
	var wg sync.WaitGroup
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImage) {
			defer wg.Done()
			for y := block.startY; y < block.endY; y += 1 {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x += 1 {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c := TraceRayV3(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples)

					// Write the pixel color to the pixel buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.pixelBuffer[index] = clampUint8(c.R)
					block.pixelBuffer[index+1] = clampUint8(c.G)
					block.pixelBuffer[index+2] = clampUint8(c.B)
					block.pixelBuffer[index+3] = clampUint8(c.A)
				}
			}
			block.image.WritePixels(block.pixelBuffer)
		}(block)
	}

	if !performance {
		wg.Wait()
	}
}

func DrawRaysBlockAdvanceV4Log(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImageAdvance, gama float32, performance bool, bvh *BVHLeanNode, textureMap *[128]Texture) {
	var wg sync.WaitGroup

	// Process each block
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			for y := block.startY; y < block.endY; y++ {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x++ {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c, normal := TraceRayV4AdvanceTexture(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples, textureMap, bvh)

					// Write the pixel color to the float buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.colorRGB_Float32[index] = c.R
					block.colorRGB_Float32[index+1] = c.G
					block.colorRGB_Float32[index+2] = c.B
					block.colorRGB_Float32[index+3] = c.A

					// Normalize the normal vector
					normal = normal.Normalize()

					// Convert the normal to the range [0 - 255]
					block.normalsBuffer[index] = uint8((normal.x + 1) * 64)
					block.normalsBuffer[index+1] = uint8((normal.y + 1) * 64)
					block.normalsBuffer[index+2] = uint8((normal.z + 1) * 64)
					block.normalsBuffer[index+3] = 255

					// Track the maximum values
					block.maxColor.R = math32.Max(block.maxColor.R, c.R)
					block.maxColor.G = math32.Max(block.maxColor.G, c.G)
					block.maxColor.B = math32.Max(block.maxColor.B, c.B)
				}
			}
		}(block)
	}

	maxColor := ColorFloat32{0, 0, 0, 0}
	for _, block := range blocks {
		maxColor.R = math32.Max(maxColor.R, block.maxColor.R)
		maxColor.G = math32.Max(maxColor.G, block.maxColor.G)
		maxColor.B = math32.Max(maxColor.B, block.maxColor.B)
	}

	// Apply color grading and write pixels
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			block.image.WritePixels(ColorGradeLogarithmic(block.colorRGB_Float32, maxColor.R, maxColor.G, maxColor.B, gama+1*gama+1*gama+1))
			block.normalImage.WritePixels(block.normalsBuffer)
		}(block)
	}
	if !performance {
		wg.Wait()
	}
}

func DrawRaysBlockAdvanceV4Lin(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImageAdvance, gama float32, performance bool, bvh *BVHLeanNode, textureMap *[128]Texture) {
	var wg sync.WaitGroup

	// Process each block
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			for y := block.startY; y < block.endY; y++ {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x++ {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c, normal := TraceRayV4AdvanceTexture(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples, textureMap, bvh)

					// Write the pixel color to the float buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.colorRGB_Float32[index] = c.R
					block.colorRGB_Float32[index+1] = c.G
					block.colorRGB_Float32[index+2] = c.B
					block.colorRGB_Float32[index+3] = c.A

					// Normalize the normal vector
					normal = normal.Normalize()

					// Convert the normal to the range [0 - 255]
					block.normalsBuffer[index] = uint8((normal.x + 1) * 64)
					block.normalsBuffer[index+1] = uint8((normal.y + 1) * 64)
					block.normalsBuffer[index+2] = uint8((normal.z + 1) * 64)
					block.normalsBuffer[index+3] = 255

					// Track the maximum values
					block.maxColor.R = math32.Max(block.maxColor.R, c.R)
					block.maxColor.G = math32.Max(block.maxColor.G, c.G)
					block.maxColor.B = math32.Max(block.maxColor.B, c.B)
				}
			}
		}(block)
	}

	maxColor := ColorFloat32{0, 0, 0, 0}
	for _, block := range blocks {
		maxColor.R = math32.Max(maxColor.R, block.maxColor.R)
		maxColor.G = math32.Max(maxColor.G, block.maxColor.G)
		maxColor.B = math32.Max(maxColor.B, block.maxColor.B)
	}

	// Apply color grading and write pixels
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			block.image.WritePixels(ColorGradeLinear(block.colorRGB_Float32, maxColor.R, maxColor.G, maxColor.B, gama+1*gama+1*gama+1))
			block.normalImage.WritePixels(block.normalsBuffer)
		}(block)
	}
	if !performance {
		wg.Wait()
	}
}

func DrawRaysBlockAdvanceV4LogOptim(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImageAdvance, gama float32, performance bool, bvh *BVHLeanNode, textureMap *[128]Texture) {
	var wg sync.WaitGroup

	// Process each block
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			for y := block.startY; y < block.endY; y++ {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x++ {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c := TraceRayV4AdvanceTextureLean(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples, textureMap, bvh)

					// Write the pixel color to the float buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.colorRGB_Float32[index] = c.R
					block.colorRGB_Float32[index+1] = c.G
					block.colorRGB_Float32[index+2] = c.B
					block.colorRGB_Float32[index+3] = c.A

					// Track the maximum values
					block.maxColor.R = math32.Max(block.maxColor.R, c.R)
					block.maxColor.G = math32.Max(block.maxColor.G, c.G)
					block.maxColor.B = math32.Max(block.maxColor.B, c.B)
				}
			}
		}(block)
	}

	maxColor := ColorFloat32{0, 0, 0, 0}
	for _, block := range blocks {
		maxColor.R = math32.Max(maxColor.R, block.maxColor.R)
		maxColor.G = math32.Max(maxColor.G, block.maxColor.G)
		maxColor.B = math32.Max(maxColor.B, block.maxColor.B)
	}

	// Apply color grading and write pixels
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			block.image.WritePixels(ColorGradeLogarithmic(block.colorRGB_Float32, maxColor.R, maxColor.G, maxColor.B, gama+1*gama+1*gama+1))
			block.normalImage.WritePixels(block.normalsBuffer)
		}(block)
	}
	if !performance {
		wg.Wait()
	}
}

func DrawRaysBlockAdvanceV4LinOptim(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImageAdvance, gama float32, performance bool, bvh *BVHLeanNode, textureMap *[128]Texture) {
	var wg sync.WaitGroup

	// Process each block
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			for y := block.startY; y < block.endY; y++ {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x++ {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c := TraceRayV4AdvanceTextureLean(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples, textureMap, bvh)

					// Write the pixel color to the float buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.colorRGB_Float32[index] = c.R
					block.colorRGB_Float32[index+1] = c.G
					block.colorRGB_Float32[index+2] = c.B
					block.colorRGB_Float32[index+3] = c.A

					// Track the maximum values
					block.maxColor.R = math32.Max(block.maxColor.R, c.R)
					block.maxColor.G = math32.Max(block.maxColor.G, c.G)
					block.maxColor.B = math32.Max(block.maxColor.B, c.B)
				}
			}
		}(block)
	}

	maxColor := ColorFloat32{0, 0, 0, 0}
	for _, block := range blocks {
		maxColor.R = math32.Max(maxColor.R, block.maxColor.R)
		maxColor.G = math32.Max(maxColor.G, block.maxColor.G)
		maxColor.B = math32.Max(maxColor.B, block.maxColor.B)
	}

	// Apply color grading and write pixels
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			block.image.WritePixels(ColorGradeLinear(block.colorRGB_Float32, maxColor.R, maxColor.G, maxColor.B, gama+1*gama+1*gama+1))
			block.normalImage.WritePixels(block.normalsBuffer)
		}(block)
	}
	if !performance {
		wg.Wait()
	}
}

func DrawRaysBlockAdvanceV4LinO2(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImageAdvance, gama float32, performance bool, bvh *BVHLeanNode, textureMap *[128]Texture) {
	var wg sync.WaitGroup

	// Process each block
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			for y := block.startY; y < block.endY; y++ {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x++ {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c := TraceRayV4AdvanceTextureLeanOptim(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples, textureMap, bvh)

					// Write the pixel color to the float buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.colorRGB_Float32[index] = c.R
					block.colorRGB_Float32[index+1] = c.G
					block.colorRGB_Float32[index+2] = c.B
					block.colorRGB_Float32[index+3] = c.A

					// Track the maximum values
					block.maxColor.R = math32.Max(block.maxColor.R, c.R)
					block.maxColor.G = math32.Max(block.maxColor.G, c.G)
					block.maxColor.B = math32.Max(block.maxColor.B, c.B)
				}
			}
		}(block)
	}

	maxColor := ColorFloat32{0, 0, 0, 0}
	for _, block := range blocks {
		maxColor.R = math32.Max(maxColor.R, block.maxColor.R)
		maxColor.G = math32.Max(maxColor.G, block.maxColor.G)
		maxColor.B = math32.Max(maxColor.B, block.maxColor.B)
	}

	// Apply color grading and write pixels
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			block.image.WritePixels(ColorGradeLinear(block.colorRGB_Float32, maxColor.R, maxColor.G, maxColor.B, gama+1*gama+1*gama+1))
			block.normalImage.WritePixels(block.normalsBuffer)
		}(block)
	}
	if !performance {
		wg.Wait()
	}
}

func DrawRaysBlockAdvanceV4LogO2(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImageAdvance, gama float32, performance bool, bvh *BVHLeanNode, textureMap *[128]Texture) {
	var wg sync.WaitGroup

	// Process each block
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			for y := block.startY; y < block.endY; y++ {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x++ {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c := TraceRayV4AdvanceTextureLeanOptim(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples, textureMap, bvh)

					// Write the pixel color to the float buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.colorRGB_Float32[index] = c.R
					block.colorRGB_Float32[index+1] = c.G
					block.colorRGB_Float32[index+2] = c.B
					block.colorRGB_Float32[index+3] = c.A

					// Track the maximum values
					block.maxColor.R = math32.Max(block.maxColor.R, c.R)
					block.maxColor.G = math32.Max(block.maxColor.G, c.G)
					block.maxColor.B = math32.Max(block.maxColor.B, c.B)
				}
			}
		}(block)
	}

	maxColor := ColorFloat32{0, 0, 0, 0}
	for _, block := range blocks {
		maxColor.R = math32.Max(maxColor.R, block.maxColor.R)
		maxColor.G = math32.Max(maxColor.G, block.maxColor.G)
		maxColor.B = math32.Max(maxColor.B, block.maxColor.B)
	}

	// Apply color grading and write pixels
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			block.image.WritePixels(ColorGradeLogarithmic(block.colorRGB_Float32, maxColor.R, maxColor.G, maxColor.B, gama+1*gama+1*gama+1))
			block.normalImage.WritePixels(block.normalsBuffer)
		}(block)
	}
	if !performance {
		wg.Wait()
	}
}

func DrawRaysBlockAdvance(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImageAdvance, gama float32, performance bool) {
	var wg sync.WaitGroup

	// Process each block
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			for y := block.startY; y < block.endY; y++ {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x++ {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c, _, normal := TraceRayV3Advance(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples)

					// Write the pixel color to the float buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.colorRGB_Float32[index] = c.R
					block.colorRGB_Float32[index+1] = c.G
					block.colorRGB_Float32[index+2] = c.B
					block.colorRGB_Float32[index+3] = c.A

					// block.distanceBuffer[index] = distance
					// block.distanceBuffer[index+1] = distance
					// block.distanceBuffer[index+2] = distance
					// block.distanceBuffer[index+3] = 255

					// if distance != math.MaxFloat32 {
					// 	block.maxDistance = math32.Max(block.maxDistance, distance)
					// 	block.minDistance = math32.Min(block.minDistance, distance)
					// }

					// Normalize the normal vector
					normal = normal.Normalize()

					// Convert the normal to the range [0 - 255]
					block.normalsBuffer[index] = uint8((normal.x + 1) * 127.5)
					block.normalsBuffer[index+1] = uint8((normal.y + 1) * 127.5)
					block.normalsBuffer[index+2] = uint8((normal.z + 1) * 127.5)
					block.normalsBuffer[index+3] = 255

					// Track the maximum values
					block.maxColor.R = math32.Max(block.maxColor.R, c.R)
					block.maxColor.G = math32.Max(block.maxColor.G, c.G)
					block.maxColor.B = math32.Max(block.maxColor.B, c.B)
				}
			}
		}(block)
	}

	if !performance {
		wg.Wait()
	}

	maxColor := ColorFloat32{0, 0, 0, 0}
	for _, block := range blocks {
		maxColor.R = math32.Max(maxColor.R, block.maxColor.R)
		maxColor.G = math32.Max(maxColor.G, block.maxColor.G)
		maxColor.B = math32.Max(maxColor.B, block.maxColor.B)
	}

	// Apply color grading and write pixels
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImageAdvance) {
			defer wg.Done()
			if renderVersion.Selected == logMode {
				block.image.WritePixels(ColorGradeLogarithmic(block.colorRGB_Float32, maxColor.R, maxColor.G, maxColor.B, gama+1*gama+1*gama+1))
			} else {
				block.image.WritePixels(ColorGradeLinear(block.colorRGB_Float32, maxColor.R, maxColor.G, maxColor.B, gama+1*gama+1*gama+1))
			}
			// block.distanceImage.WritePixels(block.distanceBufferProcessed)
			block.normalImage.WritePixels(block.normalsBuffer)
		}(block)
	}
	wg.Wait()
}

// Time taken for V2:  8.789734794642856e+06
// Time taken for V2-Unsafe:  9.21041036607143e+06
// func DrawRaysBlockUnsafe(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImage) {
// 	var wg sync.WaitGroup
// 	ScreenSpaceCoordinatesPtr := unsafe.Pointer(&ScreenSpaceCoordinates[0][0])
// 	for _, block := range blocks {
// 		wg.Add(1)
// 		go func(block BlocksImage) {
// 			pixelBufferPtr := unsafe.Pointer(&block.pixelBuffer[0])
// 			defer wg.Done()
// 			for y := block.startY; y < block.endY; y += 1 {
// 				if y*scaling >= screenHeight {
// 					continue
// 				}
// 				for x := block.startX; x < block.endX; x += 1 {
// 					if x*scaling >= screenWidth {
// 						continue
// 					}
// 					// rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
// 					rayDir := *(*Vector)(unsafe.Pointer(uintptr(ScreenSpaceCoordinatesPtr) + uintptr((x*scaling*screenHeight+y*scaling)*12)))
// 					c := TraceRayV3(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples)

// 					// Write the pixel color to the pixel buffer
// 					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
// 					*(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index))) = clampUint8(c.R)
// 					*(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index+1))) = clampUint8(c.G)
// 					*(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index+2))) = clampUint8(c.B)
// 					*(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index+3))) = clampUint8(c.A)

// 				}
// 			}
// 			block.image.WritePixels(block.pixelBuffer)
// 		}(block)
// 	}

// 	if performanceOptions.Selected == 0 {
// 		wg.Wait()
// 	}
// }

func DrawSpheres(camera Camera, scaling int, iterations int, subImages []*ebiten.Image, light Light) {
	var wg sync.WaitGroup

	// Create a pool of worker goroutines, each handling a portion of the image
	for i := 0; i < numCPU; i++ {
		wg.Add(1)
		go func(startY int, endIndex int, subImage *ebiten.Image) {
			defer wg.Done()
			yRow := 0
			width, height := subImage.Bounds().Dx(), subImage.Bounds().Dy()
			imageSize := width * height * 4
			pixelBuffer := make([]uint8, imageSize)
			for y := startY; y < endIndex; y += scaling {
				xColumn := 0
				for x := 0; x < screenWidth; x += scaling {
					rayDir := ScreenSpaceCoordinates[x][y]
					// c, _ := RayMarching(Ray{origin: camera.Position, direction: rayDir}, spheres, iterations, light)
					c, _ := RayMarchBvh(Ray{origin: camera.Position, direction: rayDir}, iterations, light)
					// Write the pixel color to the pixel buffer
					index := ((yRow*width + xColumn) * 4)
					pixelBuffer[index] = uint8(c.R)
					pixelBuffer[index+1] = uint8(c.G)
					pixelBuffer[index+2] = uint8(c.B)
					pixelBuffer[index+3] = uint8(c.A)
					xColumn++
					// // Set the pixel color in the sub-image
					// subImage.Set(x/scaling, yRow, c)
				}
				yRow++
			}
			subImage.WritePixels(pixelBuffer)
		}(i*rowSize, (i+1)*rowSize, subImages[i])
	}

	if performanceOptions.Selected == 0 {
		// Wait for all workers to finish
		wg.Wait()
	}
}

func DrawRaysHDR(camera Camera, light Light, scaling int, samples int, depth int) *ebiten.Image {
	var wg sync.WaitGroup

	// Calculate dimensions
	subImageHeight := screenHeight / numCPU
	subImageWidth := screenWidth / scaling
	rowSize := screenHeight / numCPU

	// Create properly sized sub-images
	subImages := make([][]float32, numCPU)
	for i := 0; i < numCPU; i++ {
		subImages[i] = make([]float32, screenWidth*subImageHeight*4)
	}

	// Launch worker goroutines
	for i := 0; i < numCPU; i++ {
		wg.Add(1)
		startY := i * rowSize
		endY := (i + 1) * rowSize
		subImage := subImages[i] // Capture loop variables properly

		go func(startY, endY int, subImage []float32) {
			defer wg.Done()
			yRow := 0
			for y := startY; y < endY; y += scaling {
				xColumn := 0
				for x := 0; x < screenWidth; x += scaling {
					rayDir := ScreenSpaceCoordinates[x][y]
					c := TraceRay(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples)

					index := (yRow*subImageWidth + xColumn) * 4
					subImage[index] = c.R
					subImage[index+1] = c.G
					subImage[index+2] = c.B
					subImage[index+3] = c.A
					xColumn++
				}
				yRow++
			}
		}(startY, endY, subImage)
	}

	wg.Wait()

	// Find min/max values
	maxR, maxG, maxB := float32(-math.MaxFloat32), float32(-math.MaxFloat32), float32(-math.MaxFloat32)
	minR, minG, minB := float32(math.MaxFloat32), float32(math.MaxFloat32), float32(math.MaxFloat32)

	for i := 0; i < numCPU; i++ {
		for j := 0; j < len(subImages[i]); j += 4 {
			maxR = math32.Max(maxR, subImages[i][j])
			maxG = math32.Max(maxG, subImages[i][j+1])
			maxB = math32.Max(maxB, subImages[i][j+2])
			minR = math32.Min(minR, subImages[i][j])
			minG = math32.Min(minG, subImages[i][j+1])
			minB = math32.Min(minB, subImages[i][j+2])
		}
	}

	// Normalize with checks for division by zero
	for i := 0; i < numCPU; i++ {
		for j := 0; j < len(subImages[i]); j += 4 {
			if maxR != minR {
				subImages[i][j] = (subImages[i][j] - minR) / (maxR - minR)
			}
			if maxG != minG {
				subImages[i][j+1] = (subImages[i][j+1] - minG) / (maxG - minG)
			}
			if maxB != minB {
				subImages[i][j+2] = (subImages[i][j+2] - minB) / (maxB - minB)
			}
		}
	}

	// Combine sub-images
	finalImage := make([]float32, screenWidth*screenHeight*4)
	for i := 0; i < numCPU; i++ {
		copy(finalImage[i*subImageHeight*screenWidth*4:], subImages[i])
	}

	// Convert to RGBA
	finalImageUint8 := make([]uint8, screenWidth*screenHeight*4)
	for i := 0; i < len(finalImage); i++ {
		finalImageUint8[i] = uint8(math32.Min(finalImage[i]*255, 255))
	}

	// Create final image
	newImage := ebiten.NewImage(screenWidth, screenHeight)
	newImage.WritePixels(finalImageUint8)
	return newImage
}

var (
	bgColor        = color.RGBA{50, 50, 50, 255}
	trackColor     = color.RGBA{200, 200, 200, 255}
	colorSliderInd = color.RGBA{255, 0, 0, 255}
	propSliderInd  = color.RGBA{0, 255, 255, 255}
	selectedColor  = color.RGBA{255, 0, 0, 255}

	bgUniform       = &image.Uniform{bgColor}
	trackUniform    = &image.Uniform{trackColor}
	colorSliderUnif = &image.Uniform{colorSliderInd}
	propSliderUnif  = &image.Uniform{propSliderInd}

	selectedOptionUniform = &image.Uniform{selectedColor}

	optionUniform = &image.Uniform{color.RGBA{100, 100, 100, 255}}
)

type Options struct {
	Header               string
	Options              []string
	Selected             int
	Width                int
	Height               int
	Padding              int
	PositionX, PositionY int
}

type SliderLayout struct {
	sliderWidth     int
	sliderHeight    int
	indicatorHeight int
	sliderValue     float32
	padding         int
	startX          int
	startY          int
}

func SelectOption(opts *Options, screen *ebiten.Image, mouseX, mouseY int, mousePressed bool) {
	mouseX -= opts.Width
	mouseX -= opts.PositionX

	// Draw background
	bgRect := image.Rect(opts.PositionX, opts.PositionY, opts.PositionX+opts.Width, opts.PositionY+opts.Height)
	draw.Draw(screen, bgRect, bgUniform, image.Point{}, draw.Src)

	// Calculate button size
	numButtons := len(opts.Options)
	buttonWidth := (opts.Width / numButtons) - opts.Padding
	buttonHeight := opts.Height - 2*opts.Padding

	// Draw buttons
	for i := 0; i < numButtons; i++ {
		buttonX := opts.PositionX + i*(buttonWidth+opts.Padding)
		buttonY := opts.PositionY + opts.Padding
		buttonRect := image.Rect(buttonX, buttonY, buttonX+buttonWidth, buttonY+buttonHeight)

		// Draw selected or unselected button
		if i == opts.Selected {
			draw.Draw(screen, buttonRect, selectedOptionUniform, image.Point{}, draw.Src)
		} else {
			draw.Draw(screen, buttonRect, optionUniform, image.Point{}, draw.Src)
		}

		// Display option text
		ebitenutil.DebugPrintAt(screen, opts.Options[i], buttonX+5, buttonY+5)
	}

	// Handle mouse interaction
	if mousePressed {
		for i := 0; i < numButtons; i++ {
			buttonX := opts.PositionX + i*(buttonWidth+opts.Padding)
			buttonY := opts.PositionY + opts.Padding
			if mouseX > buttonX && mouseX < buttonX+buttonWidth && mouseY > buttonY && mouseY < buttonY+buttonHeight {
				opts.Selected = i
				break
			}
		}
	}

	// Draw header text
	ebitenutil.DebugPrintAt(screen, opts.Header, opts.PositionX, opts.PositionY+10)
}

// ColorSlider handles color, reflection and specular value adjustments
func ColorSlider(x, y int, screen *ebiten.Image, width, height int, r, g, b, a *float64,
	reflection, specular *float32, mouseX, mouseY int, mousePressed bool, directToScatter *float32, m *float32, roughness *float32, metallic *float32) {

	// Calculate layout once
	layout := SliderLayout{
		sliderWidth:     width - 20,
		sliderHeight:    12,
		indicatorHeight: 10,
		padding:         5,
		startX:          x + 10,
		startY:          y + height/3 + 10,
	}

	// Draw background (single allocation)
	draw.Draw(screen, image.Rect(x, y, x+width, y+height), bgUniform, image.Point{}, draw.Src)

	// Draw preview area
	previewColor := &image.Uniform{color.RGBA{
		uint8(*r * 255),
		uint8(*g * 255),
		uint8(*b * 255),
		uint8(*a * 255),
	}}
	draw.Draw(screen, image.Rect(x, y, x+width, y+height/3), previewColor, image.Point{}, draw.Src)

	// Process sliders
	processSlider(screen, layout, "R", r, false, 0, mouseX, mouseY, mousePressed)
	processSlider(screen, layout, "G", g, false, 1, mouseX, mouseY, mousePressed)
	processSlider(screen, layout, "B", b, false, 2, mouseX, mouseY, mousePressed)
	processSlider(screen, layout, "A", a, false, 3, mouseX, mouseY, mousePressed)
	processSlider(screen, layout, "Light Intensity", m, true, 4, mouseX, mouseY, mousePressed)
	processSlider(screen, layout, "Reflection", reflection, true, 5, mouseX, mouseY, mousePressed)
	processSlider(screen, layout, "Specular", specular, true, 6, mouseX, mouseY, mousePressed)
	processSlider(screen, layout, "Direct To Scatter", directToScatter, true, 7, mouseX, mouseY, mousePressed)
	processSlider(screen, layout, "Roughness", roughness, true, 8, mouseX, mouseY, mousePressed)
	processSlider(screen, layout, "Metallic", metallic, true, 9, mouseX, mouseY, mousePressed)
}

func processSlider(screen *ebiten.Image, layout SliderLayout, label string, value interface{},
	isFloat32 bool, index int, mouseX, mouseY int, mousePressed bool) {

	// Calculate positions
	yOffset := layout.startY + (layout.sliderHeight+layout.padding)*index
	trackRect := image.Rect(
		layout.startX,
		yOffset,
		layout.startX+layout.sliderWidth,
		yOffset+layout.sliderHeight,
	)

	// Draw track
	draw.Draw(screen, trackRect, trackUniform, image.Point{}, draw.Src)

	// Get current value
	var currentValue float64
	if isFloat32 {
		currentValue = float64(*value.(*float32))
	} else {
		currentValue = *value.(*float64)
	}

	// Calculate and draw indicator
	valueX := int(currentValue*float64(layout.sliderWidth)) + layout.startX
	indicatorRect := image.Rect(
		valueX-5,
		yOffset+(layout.sliderHeight-layout.indicatorHeight)/2,
		valueX+5,
		yOffset+(layout.sliderHeight-layout.indicatorHeight)/2+layout.indicatorHeight,
	)

	// Draw indicator with appropriate color
	if isFloat32 {
		draw.Draw(screen, indicatorRect, propSliderUnif, image.Point{}, draw.Src)
	} else {
		draw.Draw(screen, indicatorRect, colorSliderUnif, image.Point{}, draw.Src)
	}

	// Draw label
	ebitenutil.DebugPrintAt(screen, label, layout.startX, yOffset+5)

	// Handle mouse interaction
	if mousePressed && trackRect.Overlaps(image.Rect(mouseX, mouseY, mouseX+1, mouseY+1)) {
		newValue := clamp(float64(mouseX-layout.startX) / float64(layout.sliderWidth))
		if isFloat32 {
			*value.(*float32) = float32(newValue)
		} else {
			*value.(*float64) = newValue
		}
	}
}

// clamp ensures a value stays between 0 and 1
func clamp(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func findIntersectionAndSetColor(node *BVHNode, ray Ray, newColor ColorFloat32, reflection float32, specular float32, directToScatter float32, multiplayer float32, roughness float32, metallic float32) bool {
	if node == nil {
		return false
	}

	// Check if ray intersects the bounding box of the node
	if !BoundingBoxCollision(node.BoundingBox, ray) {
		return false
	}

	// If this is a leaf node, check the triangles for intersection
	if node.active {
		// for i, triangle := range *node.Triangles {
		if _, hit := ray.IntersectTriangleSimple(node.Triangles); hit {
			// fmt.Println("Triangle hit", triangle.color)
			m := float32(1)
			if multiplayer > 0.05 {
				m = (multiplayer + 1)
			}

			c := ColorFloat32{
				R: newColor.R * (m * m * m * m),
				G: newColor.G * (m * m * m * m),
				B: newColor.B * (m * m * m * m),
				A: newColor.A,
			}
			NewTriangle := TriangleSimple{
				v1:              node.Triangles.v1,
				v2:              node.Triangles.v2,
				v3:              node.Triangles.v3,
				color:           c,
				Normal:          node.Triangles.Normal,
				reflection:      reflection,
				specular:        specular,
				directToScatter: directToScatter,
				Roughness:       roughness,
				Metallic:        metallic,
			}
			node.Triangles = NewTriangle
			return true
		}
		// }
		return false
	}

	// Traverse the left and right child nodes
	leftHit := findIntersectionAndSetColor(node.Left, ray, newColor, reflection, specular, directToScatter, multiplayer, roughness, metallic)
	rightHit := findIntersectionAndSetColor(node.Right, ray, newColor, reflection, specular, directToScatter, multiplayer, roughness, metallic)

	return leftHit || rightHit
}

const sensitivityX = 0.005
const sensitivityY = 0.005

func calculateMin15PercentFPS() float64 {
	sort.Float64s(FPS)
	tenPercentCount := int(0.15 * float64(len(FPS)))

	if tenPercentCount == 0 {
		return FPS[0] // Handle case with fewer than 10 samples
	}

	min10PercentValues := FPS[:tenPercentCount]
	sum := 0.0
	for _, fps := range min10PercentValues {
		sum += fps
	}
	averageMin10PercentFPS := sum / float64(tenPercentCount)
	return averageMin10PercentFPS
}

func writeCSV(filename string, data [][]string) error {
	fmt.Printf("Writing data to %s...\n", filename)

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, record := range data {
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	fmt.Println("Benchmark data saved successfully.")
	return nil
}

func getSystemInfo() (string, int, float64, uint64, error) {
	// Get CPU information
	cpuInfo, err := cpu.Info()
	if err != nil {
		return "", 0, 0, 0, err
	}

	if len(cpuInfo) == 0 {
		return "", 0, 0, 0, fmt.Errorf("no CPU information available")
	}

	cpuName := cpuInfo[0].ModelName
	clockSpeed := cpuInfo[0].Mhz / 1000 // Convert MHz to GHz
	numCores := len(cpuInfo)            // Logical cores

	// Get RAM information
	memInfo, err := mem.VirtualMemory()
	if err != nil {
		return "", 0, 0, 0, err
	}
	totalRAM := memInfo.Total / (1024 * 1024 * 1024) // Convert bytes to GB

	return cpuName, numCores, clockSpeed, totalRAM, nil
}

func dumpBenchmarkData(rendererVersion int) error {
	const csvFileName = "benchmark_results.csv"

	fmt.Println("Starting benchmark data dump...")

	// Read the main.go code
	code, err := os.ReadFile("main.go")
	if err != nil {
		return err
	}
	codeString := string(code)

	// Calculate average FPS for this run
	currentAvgFPS := AverageFrameRate / float64(FrameCount)
	min10PercentFPS := calculateMin15PercentFPS() // Calculate min 15% FPS
	fmt.Printf("Current run - Average FPS: %.2f, Min FPS: %.2f, Max FPS: %.2f, Min 15%% FPS: %.2f\n", currentAvgFPS, MinFrameRate, MaxFrameRate, min10PercentFPS)

	// Check if CSV file exists and read existing data
	var records [][]string
	file, err := os.OpenFile(csvFileName, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer file.Close()

	// Get system information
	cpuName, numCores, clockSpeed, totalRAM, err := getSystemInfo()

	fmt.Println("Reading existing benchmark results...")
	reader := csv.NewReader(file)
	records, _ = reader.ReadAll()

	// Check if the code already exists in the CSV
	for i, record := range records {
		if len(record) > 0 && record[0] == codeString && cpuName == record[5] {
			fmt.Println("Code already exists in CSV. Updating averages...")

			// Parse existing FPS and framerate values
			existingFPS, err := strconv.ParseFloat(record[1], 64)
			if err != nil {
				return err
			}
			existingMinFPS, err := strconv.ParseFloat(record[2], 64)
			if err != nil {
				return err
			}
			existingMaxFPS, err := strconv.ParseFloat(record[3], 64)
			if err != nil {
				return err
			}
			existingMin10PercentFPS, err := strconv.ParseFloat(record[4], 64)
			if err != nil {
				return err
			}

			// Calculate new averages
			newAvgFPS := (existingFPS + currentAvgFPS) / 2
			newMinFPS := (existingMinFPS + MinFrameRate) / 2
			newMaxFPS := (existingMaxFPS + MaxFrameRate) / 2
			newMin10PercentFPS := (existingMin10PercentFPS + min10PercentFPS) / 2

			fmt.Printf("Old FPS: %.2f, New FPS: %.2f, Updated Average FPS: %.2f\n", existingFPS, currentAvgFPS, newAvgFPS)
			fmt.Printf("Old Min FPS: %.2f, New Min FPS: %.2f\n", existingMinFPS, newMinFPS)
			fmt.Printf("Old Max FPS: %.2f, New Max FPS: %.2f\n", existingMaxFPS, newMaxFPS)
			fmt.Printf("Old Min 15%% FPS: %.2f, New Min 15%% FPS: %.2f\n", existingMin10PercentFPS, newMin10PercentFPS)

			// Update the record with new averages
			records[i][1] = fmt.Sprintf("%.2f", newAvgFPS)
			records[i][2] = fmt.Sprintf("%.2f", newMinFPS)
			records[i][3] = fmt.Sprintf("%.2f", newMaxFPS)
			records[i][4] = fmt.Sprintf("%.2f", newMin10PercentFPS)
			// add system information
			records[i][5] = cpuName
			records[i][6] = fmt.Sprintf("%d", numCores)
			records[i][7] = fmt.Sprintf("%.2f", clockSpeed)
			records[i][8] = fmt.Sprintf("%d", totalRAM)

			// Write updated data back to CSV
			return writeCSV(csvFileName, records)
		}
	}

	// If code is not found, add a new row
	fmt.Println("Code not found in CSV. Adding new entry.")
	newRecord := []string{
		codeString,
		fmt.Sprintf("%.2f", currentAvgFPS),   // Average FPS
		fmt.Sprintf("%.2f", MinFrameRate),    // Min FPS
		fmt.Sprintf("%.2f", MaxFrameRate),    // Max FPS
		fmt.Sprintf("%.2f", min10PercentFPS), // Min 15% FPS
		cpuName,                              // CPU
		fmt.Sprintf("%d", numCores),          // Cores
		fmt.Sprintf("%.2f", clockSpeed),      // Clock speed
		fmt.Sprintf("%d", totalRAM),          // Total RAM
		fmt.Sprintf("%d", rendererVersion),   // Render Version
	}
	records = append(records, newRecord)

	if err != nil {
		panic(err)
	}

	fmt.Println("System Information")
	fmt.Printf("CPU: %s\n", cpuName)
	fmt.Printf("Cores: %d\n", numCores)
	fmt.Printf("Clock Speed: %.2f GHz\n", clockSpeed)
	fmt.Printf("Total RAM: %d GB\n", totalRAM)
	fmt.Println("RenderVesionCode:", rendererVersion)

	// Write data back to CSV
	return writeCSV(csvFileName, records)
}

func (g *Game) Update() error {

	if g.SendImage {
		saveEbitenImageAsPNG(g.currentFrame, "current.png")
		g.SendImage = false
	}

	// if Benchmark {
	// 	// rotate the camera around the y-axis
	// 	g.camera.yAxis += 0.005
	// 	PrecomputeScreenSpaceCoordinatesSphere(g.camera)

	// 	// Move the light source
	// 	g.light.Position.x += 0.005

	// 	// Move Camera
	// 	g.camera.Position.x += 0.01
	// 	g.camera.Position.y += 0.01

	// 	if time.Since(startTime) > time.Second*40 {
	// 		// Dump code and FPS to CSV
	// 		if err := dumpBenchmarkData(int(g.version)); err != nil {
	// 			fmt.Println("Error dumping benchmark data:", err)
	// 		}
	// 		os.Exit(0)
	// 	}
	// } else {

	if g.SnapLightToCamera {
		g.light.Position = g.camera.Position
	}

	mouseX, mouseY := ebiten.CursorPosition()
	if fullScreen {
		// Get the current mouse position
		dx := float32(mouseX-g.cursorX) * sensitivityX
		g.camera.xAxis += float32(dx)
		g.cursorX = mouseX

		dy := float32(mouseY-g.cursorY) * sensitivityY
		g.camera.yAxis += dy
		g.cursorY = mouseY

		forward := Vector{1, 0, 0}
		right := Vector{0, 1, 0}
		up := Vector{0, 0, 1}

		if ebiten.IsKeyPressed(ebiten.KeyShiftLeft) {
			g.xyzLock = !g.xyzLock
		}

		if g.xyzLock {
			forward = ScreenSpaceCoordinates[screenHeight/2][screenWidth/2]
			forward = Vector{forward.x, forward.y, 0}.Normalize()
			right = forward.Cross(Vector{0, 1, 0})
			up = right.Cross(forward)
		}
		speed := float32(5)

		if ebiten.IsKeyPressed(ebiten.KeyW) {
			g.camera.Position = g.camera.Position.Add(forward.Mul(speed)) // Move forward
		}
		if ebiten.IsKeyPressed(ebiten.KeyS) {
			g.camera.Position = g.camera.Position.Sub(forward.Mul(speed)) // Move backward
		}
		if ebiten.IsKeyPressed(ebiten.KeyD) {
			g.camera.Position = g.camera.Position.Add(right.Mul(speed)) // Move right
		}
		if ebiten.IsKeyPressed(ebiten.KeyA) {
			g.camera.Position = g.camera.Position.Sub(right.Mul(speed)) // Move left
		}
		if ebiten.IsKeyPressed(ebiten.KeyE) {
			g.camera.Position = g.camera.Position.Add(up.Mul(speed)) // Move up
		}
		if ebiten.IsKeyPressed(ebiten.KeyQ) {
			g.camera.Position = g.camera.Position.Sub(up.Mul(speed)) // Move down
		}

		PrecomputeScreenSpaceCoordinatesSphere(g.camera)
	} else {
		if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) && mouseX >= 0 && mouseY >= 0 && mouseX < screenWidth/2 && mouseY < screenHeight/2 {
			ray := Ray{origin: g.camera.Position, direction: ScreenSpaceCoordinates[mouseX*2][mouseY*2]}
			c := ColorFloat32{float32(g.r * 255), float32(g.g * 255), float32(g.b * 255), float32(g.a * 255)}
			findIntersectionAndSetColor(BVH, ray, c, g.reflection, g.specular, g.directToScatter, g.ColorMultiplier, g.roughness, g.metallic)
			g.bvhLean.FindIntersectionAndSetIt(int32(g.index), ray, g.TextureMap)
			if g.RenderVoxels {

				if g.UseRandomnessForPaint {
					m := float32(g.RandomnessVoxel)
					cRandom := ColorFloat32{(rand.Float32() - 1) * m, (rand.Float32() - 1) * m, (rand.Float32() - 1) * m, rand.Float32() * m}
					c = c.Add(cRandom)
				}

				switch g.VoxelMode {
				case DrawVoxel:
					g.VoxelGrid.SetVoxelColor(c, ray, 64)
				case RemoveVoxel:
					g.VoxelGrid.RemoveVoxel(ray, 64)
				case AddVoxel:
					g.VoxelGrid.AddVoxel(ray, 64, c)
				}
			}
		}
	}

	// check if mouse button is pressed

	if ebiten.IsKeyPressed(ebiten.KeyTab) {
		fullScreen = !fullScreen
	}

	return nil
}

func saveEbitenImageAsPNG(ebitenImg *ebiten.Image, filename string) error {
	// Get the size of the Ebiten image
	width, height := ebitenImg.Size()

	// Create an RGBA image to hold the pixel data
	rgba := image.NewRGBA(image.Rect(0, 0, width, height))

	// Iterate over the pixels in the Ebiten image and copy them to the RGBA image
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c := ebitenImg.At(x, y).(color.RGBA)
			c = color.RGBA{c.R, c.G, c.B, 255}
			rgba.Set(x, y, c)
		}
	}

	// Create the output file
	outFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer outFile.Close()

	// Encode the RGBA image as a PNG and save it
	err = png.Encode(outFile, rgba)
	if err != nil {
		return err
	}

	return nil
}

var (
	GUI              = ebiten.NewImage(400, 600)
	lastMousePressed bool
	guiNeedsUpdate   = true // Start with true to ensure initial render
	depthOption      = Options{
		Header:    "Select Depth",
		Options:   []string{"1", "2", "4", "8", "16", "32"},
		Selected:  0,
		Width:     400,
		Height:    50,
		Padding:   10,
		PositionX: 0,
		PositionY: 0,
	}
	scatterOption = Options{
		Header:    "Select Scatter",
		Options:   []string{"0", "1", "2", "4", "8", "16", "32", "64"},
		Selected:  0,
		Width:     400,
		Height:    50,
		Padding:   10,
		PositionX: 0,
		PositionY: 350,
	}
	snapLightToCamera = Options{
		Header:    "Snap Light to Camera",
		Options:   []string{"No", "Yes"},
		Selected:  0,
		Width:     400,
		Height:    50,
		Padding:   10,
		PositionX: 0,
		PositionY: 400,
	}
	screenResolution = Options{
		Header:    "Render Resolution",
		Options:   []string{"Native", "2X", "4X", "8X"},
		Selected:  1,
		Width:     400,
		Height:    50,
		Padding:   10,
		PositionX: 0,
		PositionY: 450,
	}

	rayMarching = Options{
		Header:    "Ray Marching",
		Options:   []string{"No", "Yes"},
		Selected:  0,
		Width:     400,
		Height:    50,
		Padding:   10,
		PositionX: 0,
		PositionY: 500,
	}

	performanceOptions = Options{
		Header:    "Performance Options",
		Options:   []string{"No", "Yes"},
		Selected:  0,
		Width:     400,
		Height:    50,
		Padding:   10,
		PositionX: 0,
		PositionY: 550,
	}

	renderFrame = Options{
		Header:    "Render Frame",
		Options:   []string{"No", "Yes", "Show"},
		Selected:  0,
		Width:     400,
		Height:    25,
		Padding:   5,
		PositionX: 0,
		PositionY: 300,
	}

	renderVersion = Options{
		Header:    "Render Version",
		Options:   []string{"V1", "V2", "V2-Log", "V2-Linear"},
		Selected:  1,
		Width:     400,
		Height:    25,
		Padding:   5,
		PositionX: 0,
		PositionY: 325,
	}

	gamaSlider = SliderLayout{
		sliderWidth:     400,
		sliderHeight:    12,
		indicatorHeight: 10,
		sliderValue:     2,
		padding:         5,
		startX:          0,
		startY:          100,
	}
)

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 800, 608
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Increment frame count and add current FPS to the average
	fps := ebiten.ActualFPS()
	// if Benchmark {
	// 	FrameCount++
	// 	AverageFrameRate += fps

	// 	MinFrameRate = math.Min(MinFrameRate, fps)
	// 	MaxFrameRate = math.Max(MaxFrameRate, fps)

	// 	FPS = append(FPS, fps)
	// }

	/// Clear the current frame
	g.currentFrame.Clear()

	// Perform path tracing and draw rays into the current frame

	depth := 2
	if !Benchmark {
		depth = depthOption.Selected
		depth = depth*2 + 1
	}

	scatter := 0
	if !Benchmark {
		scatter = scatterOption.Selected
		if scatter > 1 {
			scatter *= 2
		}
	}

	// Render a single frame
	if renderFrame.Selected == 1 {
		// Draw the frame
		Blocks := MakeNewBlocks(g.scaleFactor / 2)

		if renderVersion.Selected == 0 {
			DrawRaysBlock(g.camera, g.light, g.scaleFactor, scatter*8, depth, Blocks, g.PerformanceOptions)
		} else {
			DrawRaysBlockV2(g.camera, g.light, g.scaleFactor, scatter*8, depth, Blocks, g.PerformanceOptions)
		}
		for _, block := range g.BlocksImage {
			op := &ebiten.DrawImageOptions{}
			op.GeoM.Translate(float64(block.startX), float64(block.startY))
			g.currentFrame.DrawImage(block.image, op)
		}

		renderFrame.Selected = 0

		randomNumber := rand.Intn(100000)
		saveEbitenImageAsPNG(g.currentFrame, fmt.Sprintf("rendered_frame_%d.png", randomNumber))
	}

	// switch renderVersion.Selected {
	// case 0:
	// 	DrawRaysBlock(g.camera, g.light, g.scaleFactor, scatter, depth, g.BlocksImage)
	// case 1:
	// 	DrawRaysBlockV2(g.camera, g.light, g.scaleFactor, scatter, depth, g.BlocksImage)
	// case 2:
	// 	DrawRaysBlockAdvance(g.camera, g.light, g.scaleFactor, scatter, depth, g.BlocksImageAdvance, gamaSlider.sliderValue)
	// case 3:
	// 	DrawRaysBlockAdvance(g.camera, g.light, g.scaleFactor, scatter, depth, g.BlocksImageAdvance, gamaSlider.sliderValue)
	// }

	// elapsed2 := time.Since(start)

	// SpeedUp += float64(elapsed.Nanoseconds()) - float64(elapsed2.Nanoseconds())

	if !Benchmark && rayMarching.Selected == 1 {
		DrawSpheres(g.camera, g.scaleFactor, 2, g.subImagesRayMarching, g.light)
	}

	// Handle GUI separately
	if !fullScreen {
		mouseX, mouseY := ebiten.CursorPosition()
		mousePressed := ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft)

		// Check if GUI needs updating
		if mousePressed || lastMousePressed != mousePressed {
			guiNeedsUpdate = true
		}

		// Only update GUI if needed
		if guiNeedsUpdate {
			GUI.Clear()
			ColorSlider(0, 50, GUI, 400, 200, &g.r, &g.g, &g.b, &g.a, &g.reflection, &g.specular, mouseX-400, mouseY, mousePressed, &g.directToScatter, &g.ColorMultiplier, &g.roughness, &g.metallic)
			SelectOption(&depthOption, GUI, mouseX, mouseY, mousePressed)
			SelectOption(&scatterOption, GUI, mouseX, mouseY, mousePressed)
			SelectOption(&snapLightToCamera, GUI, mouseX, mouseY, mousePressed)
			// SelectOption(&screenResolution, GUI, mouseX, mouseY, mousePressed)
			SelectOption(&rayMarching, GUI, mouseX, mouseY, mousePressed)
			SelectOption(&performanceOptions, GUI, mouseX, mouseY, mousePressed)
			SelectOption(&renderFrame, GUI, mouseX, mouseY, mousePressed)
			SelectOption(&renderVersion, GUI, mouseX, mouseY, mousePressed)
			processSlider(GUI, gamaSlider, "Gama", &gamaSlider.sliderValue, true, 0, mouseX-400, mouseY, mousePressed)

			guiNeedsUpdate = false
			// if screenResolution.Selected == 0 {
			// 	g.scaleFactor = 1
			// }
			// g.scaleFactor = screenResolution.Selected * 2

			// g.BlocksImage = MakeNewBlocks(g.scaleFactor)
		}
		lastMousePressed = mousePressed

		// Draw GUI on top of the main render
		guiOp := &ebiten.DrawImageOptions{}
		guiOp.GeoM.Translate(400, 0)
		screen.DrawImage(GUI, guiOp)

		lastMousePressed = mousePressed
	}

	// Scale the main render
	mainOp := &ebiten.DrawImageOptions{}
	mainOp.Filter = ebiten.FilterLinear

	if !fullScreen {
		mainOp.GeoM.Scale(
			float64(g.scaleFactor/2),
			float64(g.scaleFactor/2),
		)
	} else {
		if g.scaleFactor == 1 {
			g.scaleFactor = 2
		}
		mainOp.GeoM.Scale(
			float64(g.scaleFactor),
			float64(g.scaleFactor),
		)
	}

	if performanceOptions.Selected == 1 {
		wg := sync.WaitGroup{}
		wg.Wait()
	}

	// for i, subImage := range g.subImages {
	// 	op := &ebiten.DrawImageOptions{}
	// 	// if !fullScreen {
	// 	op.GeoM.Translate(0, float64(subImageHeight/screenResolution.Selected)*float64(i))
	// 	// } else {
	// 	// 	op.GeoM.Translate(0, float64(subImageHeight)*float64(i))
	// 	// }
	// 	g.currentFrame.DrawImage(subImage, op)
	// }

	g.previousFrame = g.currentFrame

	depth = int(g.depth)

	switch g.version {
	case V1:
		DrawRaysBlock(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImage, g.PerformanceOptions)
	case V2:
		DrawRaysBlockV2(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImage, g.PerformanceOptions)
	case V2M:
		DrawRaysBlockV2M(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImage, g.PerformanceOptions)
	case V2Log:
		DrawRaysBlockAdvance(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImageAdvance, g.gamma, g.PerformanceOptions)
	case V2Linear:
		DrawRaysBlockAdvance(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImageAdvance, g.gamma, g.PerformanceOptions)
	case V2LinearTexture:
		DrawRaysBlockAdvanceTexture(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImageAdvance, g.gamma, g.TextureMap, g.PerformanceOptions)
	case V2LinearTexture2:
		DrawRaysBlockAdvanceTexture(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImageAdvance, g.gamma, g.TextureMap, g.PerformanceOptions)
	case V4Log:
		DrawRaysBlockAdvanceV4Log(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImageAdvance, g.gamma, g.PerformanceOptions, g.bvhLean, g.TextureMap)
	case V4Lin:
		DrawRaysBlockAdvanceV4Lin(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImageAdvance, g.gamma, g.PerformanceOptions, g.bvhLean, g.TextureMap)
	case V4LinOptim:
		DrawRaysBlockAdvanceV4LinOptim(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImageAdvance, g.gamma, g.PerformanceOptions, g.bvhLean, g.TextureMap)
	case V4LogOptim:
		DrawRaysBlockAdvanceV4LogOptim(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImageAdvance, g.gamma, g.PerformanceOptions, g.bvhLean, g.TextureMap)
	case V4LinO2:
		DrawRaysBlockAdvanceV4LinO2(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImageAdvance, g.gamma, g.PerformanceOptions, g.bvhLean, g.TextureMap)
	case V4LogO2:
		DrawRaysBlockAdvanceV4LogO2(g.camera, g.light, g.scaleFactor, g.scatter, depth, g.BlocksImageAdvance, g.gamma, g.PerformanceOptions, g.bvhLean, g.TextureMap)
	}

	if g.version == V4LogO2 || g.version == V4LinO2 || g.version == V4LogOptim || g.version == V4LinOptim || g.version == V4Log || g.version == V4Lin || g.version == V2Log || g.version == V2Linear || g.version == V2LinearTexture || g.version == V2LinearTexture2 {
		switch g.mode {
		case Classic:
			for _, block := range g.BlocksImageAdvance {
				op := &ebiten.DrawImageOptions{}
				op.GeoM.Translate(float64(block.startX), float64(block.startY))
				g.currentFrame.DrawImage(block.image, op)
			}
		// case Depth:
		// 	for _, block := range g.BlocksImageAdvance {
		// 		op := &ebiten.DrawImageOptions{}
		// 		op.GeoM.Translate(float64(block.startX), float64(block.startY))
		// 		g.currentFrame.DrawImage(block.distanceImage, op)
		// 	}
		case Normals:
			for _, block := range g.BlocksImageAdvance {
				op := &ebiten.DrawImageOptions{}
				op.GeoM.Translate(float64(block.startX), float64(block.startY))
				g.currentFrame.DrawImage(block.normalImage, op)
			}
		}
	} else {
		for _, block := range g.BlocksImage {
			op := &ebiten.DrawImageOptions{}
			op.GeoM.Translate(float64(block.startX), float64(block.startY))
			g.currentFrame.DrawImage(block.image, op)
		}
	}

	// if renderVersion.Selected == 0 {
	// 	for _, block := range g.BlocksImage {
	// 		op := &ebiten.DrawImageOptions{}
	// 		op.GeoM.Translate(float64(block.startX), float64(block.startY))
	// 		g.currentFrame.DrawImage(block.image, op)
	// 	}
	// } else if renderVersion.Selected == 1 {
	// 	for _, block := range g.BlocksImage {
	// 		op := &ebiten.DrawImageOptions{}
	// 		op.GeoM.Translate(float64(block.startX), float64(block.startY))
	// 		g.currentFrame.DrawImage(block.image, op)
	// 	}
	// } else {
	// 	for _, block := range g.BlocksImageAdvance {
	// 		op := &ebiten.DrawImageOptions{}
	// 		op.GeoM.Translate(float64(block.startX), float64(block.startY))
	// 		g.currentFrame.DrawImage(block.distanceImage, op)
	// 	}
	// }

	for i, subImage := range g.subImagesRayMarching {
		op := &ebiten.DrawImageOptions{}
		// if !fullScreen {
		op.GeoM.Translate(0, float64(subImageHeight/screenResolution.Selected)*float64(i))
		// } else {
		// 	op.GeoM.Translate(0, float64(subImageHeight)*float64(i))
		// }
		g.currentFrame.DrawImage(subImage, op)
	}

	// Draw Voxel Grid
	if g.RenderVolume {
		DrawRaysBlockVoxelGrid(g.camera, g.scaleFactor, 12, g.VoxelGridBlocksImage, g.VoxelGrid, g.light, g.VolumeMaterial)
		for _, block := range g.VoxelGridBlocksImage {
			op := &ebiten.DrawImageOptions{}
			op.GeoM.Translate(float64(block.startX), float64(block.startY))
			g.currentFrame.DrawImage(block.image, op)
		}
	}
	if g.RenderVoxels {
		DrawRaysBlockVoxels(g.camera, g.scaleFactor, 32, g.VoxelGridBlocksImage, g.VoxelGrid, g.light, g.VolumeMaterial)
		for _, block := range g.VoxelGridBlocksImage {
			op := &ebiten.DrawImageOptions{}
			op.GeoM.Translate(float64(block.startX), float64(block.startY))
			g.currentFrame.DrawImage(block.image, op)
		}
	}

	for _, shader := range g.Shaders {
		g.currentFrame = ApplyShader(g.currentFrame, shader)
	}
	screen.DrawImage(g.currentFrame, mainOp)

	g.previousFrame = ebiten.NewImageFromImage(screen)

	if renderFrame.Selected == 2 {
		// Draw the render
		screen.DrawImage(g.renderedFrame, mainOp)
	}

	ebitenutil.DebugPrint(screen, fmt.Sprintf("FPS: %.2f", fps))
}

var BVH *BVHNode
var FrameCount int

type Shader struct {
	shader    *ebiten.Shader
	options   map[string]interface{}
	amount    float32
	multipass int
}

func ApplyShader(image *ebiten.Image, shader Shader) *ebiten.Image {
	if image == nil {
		return nil
	}

	if shader.multipass > 1 {
		for i := 0; i < shader.multipass; i++ {
			newImage := ebiten.NewImageFromImage(image)
			opts := &ebiten.DrawRectShaderOptions{}
			opts.Images[0] = image
			// modify the shader options
			// r := rand.Float32()
			shader.options["Alpha"] = shader.amount
			opts.Uniforms = shader.options

			// Apply the shader
			newImage.DrawRectShader(
				newImage.Bounds().Dx(),
				newImage.Bounds().Dy(),
				shader.shader,
				opts,
			)
			image = newImage
		}
		return image
	}

	newImage := ebiten.NewImageFromImage(image)
	opts := &ebiten.DrawRectShaderOptions{}
	opts.Images[0] = image
	// modify the shader options
	// r := rand.Float32()
	shader.options["Alpha"] = shader.amount
	opts.Uniforms = shader.options

	// Apply the shader
	newImage.DrawRectShader(
		newImage.Bounds().Dx(),
		newImage.Bounds().Dy(),
		shader.shader,
		opts,
	)

	return newImage
}

// func ApplyMixShader(currentFrame *ebiten.Image, image *ebiten.Image, shader Shader, shaderMix *ebiten.Shader) *ebiten.Image {
// 	if image == nil {
// 		return nil
// 	}

// 	newImage := ebiten.NewImageFromImage(image)
// 	opts := &ebiten.DrawRectShaderOptions{}
// 	opts.Images[0] = image
// 	opts.Images[1] = currentFrame
// 	opts.Uniforms = map[string]interface{}{
// 		"Type":   shader.MixType,
// 		"Amount": shader.amount,
// 	}

// 	// Apply the shader
// 	newImage.DrawRectShader(
// 		newImage.Bounds().Dx(),
// 		newImage.Bounds().Dy(),
// 		shader.shader,
// 		opts,
// 	)

// 	return newImage
// }

const (
	V1               = uint8(iota)
	V2               = uint8(iota)
	V2Log            = uint8(iota)
	V2Linear         = uint8(iota)
	Native           = uint8(iota)
	TwoX             = uint8(iota)
	FourX            = uint8(iota)
	EightX           = uint8(iota)
	Classic          = uint8(iota)
	Normals          = uint8(iota)
	Depth            = uint8(iota)
	V2LinearTexture  = uint8(iota)
	V2LinearTexture2 = uint8(iota)
	V4Log            = uint8(iota)
	V4Lin            = uint8(iota)
	V4LogOptim       = uint8(iota)
	V4LinOptim       = uint8(iota)
	DrawVoxel        = uint8(iota)
	RemoveVoxel      = uint8(iota)
	AddVoxel         = uint8(iota)
	None             = uint8(iota)
	V2M              = uint8(iota)
	V4LogO2          = uint8(iota)
	V4LinO2          = uint8(iota)
)

type Game struct {
	// 64-bit pointers (8 bytes each) grouped together
	currentFrame  *ebiten.Image
	previousFrame *ebiten.Image
	renderedFrame *ebiten.Image
	VoxelGrid     *VoxelGrid
	bvhLean       *BVHLeanNode
	// bvhArray       *BVHArray
	VolumeMaterial VolumeMaterial

	// 64-bit floats (8 bytes each) grouped together
	r, g, b, a float64
	scatter    int
	TextureMap *[128]Texture

	// 32-bit floats (4 bytes each) grouped together
	specular        float32
	reflection      float32
	directToScatter float32
	ColorMultiplier float32
	roughness       float32
	metallic        float32
	gamma           float32
	// FOV             float32

	// 28-bit pointers (3.5 bytes each) grouped together
	light Light

	// 24-bit pointers (3 bytes each) grouped together
	subImagesRayMarching []*ebiten.Image
	VoxelGridBlocksImage []BlocksImage
	BlocksImage          []BlocksImage
	BlocksImageAdvance   []BlocksImageAdvance
	Shaders              []Shader

	// 20-bit pointers (2.5 bytes each) grouped together
	camera Camera

	// Integer values (4 bytes each) grouped together
	cursorX, cursorY int
	scaleFactor      int

	// Uint8 values (1 byte each) grouped together
	mode            uint8
	version         uint8
	depth           uint8
	index           uint8
	VoxelMode       uint8
	RandomnessVoxel uint8

	// Boolean flags (1 byte each) at the end
	RenderVolume          bool
	RenderVoxels          bool
	xyzLock               bool
	SnapLightToCamera     bool
	RayMarching           bool
	PerformanceOptions    bool
	UseRandomnessForPaint bool
	SendImage             bool
}

// LoadShader reads a shader file from the provided path and returns its content as a byte slice.
func LoadShader(filePath string) ([]byte, error) {
	// Read the entire file into memory
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	return data, nil
}

// Bayer matrix data
var bayerMatrix = [16]float32{
	15.0 / 255.0, 195.0 / 255.0, 60.0 / 255.0, 240.0 / 255.0,
	135.0 / 255.0, 75.0 / 255.0, 180.0 / 255.0, 120.0 / 255.0,
	45.0 / 255.0, 225.0 / 255.0, 30.0 / 255.0, 210.0 / 255.0,
	165.0 / 255.0, 105.0 / 255.0, 150.0 / 255.0, 90.0 / 255.0,
}

const subImageHeight = screenHeight / numCPU / 2
const subImageWidth = screenWidth

var fullScreen = false
var startTime time.Time

var Spheres = []SphereSimple{}

type BlocksImage struct {
	startX, startY, endX, endY int
	image                      *ebiten.Image
	pixelBuffer                []uint8
}

type BlocksImageAdvance struct {
	startX, startY, endX, endY int
	image                      *ebiten.Image
	// distanceImage              *ebiten.Image
	normalImage *ebiten.Image
	pixelBuffer []uint8
	// distanceBufferProcessed    []uint8
	normalsBuffer    []uint8
	colorRGB_Float32 []float32
	maxColor         ColorFloat32
	// maxDistance                float32
	// minDistance                float32
}

func MakeNewBlocks(scaling int) []BlocksImage {
	blocks := []BlocksImage{}
	blockSize := 32 / scaling

	for w := 0; w < screenWidth/scaling; w += blockSize {
		for h := 0; h < screenHeight/scaling; h += blockSize {
			blocks = append(blocks, BlocksImage{startX: w, startY: h, endX: w + blockSize, endY: h + blockSize, image: ebiten.NewImage(blockSize, blockSize), pixelBuffer: make([]uint8, blockSize*blockSize*4)})
		}
	}
	return blocks
}

func MakeNewBlocksAdvance(scaling int) []BlocksImageAdvance {
	blocks := []BlocksImageAdvance{}
	blockSize := 32 / scaling

	for w := 0; w < screenWidth/scaling; w += blockSize {
		for h := 0; h < screenHeight/scaling; h += blockSize {
			blocks = append(blocks,
				BlocksImageAdvance{startX: w,
					startY:           h,
					endX:             w + blockSize,
					endY:             h + blockSize,
					image:            ebiten.NewImage(blockSize, blockSize),
					pixelBuffer:      make([]uint8, blockSize*blockSize*4),
					colorRGB_Float32: make([]float32, blockSize*blockSize*4),
					// distanceBufferProcessed: make([]uint8, blockSize*blockSize*4),
					normalsBuffer: make([]uint8, blockSize*blockSize*4),
					// distanceImage:           ebiten.NewImage(blockSize, blockSize),
					normalImage: ebiten.NewImage(blockSize, blockSize),
					// minDistance:             math.MaxFloat32,
					// maxDistance:             0,
				})
		}
	}
	return blocks
}

// Handler
func hello(c echo.Context) error {
	return c.String(http.StatusOK, "Hello, World!")
}

func (g *Game) submitColor(c echo.Context) error {
	type Color struct {
		R               float64 `json:"r"`
		G               float64 `json:"g"`
		B               float64 `json:"b"`
		A               float64 `json:"a"`
		Reflection      float64 `json:"reflection"`
		Roughness       float64 `json:"roughness"`
		DirectToScatter float64 `json:"directToScatter"`
		Metallic        float64 `json:"metalic"`
		RenderVolume    bool    `json:"renderVolume"`
		RenderVoxels    bool    `json:"renderVoxels"`
	}

	color := new(Color)
	if err := c.Bind(color); err != nil {
		return err
	}

	// write old and new color to the console
	fmt.Println("Old Color:", g.r, g.g, g.b, g.a)
	fmt.Println("New Color:", color.R, color.G, color.B, color.A)

	// Unsafe assignment
	*(*float64)(unsafe.Pointer(&g.r)) = color.R
	*(*float64)(unsafe.Pointer(&g.g)) = color.G
	*(*float64)(unsafe.Pointer(&g.b)) = color.B
	*(*float64)(unsafe.Pointer(&g.a)) = color.A
	*(*float32)(unsafe.Pointer(&g.reflection)) = float32(color.Reflection)
	*(*float32)(unsafe.Pointer(&g.roughness)) = float32(color.Roughness)
	*(*float32)(unsafe.Pointer(&g.directToScatter)) = float32(color.DirectToScatter)
	*(*float32)(unsafe.Pointer(&g.metallic)) = float32(color.Metallic)

	return c.JSON(http.StatusOK, color)
}

func (g *Game) submitVoxelData(c echo.Context) error {
	type Volume struct {
		Density               float64 `json:"density"`
		Transmittance         float64 `json:"transmittance"`
		Randomnes             float64 `json:"randomness"`
		SmokeColorR           float64 `json:"smokeColorR"`
		SmokeColorG           float64 `json:"smokeColorG"`
		SmokeColorB           float64 `json:"smokeColorB"`
		SmokeColorA           float64 `json:"smokeColorA"`
		VoxelColorR           float64 `json:"voxelColorR"`
		VoxelColorG           float64 `json:"voxelColorG"`
		VoxelColorB           float64 `json:"voxelColorB"`
		VoxelColorA           float64 `json:"voxelColorA"`
		RandomnessVoxel       float64 `json:"randomnessVoxel"`
		RenderVolume          bool    `json:"renderVolume"`
		RenderVoxel           bool    `json:"renderVoxel"`
		OverWriteVoxel        bool    `json:"overWriteVoxel"`
		VoxelModification     string  `json:"voxelModification"`
		UseRandomnessForPaint bool    `json:"useRandomnessForPaint"`
		ConvertVoxelsToSmoke  bool    `json:"convertVoxelsToSmoke"`
	}

	volume := new(Volume)
	if err := c.Bind(volume); err != nil {
		return err
	}

	switch volume.VoxelModification {
	case "draw":
		*(*uint8)(unsafe.Pointer(&g.VoxelMode)) = DrawVoxel
	case "erase":
		*(*uint8)(unsafe.Pointer(&g.VoxelMode)) = RemoveVoxel
	case "add":
		*(*uint8)(unsafe.Pointer(&g.VoxelMode)) = AddVoxel
	case "none":
		*(*uint8)(unsafe.Pointer(&g.VoxelMode)) = None
	}

	fmt.Println("Volume", volume)

	*(*bool)(unsafe.Pointer(&g.UseRandomnessForPaint)) = volume.UseRandomnessForPaint
	*(*uint8)(unsafe.Pointer(&g.RandomnessVoxel)) = uint8(volume.RandomnessVoxel)

	fmt.Println("Use Randomness For Paint", g.UseRandomnessForPaint)
	fmt.Println("Randomness Voxel", g.RandomnessVoxel)

	if volume.ConvertVoxelsToSmoke {
		g.VoxelGrid.ConvertVoxelsToSmoke()
	} else {
		if volume.Density > 0 {
			g.VoxelGrid.SetBlockSmokeColorWithRandomnesUnsafe(
				ColorFloat32{float32(volume.SmokeColorR), float32(volume.SmokeColorG), float32(volume.SmokeColorB), float32(volume.SmokeColorA)},
				float32(volume.Randomnes))
		} else {
			g.VoxelGrid.SetBlockSmokeColorUnsafe(ColorFloat32{float32(volume.SmokeColorR), float32(volume.SmokeColorG), float32(volume.SmokeColorB), float32(volume.SmokeColorA)})
		}
	}

	if volume.OverWriteVoxel && volume.VoxelModification == "none" {
		fmt.Println("Overwrite Voxel", volume.OverWriteVoxel)
		if volume.RandomnessVoxel > 0 {
			g.VoxelGrid.SetBlockLightColorWithRandomnesUnsafe(
				ColorFloat32{float32(volume.VoxelColorR), float32(volume.VoxelColorG), float32(volume.VoxelColorB), float32(volume.VoxelColorA)},
				float32(volume.RandomnessVoxel))
		} else {
			g.VoxelGrid.SetBlockLightColorUnsafe(
				ColorFloat32{float32(volume.VoxelColorR), float32(volume.VoxelColorG), float32(volume.VoxelColorB), float32(volume.VoxelColorA)})
		}
	} else if volume.VoxelModification == "none" {
		fmt.Println("Overwrite Voxel", volume.OverWriteVoxel)
		c := ColorFloat32{float32(volume.VoxelColorR), float32(volume.VoxelColorG), float32(volume.VoxelColorB), float32(volume.VoxelColorA)}
		if volume.RandomnessVoxel > 0 {
			g.VoxelGrid.SetBlockLightColorWhereColorExistsWithRandomnesUnsafe(c, float32(volume.RandomnessVoxel))
		} else {
			g.VoxelGrid.SetBlockLightColorWhereColorExistsUnsafe(c)
		}
	}

	// write old and new color to the console
	fmt.Println("Old Volume:", g.VolumeMaterial.density, g.VolumeMaterial.transmittance)
	fmt.Println("New Volume:", volume.Density, volume.Transmittance)

	*(*float32)(unsafe.Pointer(&g.VolumeMaterial.density)) = float32(volume.Density)
	*(*float32)(unsafe.Pointer(&g.VolumeMaterial.transmittance)) = float32(volume.Transmittance)

	fmt.Println("Old Render Volume:", g.RenderVolume)
	fmt.Println("New Render Volume:", volume.RenderVolume)

	fmt.Println("Old Render Voxel:", g.RenderVoxels)
	fmt.Println("New Render Voxel:", volume.RenderVoxel)

	*(*bool)(unsafe.Pointer(&g.RenderVolume)) = volume.RenderVolume
	*(*bool)(unsafe.Pointer(&g.RenderVoxels)) = volume.RenderVoxel

	return c.JSON(http.StatusOK, volume)
}

type ShaderParam struct {
	Type       string                 `json:"type"`
	Parameters map[string]interface{} `json:"params"`
}

type ShaderMenu []ShaderParam

func (g *Game) SubmitShader(c echo.Context) error {
	var shaderMenu ShaderMenu
	if err := c.Bind(&shaderMenu); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Invalid shader menu format",
		})
	}

	for _, shader := range shaderMenu {
		fmt.Println("Shader Type:", shader.Type)
		fmt.Println("Shader Parameters:", shader.Parameters)
	}

	shaders := []Shader{}

	fmt.Println("Contrast Shader:", contrastShader)
	fmt.Println("Tint Shader:", tintShader)
	fmt.Println("Bloom Shader:", bloomShader)
	fmt.Println("Sharpness Shader:", sharpnessShader)
	fmt.Println("Color Mapping Shader:", colorMappingShader)
	fmt.Println("Chromatic Aberration Shader:", chromaticAberrationShader)

	// Process each shader
	for _, shader := range shaderMenu {
		switch shader.Type {
		case "contrast":
			shaders = append(shaders, Shader{
				shader:    contrastShader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		case "tint":
			// Convert TintColor array from interface{} to [3]float32
			tintColorInterface := shader.Parameters["TintColor"].([]interface{})
			tintColor := [3]float32{
				float32(tintColorInterface[0].(float64)),
				float32(tintColorInterface[1].(float64)),
				float32(tintColorInterface[2].(float64)),
			}

			// Update parameters with converted TintColor
			shader.Parameters["TintColor"] = tintColor

			shaders = append(shaders, Shader{
				shader:    tintShader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		case "bloom":
			shaders = append(shaders, Shader{
				shader:    bloomShader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		case "bloomV2":
			shaders = append(shaders, Shader{
				shader:    bloomV2Shader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		case "sharpen":
			shaders = append(shaders, Shader{
				shader:    sharpnessShader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		case "colorMapping":
			shaders = append(shaders, Shader{
				shader:    colorMappingShader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		case "chromaticAberration":
			fmt.Println("Chromatic Aberration Shader:", shader)
			shaders = append(shaders, Shader{
				shader:    chromaticAberrationShader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		case "edgeDetection":
			shaders = append(shaders, Shader{
				shader:    edgeDetectionShader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		case "colorMappingV2":
			shaders = append(shaders, Shader{
				shader:    colorMappingV2Shader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		case "Lighten":
			shaders = append(shaders, Shader{
				shader:    LightenDarkenShader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		case "CRT":
			shaders = append(shaders, Shader{
				shader:    CRTShader,
				options:   shader.Parameters,
				amount:    float32(shader.Parameters["amount"].(float64)),
				multipass: int(shader.Parameters["multipass"].(float64)),
			})
		}
	}

	fmt.Println("Shaders:", shaders)

	// assign the shaders to the game shaders using unsafe
	*(*[]Shader)(unsafe.Pointer(&g.Shaders)) = shaders

	return c.JSON(http.StatusOK, map[string]string{
		"message": "Shader menu updated successfully",
	})
}

func (g *Game) submitTextures(c echo.Context) error {
	type TextureRequest struct {
		Textures map[string]interface{} `json:"textures"`
		Normals  map[string]interface{} `json:"normals"`
		// Normal          map[string]interface{} `json:"normal"`
		DirectToScatter float64 `json:"directToScatter"`
		Reflection      float64 `json:"reflection"`
		Roughness       float64 `json:"roughness"`
		Metallic        float64 `json:"metallic"`
		Index           int     `json:"index"`
		Specular        float64 `json:"specular"`
		ColorR          float64 `json:"colorR"`
		ColorG          float64 `json:"colorG"`
		ColorB          float64 `json:"colorB"`
		ColorA          float64 `json:"colorA"`
	}

	request := new(TextureRequest)
	if err := c.Bind(request); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Failed to parse request: " + err.Error(),
		})
	}

	// Convert texture data
	expectedLength := 128 * 128 * 4
	textureData := make([]float32, expectedLength)
	normalData := make([]float32, expectedLength)

	// Process color texture
	if textureObj, ok := request.Textures["data"].(map[string]interface{}); ok {
		for key, value := range textureObj {
			if index, err := strconv.Atoi(key); err == nil && index < len(textureData) {
				switch v := value.(type) {
				case float64:
					textureData[index] = float32(v)
				case float32:
					textureData[index] = v
				case int:
					textureData[index] = float32(v)
				}
			}
		}
	}

	// Process normal texture
	if normalObj, ok := request.Normals["data"].(map[string]interface{}); ok {
		for key, value := range normalObj {
			if index, err := strconv.Atoi(key); err == nil && index < len(normalData) {
				switch v := value.(type) {
				case float64:
					normalData[index] = float32(v)
				case float32:
					normalData[index] = v
				case int:
					normalData[index] = float32(v)
				}
			}
		}
	}

	fmt.Println("Roughness", request.Roughness)
	fmt.Println("Metallic", request.Metallic)
	fmt.Println("DirectToScatter", request.DirectToScatter)
	fmt.Println("Reflection", request.Reflection)
	fmt.Println("Specular", request.Specular)

	fmt.Println("Color", request.ColorR, request.ColorG, request.ColorB, request.ColorA)

	// Update material properties using unsafe
	*(*float32)(unsafe.Pointer(&g.directToScatter)) = float32(request.DirectToScatter)
	*(*float32)(unsafe.Pointer(&g.reflection)) = float32(request.Reflection)
	*(*float32)(unsafe.Pointer(&g.roughness)) = float32(request.Roughness)
	*(*float32)(unsafe.Pointer(&g.metallic)) = float32(request.Metallic)
	*(*float32)(unsafe.Pointer(&g.specular)) = float32(request.Specular)
	*(*uint8)(unsafe.Pointer(&g.index)) = uint8(request.Index)

	// BVH.SetPropertiesWithID(uint8(1), float32(request.Reflection), float32(request.Specular), float32(request.DirectToScatter), float32(request.Roughness), float32(request.Metallic))

	// Convert and update color texture
	texture := Texture{}
	for i := 0; i < 128*128; i++ {
		x := i % 128
		y := i / 128
		texture.texture[x][y] = ColorFloat32{
			textureData[i*4] * float32(request.ColorR),
			textureData[i*4+1] * float32(request.ColorG),
			textureData[i*4+2] * float32(request.ColorB),
			textureData[i*4+3] * float32(request.ColorA),
		}
	}

	// Convert and update normal texture
	normalTexture := [128][128]Vector{}
	for i := 0; i < 128*128; i++ {
		x := i % 128
		y := i / 128
		normal := Vector{
			normalData[i*4],
			normalData[i*4+1],
			normalData[i*4+2],
		}
		normalTexture[x][y] = normal
	}

	// Unsafe update both textures
	*(*Texture)(unsafe.Pointer(&g.TextureMap[g.index].texture)) = texture
	*(*[128][128]Vector)(unsafe.Pointer(&g.TextureMap[g.index].normals)) = normalTexture
	*(*float32)(unsafe.Pointer(&g.TextureMap[g.index].reflection)) = float32(request.Reflection)
	*(*float32)(unsafe.Pointer(&g.TextureMap[g.index].specular)) = float32(request.Specular)
	*(*float32)(unsafe.Pointer(&g.TextureMap[g.index].directToScatter)) = float32(request.DirectToScatter)
	*(*float32)(unsafe.Pointer(&g.TextureMap[g.index].Roughness)) = float32(request.Roughness)
	*(*float32)(unsafe.Pointer(&g.TextureMap[g.index].Metallic)) = float32(request.Metallic)

	return c.JSON(http.StatusOK, map[string]interface{}{
		"status":      "success",
		"index":       request.Index,
		"textureSize": len(textureData),
		"normalSize":  len(normalData),
	})
}

func (g *Game) submitRenderOptions(c echo.Context) error {
	type RenderOptions struct {
		Depth          int     `json:"depth"`
		Scatter        int     `json:"scatter"`
		Gamma          float64 `json:"gamma"`
		SnapLight      string  `json:"snapLight"`
		RayMarching    string  `json:"rayMarching"`
		Performance    string  `json:"performance"`
		Mode           string  `json:"mode"`
		Resolution     string  `json:"resolution"`
		Version        string  `json:"version"`
		FOV            float64 `json:"fov"`
		LightIntensity float64 `json:"lightIntensity"`
		R              float64 `json:"r"`
		G              float64 `json:"g"`
		B              float64 `json:"b"`
	}

	renderOptions := new(RenderOptions)
	if err := c.Bind(renderOptions); err != nil {
		return err
	}

	fmt.Println("Render Options", renderOptions)

	// set unsafe values
	*(*uint8)(unsafe.Pointer(&g.depth)) = uint8(renderOptions.Depth)
	*(*uint8)(unsafe.Pointer(&g.scatter)) = uint8(renderOptions.Scatter)
	*(*float32)(unsafe.Pointer(&g.gamma)) = float32(renderOptions.Gamma)

	if renderOptions.SnapLight == "yes" {
		*(*bool)(unsafe.Pointer(&g.SnapLightToCamera)) = true
	} else {
		*(*bool)(unsafe.Pointer(&g.SnapLightToCamera)) = false
	}

	if renderOptions.RayMarching == "yes" {
		*(*bool)(unsafe.Pointer(&g.RayMarching)) = true
	} else {
		*(*bool)(unsafe.Pointer(&g.RayMarching)) = false
	}

	if renderOptions.Performance == "yes" {
		*(*bool)(unsafe.Pointer(&g.PerformanceOptions)) = true
	} else {
		*(*bool)(unsafe.Pointer(&g.PerformanceOptions)) = false
	}

	*(*float32)(unsafe.Pointer(&g.light.intensity)) = float32(renderOptions.LightIntensity)
	*(*float32)(unsafe.Pointer(&g.light.Color[0])) = float32(renderOptions.R)
	*(*float32)(unsafe.Pointer(&g.light.Color[1])) = float32(renderOptions.G)
	*(*float32)(unsafe.Pointer(&g.light.Color[2])) = float32(renderOptions.B)

	fmt.Println("Light Color", g.light.Color)
	fmt.Println("Light Intensity", g.light.intensity)

	*(*float32)(unsafe.Pointer(&FOV)) = float32(renderOptions.FOV)
	fmt.Println("FOV", FOV)

	fmt.Println("FOV Rad", FOVRadians)
	FOVRad := FOV * math.Pi / 180.0
	*(*float32)(unsafe.Pointer(&FOVRadians)) = float32(FOVRad)
	fmt.Println("FOV Rad", FOVRadians)

	switch renderOptions.Version {
	case "V1":
		fmt.Println("V1")
		*(*uint8)(unsafe.Pointer(&g.version)) = V1
	case "V2":
		fmt.Println("V2")
		*(*uint8)(unsafe.Pointer(&g.version)) = V2
	case "V2M":
		fmt.Println("V2M")
		*(*uint8)(unsafe.Pointer(&g.version)) = V2M
	case "V2-Log":
		fmt.Println("V2-Log")
		*(*uint8)(unsafe.Pointer(&g.version)) = V2Log
	case "V2-Linear":
		fmt.Println("V2-Linear")
		*(*uint8)(unsafe.Pointer(&g.version)) = V2Linear
	case "V2-Linear-Texture":
		fmt.Println("V2-Linear-Texture")
		*(*uint8)(unsafe.Pointer(&g.version)) = V2LinearTexture
	case "V2-Log-Texture":
		fmt.Println("V2-Log-Texture")
		*(*uint8)(unsafe.Pointer(&g.version)) = V2LinearTexture2
	case "V4-Log":
		fmt.Println("V4-Log")
		*(*uint8)(unsafe.Pointer(&g.version)) = V4Log
	case "V4-Linear":
		fmt.Println("V4-Lin")
		*(*uint8)(unsafe.Pointer(&g.version)) = V4Lin
	case "V4-Log-Optim":
		fmt.Println("V4-Log-Optim")
		*(*uint8)(unsafe.Pointer(&g.version)) = V4LogOptim
	case "V4-Linear-Optim":
		fmt.Println("V4-Lin-Optim")
		*(*uint8)(unsafe.Pointer(&g.version)) = V4LinOptim
	case "V4-Log-Optim-V2":
		fmt.Println("V4-Log-Optim-V2")
		*(*uint8)(unsafe.Pointer(&g.version)) = V4LogO2
	case "V4-Linear-Optim-V2":
		fmt.Println("V4-Lin-Optim-V2")
		*(*uint8)(unsafe.Pointer(&g.version)) = V4LinO2
	}

	switch renderOptions.Resolution {
	case "Native":
		fmt.Println("Native")
		// *(*uint8)(unsafe.Pointer(&g.resolution)) = Native
		*(*int)(unsafe.Pointer(&g.scaleFactor)) = 1
	case "2X":
		fmt.Println("2X")
		// *(*uint8)(unsafe.Pointer(&g.resolution)) = TwoX
		*(*int)(unsafe.Pointer(&g.scaleFactor)) = 2
	case "4X":
		fmt.Println("4X")
		// *(*uint8)(unsafe.Pointer(&g.resolution)) = FourX
		*(*int)(unsafe.Pointer(&g.scaleFactor)) = 4
	case "8X":
		fmt.Println("8X")
		// *(*uint8)(unsafe.Pointer(&g.resolution)) = EightX
		*(*int)(unsafe.Pointer(&g.scaleFactor)) = 8
	}

	*(*[]BlocksImage)(unsafe.Pointer(&g.VoxelGridBlocksImage)) = MakeNewBlocks(g.scaleFactor)
	*(*[]BlocksImageAdvance)(unsafe.Pointer(&g.BlocksImageAdvance)) = MakeNewBlocksAdvance(g.scaleFactor)

	switch renderOptions.Mode {
	case "Classic":
		fmt.Println("Classic")
		*(*uint8)(unsafe.Pointer(&g.mode)) = Classic
	case "Normals":
		fmt.Println("Normals")
		*(*uint8)(unsafe.Pointer(&g.mode)) = Normals
	case "Depth":
		fmt.Println("Depth")
		*(*uint8)(unsafe.Pointer(&g.mode)) = Depth
	}

	return c.JSON(http.StatusOK, renderOptions)
}

type Position struct {
	X       float64 `json:"x"`
	Y       float64 `json:"y"`
	Z       float64 `json:"z"`
	CameraX float64 `json:"cameraX"`
	CameraY float64 `json:"cameraY"`
}

func (g *Game) GetPositions(c echo.Context) error {
	pos := Position{
		X:       float64(g.camera.Position.x),
		Y:       float64(g.camera.Position.y),
		Z:       float64(g.camera.Position.z),
		CameraX: float64(g.camera.xAxis),
		CameraY: float64(g.camera.yAxis),
	}

	return c.JSON(http.StatusOK, pos)
}

func InterpolateBetweenPositions(timeSec time.Duration, Positions []Position) []Position {
	if len(Positions) < 2 {
		return Positions
	}

	// Calculate total frames needed based on time duration
	fps := 60
	totalFrames := int(timeSec.Seconds() * float64(fps))

	// Create array to hold all interpolated positions
	interpolatedPositions := make([]Position, totalFrames)

	// Calculate step size between keyframes
	framesPerSegment := totalFrames / (len(Positions) - 1)

	// Interpolate between each pair of positions
	for i := 0; i < len(Positions)-1; i++ {
		start := Positions[i]
		end := Positions[i+1]

		// Calculate deltas
		dx := (end.X - start.X) / float64(framesPerSegment)
		dy := (end.Y - start.Y) / float64(framesPerSegment)
		dz := (end.Z - start.Z) / float64(framesPerSegment)
		dCamX := (end.CameraX - start.CameraX) / float64(framesPerSegment)
		dCamY := (end.CameraY - start.CameraY) / float64(framesPerSegment)

		// Generate interpolated frames for this segment
		for frame := 0; frame < framesPerSegment; frame++ {
			// Calculate interpolation factor
			t := float64(frame) / float64(framesPerSegment)

			// Use smooth step interpolation for more natural movement
			smoothT := t * t * (3 - 2*t)

			index := i*framesPerSegment + frame
			interpolatedPositions[index] = Position{
				X:       start.X + dx*float64(frame)*smoothT,
				Y:       start.Y + dy*float64(frame)*smoothT,
				Z:       start.Z + dz*float64(frame)*smoothT,
				CameraX: start.CameraX + dCamX*float64(frame)*smoothT,
				CameraY: start.CameraY + dCamY*float64(frame)*smoothT,
			}
		}
	}

	// Fill in the final position
	interpolatedPositions[len(interpolatedPositions)-1] = Positions[len(Positions)-1]

	return interpolatedPositions
}

func (g *Game) MoveToCameraPosition(c echo.Context) error {
	pos := Position{}
	if err := c.Bind(&pos); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Failed to parse request: " + err.Error(),
		})
	}

	fmt.Println("Move to position", pos)
	newCamera := Camera{
		Position: Vector{float32(pos.X), float32(pos.Y), float32(pos.Z)},
		xAxis:    float32(pos.CameraX),
		yAxis:    float32(pos.CameraY),
	}

	*(*Camera)(unsafe.Pointer(&g.camera)) = newCamera
	return c.JSON(http.StatusOK, pos)
}

func corsMiddleware(next echo.HandlerFunc) echo.HandlerFunc {
	return func(c echo.Context) error {
		c.Response().Header().Set("Access-Control-Allow-Origin", "*")
		c.Response().Header().Set("Access-Control-Allow-Methods", "GET, PUT, POST, DELETE")
		c.Response().Header().Set("Access-Control-Allow-Headers", "Content-Type, Accept")

		if c.Request().Method == "OPTIONS" {
			return c.NoContent(http.StatusOK)
		}

		return next(c)
	}
}

func (g *Game) GetCurrentImage(c echo.Context) error {
	*(*bool)(unsafe.Pointer(&g.SendImage)) = true
	return c.JSON(http.StatusOK, map[string]string{
		"message": "Rendering image",
	})
}

func (g *Game) SendImageToClient(c echo.Context) error {
	// load png image current.png from disk

	fmt.Println("Send Image to Client")

	path := "current.png"
	file, err := os.Open(path)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to open image file",
		})
	}
	defer file.Close()

	// read image data
	data, err := ioutil.ReadAll(file)
	// send image data to client
	return c.Blob(http.StatusOK, "image/png", data)
}

func startServer(game *Game) {
	e := echo.New()
	// CORS middleware
	e.Use(corsMiddleware)

	e.POST("/submitColor", game.submitColor)
	e.POST("/submitVoxel", game.submitVoxelData)
	e.POST("/submitRenderOptions", game.submitRenderOptions)
	e.POST("/submitTextures", game.submitTextures)
	e.POST("/submitShader", game.SubmitShader)
	e.GET("/getCameraPosition", game.GetPositions)
	e.POST("/moveToPosition", game.MoveToCameraPosition)
	e.GET("/getCurrentImage", game.GetCurrentImage)
	e.GET("/sendImage", game.SendImageToClient)

	// Start server
	if err := e.Start(":5053"); err != nil && !errors.Is(err, http.ErrServerClosed) {
		e.Logger.Fatal("failed to start server:", err)
	}
}

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

var (
	bloomShader               *ebiten.Shader
	contrastShader            *ebiten.Shader
	tintShader                *ebiten.Shader
	sharpnessShader           *ebiten.Shader
	colorMappingShader        *ebiten.Shader
	bloomV2Shader             *ebiten.Shader
	chromaticAberrationShader *ebiten.Shader
	edgeDetectionShader       *ebiten.Shader
	colorMappingV2Shader      *ebiten.Shader
	LightenDarkenShader       *ebiten.Shader
	CRTShader                 *ebiten.Shader
)

// func math32.Sin(x float32) float32 {
//     // Reduce angle to [-π, π]
//     x = float32(math.Mod(float64(x), float64(2*math32.Pi)))
//     if x < -math32.Pi {
//         x += 2 * math32.Pi
//     } else if x > math32.Pi {
//         x -= 2 * math32.Pi
//     }

//     // Fast sine approximation using Bhaskara's approximation
//     if x < 0 {
//         x = -x
//         return -x*(2-x)/(1+x*(1-0.5*x))
//     }
//     return x*(2-x)/(1+x*(1-0.5*x))
// }

// Not accurate dont use
func fastSin(x float32) float32 {
	// Reduce angle to [-π, π]
	x = float32(math.Mod(float64(x), float64(2*math32.Pi)))
	if x < -math32.Pi {
		x += 2 * math32.Pi
	} else if x > math32.Pi {
		x -= 2 * math32.Pi
	}

	// Constants for polynomial approximation
	const (
		B = 4 / math32.Pi
		C = -4 / (math32.Pi * math32.Pi)
	)

	// Polynomial approximation
	y := B*x + C*x*math32.Abs(x)

	return y
}

// Not accurate dont use
func fastCos(x float32) float32 {
	// Reduce angle to [-π, π]
	x = float32(math.Mod(float64(x), float64(2*math32.Pi)))
	if x < -math32.Pi {
		x += 2 * math32.Pi
	} else if x > math32.Pi {
		x -= 2 * math32.Pi
	}

	// Shift x by π/2 to convert cosine to sine
	x += math32.Pi / 2
	if x > math32.Pi {
		x -= 2 * math32.Pi
	}

	// Fast cosine approximation using modified Bhaskara's approximation
	if x < 0 {
		x = -x
		return -x * (2 - x) / (1 + x*(1-0.5*x))
	}
	return x * (2 - x) / (1 + x*(1-0.5*x))
}

func main() {
	start := time.Now()
	for i := 0; i < 1000_000; i++ {
		x := rand.Float32()
		_ = math32.Sin(x)
	}
	fmt.Println("Sin:", time.Since(start))

	start = time.Now()
	for i := 0; i < 1000_000; i++ {
		x := rand.Float32()
		_ = fastSin(x)
	}
	fmt.Println("Fast Sin:", time.Since(start))

	start = time.Now()
	for i := 0; i < 1000_000; i++ {
		x := rand.Float32()
		_ = math32.Cos(x)
	}
	fmt.Println("Cos:", time.Since(start))

	start = time.Now()
	for i := 0; i < 1000_000; i++ {
		x := rand.Float32()
		_ = fastCos(x)
	}
	fmt.Println("Fast Cos:", time.Since(start))

	start = time.Now()
	for i := 0; i < 1000_000; i++ {
		_ = rand.Float32()
	}
	fmt.Println("Rand 32:", time.Since(start))

	start = time.Now()
	for i := 0; i < 1000_000; i++ {
		_ = rand.Float64()
	}
	fmt.Println("Rand 64:", time.Since(start))

	start = time.Now()
	for i := 0; i < 2*1000_000; i++ {
		bBoxMin := Vector{rand.Float32(), rand.Float32(), rand.Float32()}
		bBoxMax := Vector{rand.Float32(), rand.Float32(), rand.Float32()}
		ray := Ray{origin: Vector{rand.Float32(), rand.Float32(), rand.Float32()}, direction: Vector{rand.Float32(), rand.Float32(), rand.Float32()}}
		_, _ = BoundingBoxCollisionVector(bBoxMin, bBoxMax, ray)
	}
	fmt.Println("BoundingBoxCollisionVector:", time.Since(start))

	start = time.Now()
	for i := 0; i < 1000_000; i++ {
		bBoxMin := Vector{rand.Float32(), rand.Float32(), rand.Float32()}
		bBoxMax := Vector{rand.Float32(), rand.Float32(), rand.Float32()}
		bBoxMin1 := Vector{rand.Float32(), rand.Float32(), rand.Float32()}
		bBoxMax1 := Vector{rand.Float32(), rand.Float32(), rand.Float32()}
		ray := Ray{origin: Vector{rand.Float32(), rand.Float32(), rand.Float32()}, direction: Vector{rand.Float32(), rand.Float32(), rand.Float32()}}
		_, _, _, _ = BoundingBoxCollisionPair(bBoxMin, bBoxMax, bBoxMin1, bBoxMax1, ray)
	}
	fmt.Println("BoundingBoxCollisionPair:", time.Since(start))

	// if Benchmark {
	// 	debug.SetGCPercent(-1)
	// } else {
	// 	debug.SetGCPercent(200)
	// }
	// runtime.SetBlockProfileRate(0)

	// Compare the performance of the PrecomputeScreenSpaceCoordinatesSphereOptimalized and PrecomputeScreenSpaceCoordinatesSphere
	// start := time.Now()
	// PrecomputeScreenSpaceCoordinatesSphereOptimalized(Camera{})
	// fmt.Println("PrecomputeScreenSpaceCoordinatesSphereOptimalized:", time.Since(start))
	// start = time.Now()
	// PrecomputeScreenSpaceCoordinatesSphere(Camera{})
	// fmt.Println("PrecomputeScreenSpaceCoordinatesSphere:", time.Since(start))

	// set GOAMD64 to v3 to use AVX2
	// os.Setenv("GOAMD64", "v3")

	// TestGetBlock(100_000)
	// TestGetBlockUnsafe(100_000)

	src, err := LoadShader("shaders/ditherColor.kage")
	if err != nil {
		panic(err)
	}
	ditherShaderColor, err := ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	src, err = LoadShader("shaders/crt.kage")
	if err != nil {
		panic(err)
	}

	CRTShader, err = ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	fmt.Println("Shader:", CRTShader)

	// src, err = LoadShader("shaders/ColorMappingV2.kage")
	// if err != nil {
	// 	panic(err)
	// }
	// colorMappingV2Shader, err = ebiten.NewShader(src)
	// if err != nil {
	// 	panic(err)
	// }

	src, err = LoadShader("shaders/Lighten.kage")
	if err != nil {
		panic(err)
	}
	LightenDarkenShader, err = ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	src, err = LoadShader("shaders/chromatic.kage")
	if err != nil {
		panic(err)
	}
	chromaticAberrationShader, err = ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	src, err = LoadShader("shaders/edge.kage")
	if err != nil {
		panic(err)
	}
	edgeDetectionShader, err = ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	fmt.Println("Shader:", ditherShaderColor)

	// src, err = LoadShader("shaders/Mix.kage")
	// if err != nil {
	// 	panic(err)
	// }
	// mixShader, err := ebiten.NewShader(src)
	// if err != nil {
	// 	panic(err)
	// }

	// src, err = LoadShader("shaders/ditherGray.kage")
	// if err != nil {RotationMatrix
	// ditherGrayShader, err := ebiten.NewShader(src)
	// if err != nil {
	// 	panic(err)
	// }

	// src, err = LoadShader("shaders/RayCaster.kage")
	// if err != nil {
	// 	panic(err)
	// }
	// rayCasterShader, err := ebiten.NewShader(src)
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println("Shader:", rayCasterShader)

	src, err = LoadShader("shaders/bloom.kage")
	if err != nil {
		panic(err)
	}
	bloomShader, err = ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	src, err = LoadShader("shaders/bloomV2.kage")
	if err != nil {
		panic(err)
	}
	bloomV2Shader, err = ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	src, err = LoadShader("shaders/contrast.kage")
	if err != nil {
		panic(err)
	}
	contrastShader, err = ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	fmt.Println("Shader:", contrastShader)

	src, err = LoadShader("shaders/tint.kage")
	if err != nil {
		panic(err)
	}
	tintShader, err = ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	fmt.Println("Shader:", tintShader)

	src, err = LoadShader("shaders/sharpness.kage")
	if err != nil {
		panic(err)
	}
	sharpnessShader, err = ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	// src, err = LoadShader("shaders/rayMarching.kage")
	// if err != nil {
	// 	panic(err)
	// }

	// rayMarchingShader, err := ebiten.NewShader(src)
	// if err != nil {
	// 	panic(err)
	// }

	src, err = LoadShader("shaders/ColorMapping.kage")
	if err != nil {
		panic(err)
	}

	colorMappingShader, err = ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	// src, err = LoadShader("shaders/AverageFrames.kage")
	// if err != nil {
	// 	panic(err)
	// }

	// averageFramesShader, err := ebiten.NewShader(src)
	// if err != nil {
	// 	panic(err)
	// }

	// fmt.Println("Shader:", rayMarchingShader)

	fmt.Println("Shader:", sharpnessShader)
	// fmt.Println("Shader:", bloomShader)
	// fmt.Println("Shader:", ditherGrayShader)

	fmt.Println("Number of CPUs:", numCPU)

	ebiten.SetVsyncEnabled(false)
	ebiten.SetTPS(20)

	// spheres := GenerateRandomSpheres(15)
	// cubes := GenerateRandomCubes(30)

	obj := object{}
	if Benchmark {
		fullScreen = true
		obj, err = LoadOBJ("monkey.obj")
		if err != nil {
			panic(err)
		}
		obj.Scale(75)
	} else {
		obj, err = LoadOBJ("monkey.obj")
		if err != nil {
			panic(err)
		}
		obj.Scale(75)
	}

	objects := []object{}
	objects = append(objects, obj)

	camera := Camera{Position: Vector{0, 200, 100}, xAxis: 0, yAxis: 0}
	light := Light{Position: Vector{0, 1500, 1000}, Color: [3]float32{10.0, 10.0, 10.0}, intensity: 2.0}

	// bestDepth := OptimizeBVHDepth(objects, camera, light)

	// objects = append(objects, spheres...)
	// objects = append(objects, cubes...)

	BVH = ConvertObjectsToBVH(objects, maxDepth)
	bvhLean := BVH.ConvertToLeanBVH()
	fmt.Println("BVH Lean:", *bvhLean)
	fmt.Println("BVH Lean left:", *bvhLean.Left)
	fmt.Println("BVH Lean right:", *bvhLean.Right)

	texture := &[128]Texture{}

	start = time.Now()
	for i := 0; i < 1000_000; i++ {
		ray := Ray{origin: Vector{rand.Float32(), rand.Float32(), rand.Float32()}, direction: Vector{rand.Float32(), rand.Float32(), rand.Float32()}}
		_, _ = ray.IntersectBVHLean_TextureLeanOptim(bvhLean, texture)
	}
	fmt.Println("IntersectBVHLean_TextureLeanOptim:", time.Since(start))

	start = time.Now()
	for i := 0; i < 1000_000; i++ {
		ray := Ray{origin: Vector{rand.Float32(), rand.Float32(), rand.Float32()}, direction: Vector{rand.Float32(), rand.Float32(), rand.Float32()}}
		_, _ = ray.IntersectBVHLean_TextureLean(bvhLean, texture)
	}
	fmt.Println("IntersectBVHLean_TextureLean:", time.Since(start))

	start = time.Now()
	for i := 0; i < 1000_000; i++ {
		ray := Ray{origin: Vector{rand.Float32(), rand.Float32(), rand.Float32()}, direction: Vector{rand.Float32(), rand.Float32(), rand.Float32()}}
		_, _ = ray.IntersectBVH(BVH)
	}
	fmt.Println("IntersectBVHLean-V1:", time.Since(start))

	// remove obj and objects from memory
	// obj = object{}
	// objects = []object{}

	// BVHArray.textures[0].directToScatter = 0.5
	// BVHArray.textures[0].reflection = 0.5
	// BVHArray.textures[0].specular = 0.5
	// BVHArray.textures[0].Metallic = 0.5
	// BVHArray.textures[0].Roughness = 0.5

	// for i := 0; i < 128; i++ {
	// 	for j := 0; j < 128; j++ {
	// 		BVHArray.textures[0].texture[i][j] = ColorFloat32{rand.Float32() * 512, rand.Float32() * 512, rand.Float32() * 512, 255}
	// 	}
	// }

	// BVHArray := BVHArray{}
	// BVH.ConvertToArray(1, &BVHArray)

	// fmt.Println("BVHArray:", BVHArray.triangles[1])
	// fmt.Println("BVH:", BVH)

	// test speed of BVH

	start = time.Now()
	for i := 0; i < 1_000_000; i++ {
		// generate random ray
		ray := Ray{origin: Vector{rand.Float32() * 100, rand.Float32() * 100, rand.Float32() * 100}, direction: Vector{rand.Float32() * 100, rand.Float32() * 100, rand.Float32() * 100}}
		_, _ = ray.IntersectBVH(BVH)
	}
	fmt.Println("Calssic Bvh:", time.Since(start))

	start = time.Now()
	for i := 0; i < 1_000_000; i++ {
		// generate random ray
		ray := Ray{origin: Vector{rand.Float32() * 100, rand.Float32() * 100, rand.Float32() * 100}, direction: Vector{rand.Float32() * 100, rand.Float32() * 100, rand.Float32() * 100}}
		_, _ = ray.IntersectBVHLean_Texture(bvhLean, texture)
	}
	fmt.Println("Lean Bvh:", time.Since(start))

	// start = time.Now()
	// for i := 0; i < 1_000_000; i++ {
	// 	// generate random ray
	// 	ray := Ray{origin: Vector{rand.Float32() * 100, rand.Float32() * 100, rand.Float32() * 100}, direction: Vector{rand.Float32() * 100, rand.Float32() * 100, rand.Float32() * 100}}
	// 	_, _ = BVHArray.IntersectBVH(ray)
	// }
	// fmt.Println("Array Bvh:", time.Since(start))

	// start = time.Now()
	// textureMap := [128]Texture{}
	// for i := 0; i < 1_000_000; i++ {
	// 	// generate random ray
	// 	ray := Ray{origin: Vector{rand.Float32() * 100, rand.Float32() * 100, rand.Float32() * 100}, direction: Vector{rand.Float32() * 100, rand.Float32() * 100, rand.Float32() * 100}}
	// 	_, _ = ray.IntersectBVH_Texture(BVH, &textureMap)
	// }
	// fmt.Println("Texture Bvh:", time.Since(start))

	// start = time.Now()
	// for i := 0; i < 1_000_000; i++ {
	// 	// generate random ray
	// 	ray := Ray{origin: Vector{rand.Float32() * 100, rand.Float32() * 100, rand.Float32() * 100}, direction: Vector{rand.Float32() * 100, rand.Float32() * 100, rand.Float32() * 100}}
	// 	_, _ = ray.IntersectBVH_V2(BVH)
	// }
	// fmt.Println("V2 Bvh:", time.Since(start))

	PrecomputeScreenSpaceCoordinatesSphere(camera)

	VolumeMaterial := VolumeMaterial{transmittance: 50, density: 0.001}

	VoxelGrid := NewVoxelGrid(32, obj.BoundingBox[0].Mul(0.75), obj.BoundingBox[1].Mul(0.75), ColorFloat32{0, 0, 0, 2}, VolumeMaterial)

	// VoxelGrid.SetBlockSmokeColorWithRandomnes(ColorFloat32{125, 55, 25, 15}, 50)
	// VoxelGrid.SetRandomLightColor()
	for triangle := range obj.triangles {
		VoxelGrid.ConvertTriangleToVoxels(obj.triangles[triangle])
	}
	// fmt.Println("BVH:", BVH)
	// VoxelGrid.ConvertBVHtoVoxelGrid(BVH)

	// print some color values
	// for i := 0; i < 8; i++ {
	// 	fmt.Println(VoxelGrid.Blocks[i*8].LightColor)
	// }

	// generate random material
	// texture := new(Texture)
	// for i, row := range texture.texture {
	// 	for j := range row {
	// 		texture.texture[i][j] = ColorFloat32{rand.Float32() * 512, rand.Float32() * 512, rand.Float32() * 512, 255}
	// 	}
	// }

	// fmt.Println("Texture:", texture)

	// subImages := make([]*ebiten.Image, numCPU)
	subImages := [numCPU]*ebiten.Image{}

	for i := range numCPU {
		subImages[i] = ebiten.NewImage(int(subImageWidth), int(subImageHeight))
	}

	subImagesRayMarching := make([]*ebiten.Image, numCPU)

	for i := range numCPU {
		subImagesRayMarching[i] = ebiten.NewImage(int(subImageWidth), int(subImageHeight))
	}

	sphereBVH = *BuildBvhForSpheres(obj.ConvertToSquare(256), 6)

	// for _, t := range texture {
	// 	t.directToScatter = 0.5
	// 	t.reflection = 0.5
	// 	t.specular = 0.5
	// 	t.Metallic = 0.5
	// 	t.Roughness = 0.5
	// 	for i, row := range t.texture {
	// 		for j := range row {
	// 			t.texture[i][j] = ColorFloat32{rand.Float32() * 512, rand.Float32() * 512, rand.Float32() * 512, 255}
	// 		}
	// 	}
	// }

	const scale = 2

	game := &Game{
		version:              V2M,
		xyzLock:              true,
		cursorX:              screenHeight / 2,
		cursorY:              screenWidth / 2,
		subImagesRayMarching: subImagesRayMarching,
		camera:               camera,
		light:                light,
		scaleFactor:          scale,
		bvhLean:              bvhLean,
		// ditherColor:     ditherShaderColor,
		// ditherGrayScale: ditherGrayShader,
		// bloomShader:     bloomShader,
		// mixShader:          mixShader,
		currentFrame:         ebiten.NewImage(screenWidth/scale, screenHeight/scale),
		previousFrame:        ebiten.NewImage(screenWidth/scale, screenHeight/scale),
		BlocksImage:          MakeNewBlocks(scale),
		BlocksImageAdvance:   MakeNewBlocksAdvance(scale),
		VoxelGridBlocksImage: MakeNewBlocks(scale),
		VoxelGrid:            VoxelGrid,
		TextureMap:           texture,
		// bvhArray:             &BVHArray,
		// RayMarchShader: rayMarchingShader,
		// TriangleShader: 	   rayCasterShader,
		// averageFramesShader: averageFramesShader,
		Shaders: []Shader{
			Shader{
				shader: colorMappingShader,
				options: map[string]interface{}{
					"ColorR": 16.0,
					"ColorG": 16.0,
					"ColorB": 16.0,
					"Alpha":  0.8,
				},
				amount:    0.1,
				multipass: 1,
			},
			// Shader{shader: contrastShader, options: map[string]interface{}{"Contrast": 1.5, "Alpha": 0.1}, amount: 0.1},
			// Shader{shader: tintShader, options: map[string]interface{}{"TintColor": []float32{0.2, 0.6, 0.1}, "TintStrength": 0.1, "Alpha": 1}, amount: 0.5},
			// Shader{shader: ditherShaderColor, options: map[string]interface{}{"BayerMatrix": bayerMatrix, "Alpha": float32(0.5)}, amount: 1.0,},
			Shader{shader: bloomShader, options: map[string]interface{}{"BloomThreshold": 0.05, "BloomIntensity": 1.1, "Alpha": 1.0}, amount: 0.2, multipass: 2},
			// Shader{shader: sharpnessShader, options: map[string]interface{}{"Sharpness": 1.0, "Alpha": 1.0}, amount: 0.2},
		},
		VolumeMaterial: VolumeMaterial,
		// RenderVoxels:   true,
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Ebiten Benchmark")

	if Benchmark {
		renderVersions := []uint8{V1, V2, V2M, V2Log, V2Linear, V2LinearTexture, V2LinearTexture2, V4Log, V4Lin, V4LogOptim, V4LinOptim, V4LinO2, V4LogO2}

		cPositions := []Position{
			{X: -424.48, Y: 986.71, Z: 17.54, CameraX: 0.24, CameraY: -2.08},
			{X: 54.16, Y: 784.00, Z: 17.54, CameraX: 1.19, CameraY: -1.95},
			{X: 669.52, Y: 48.41, Z: 17.54, CameraX: -0.72, CameraY: -1.91}}
		CameraPositions := InterpolateBetweenPositions(10*time.Second, cPositions)
		camera = Camera{}

		const depth = 3
		const scatter = 8
		const scaleFactor = 2
		const gamma = 0.285

		BlocksImage := MakeNewBlocks(scaleFactor)
		BlocksImageAdvance := MakeNewBlocksAdvance(scaleFactor)

		TextureMap := [128]Texture{}
		for i := range TextureMap {
			for j := range TextureMap[i].texture {
				for k := range TextureMap[i].texture[j] {
					TextureMap[i].texture[j][k] = ColorFloat32{rand.Float32() * 256, rand.Float32() * 256, rand.Float32() * 256, 255}
				}
			}

			TextureMap[i] = Texture{
				directToScatter: 0.5,
				reflection:      0.5,
				specular:        0.5,
				Metallic:        0.5,
				Roughness:       0.5,
			}
		}

		// Preformance Options Off
		versionTimes := make(map[string][]float64)

		preformance := false

		for _, version := range renderVersions {
			var name string
			switch version {
			case V1:
				if preformance {
					name = "V1Preformance"
				} else {
					name = "V1"
				}
			case V2:
				if preformance {
					name = "V2Preformance"
				} else {
					name = "V2"
				}
			case V2Log:
				if preformance {
					name = "V2LogPreformance"
				} else {
					name = "V2Log"
				}
			case V2Linear:
				if preformance {
					name = "V2LinearPreformance"
				} else {
					name = "V2Linear"
				}
			case V2LinearTexture:
				if preformance {
					name = "V2LinearTexturePreformance"
				} else {
					name = "V2LinearTexture"
				}
			case V2LinearTexture2:
				if preformance {
					name = "V2LinearTexture2Preformance"
				} else {
					name = "V2LinearTexture"
				}
			case V4Log:
				if preformance {
					name = "V4LogPreformance"
				} else {
					name = "V4Log"
				}
			case V4Lin:
				if preformance {
					name = "V4LinPreformance"
				} else {
					name = "V4Lin"
				}
			case V4LinOptim:
				if preformance {
					name = "V4LinOptimPreformance"
				} else {
					name = "V4LinOptim"
				}
			case V4LogOptim:
				if preformance {
					name = "V4LogOptimPreformance"
				} else {
					name = "V4LogOptim"
				}
			case V2M:
				if preformance {
					name = "V2MPreformance"
				} else {
					name = "V2M"
				}
			case V4LinO2:
				if preformance {
					name = "V4LinO2Preformance"
				} else {
					name = "V4LinO2"
				}
			case V4LogO2:
				if preformance {
					name = "V4LogO2Preformance"
				} else {
					name = "V4LogO2"
				}
			}

			profileFilename := fmt.Sprintf("profiles/cpu_profile_v%s.prof", name)
			f, err := os.Create(profileFilename)
			if err != nil {
				log.Fatal(err)
			}

			// Start CPU profiling
			if err := pprof.StartCPUProfile(f); err != nil {
				log.Fatal(err)
			}

			TimeProfile := []float64{}
			for _, cPos := range CameraPositions {
				camera.Position = Vector{float32(cPos.X), float32(cPos.Y), float32(cPos.Z)}
				camera.xAxis = float32(cPos.CameraX)
				camera.yAxis = float32(cPos.CameraY)

				PrecomputeScreenSpaceCoordinatesSphere(camera)

				switch version {
				case V1:
					startTime = time.Now()
					DrawRaysBlock(camera, light, scaleFactor, scatter, depth, BlocksImage, preformance)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V2:
					startTime = time.Now()
					DrawRaysBlockV2(camera, light, scaleFactor, scatter, depth, BlocksImage, preformance)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V2M:
					startTime = time.Now()
					DrawRaysBlockV2M(camera, light, scaleFactor, scatter, depth, BlocksImage, preformance)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V2Log:
					startTime = time.Now()
					DrawRaysBlockAdvance(camera, light, scaleFactor, scatter, depth, BlocksImageAdvance, gamma, preformance)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V2Linear:
					startTime = time.Now()
					DrawRaysBlockAdvance(camera, light, scaleFactor, scatter, depth, BlocksImageAdvance, gamma, preformance)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V2LinearTexture:
					startTime = time.Now()
					DrawRaysBlockAdvanceTexture(camera, light, scaleFactor, scatter, depth, BlocksImageAdvance, gamma, &TextureMap, preformance)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V2LinearTexture2:
					startTime = time.Now()
					DrawRaysBlockAdvanceTexture(camera, light, scaleFactor, scatter, depth, BlocksImageAdvance, gamma, &TextureMap, preformance)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V4Log:
					startTime = time.Now()
					DrawRaysBlockAdvanceV4Log(camera, light, scaleFactor, scatter, depth, BlocksImageAdvance, gamma, preformance, bvhLean, &TextureMap)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V4Lin:
					startTime = time.Now()
					DrawRaysBlockAdvanceV4Lin(camera, light, scaleFactor, scatter, depth, BlocksImageAdvance, gamma, preformance, bvhLean, &TextureMap)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V4LinOptim:
					startTime = time.Now()
					DrawRaysBlockAdvanceV4LinOptim(camera, light, scaleFactor, scatter, depth, BlocksImageAdvance, gamma, preformance, bvhLean, &TextureMap)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V4LogOptim:
					startTime = time.Now()
					DrawRaysBlockAdvanceV4LogOptim(camera, light, scaleFactor, scatter, depth, BlocksImageAdvance, gamma, preformance, bvhLean, &TextureMap)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V4LinO2:
					startTime = time.Now()
					DrawRaysBlockAdvanceV4LinO2(camera, light, scaleFactor, scatter, depth, BlocksImageAdvance, gamma, preformance, bvhLean, &TextureMap)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				case V4LogO2:
					startTime = time.Now()
					DrawRaysBlockAdvanceV4LogO2(camera, light, scaleFactor, scatter, depth, BlocksImageAdvance, gamma, preformance, bvhLean, &TextureMap)
					TimeProfile = append(TimeProfile, float64(time.Since(startTime).Microseconds()))
				}

			}

			// Stop CPU profiling
			pprof.StopCPUProfile()
			f.Close()

			versionTimes[name] = TimeProfile
			averageTime := float64(0)
			for _, time := range TimeProfile {
				averageTime += time
			}
			averageTime = averageTime / float64(len(TimeProfile))
			fmt.Println("Version:", name, "AverageTime:", averageTime, "µs", "samples:", len(TimeProfile))

		}

		fmt.Println("Version Times:", versionTimes)

		cpuName, numCores, clockSpeed, totalRAM, err := getSystemInfo()
		if err != nil {
			panic(err)
		}

		type HWInfo struct {
			CPUName    string  `json:"CPUName"`
			NumCores   int     `json:"NumCores"`
			ClockSpeed float64 `json:"ClockSpeed"`
			TotalRAM   uint64  `json:"TotalRAM"`
		}

		hwInfo := HWInfo{
			CPUName:    cpuName,
			NumCores:   numCores,
			ClockSpeed: clockSpeed,
			TotalRAM:   totalRAM,
		}

		type Report struct {
			HWInfo       HWInfo             `json:"HWInfo"`
			VersionTimes map[string][]float64 `json:"VersionTimes"`
		}

		report := Report{
			HWInfo:       hwInfo,
			VersionTimes: versionTimes,
		}

		// dump times to file
		dump, err := json.MarshalIndent(report, "", " ")
		if err != nil {
			panic(err)
		}

		err = ioutil.WriteFile("profiles/versionTimes.json", dump, 0644)
		if err != nil {
			panic(err)
		}
	}

	go startServer(game)

	// set start time
	startTime = time.Now()
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}

}

// Voxel Smoke Simulation

type Block struct {
	Position   Vector
	LightColor ColorFloat32
	SmokeColor ColorFloat32
}

type VoxelGrid struct {
	BlocksPointer  unsafe.Pointer
	Blocks         []Block
	BBMin          Vector
	BBMax          Vector
	Resolution     int
	VolumeMaterial VolumeMaterial
}

// safe
// func (v *VoxelGrid) CalcualteSDF() {
// 	for i, b1 := range v.Blocks {
// 		for j, b2 := range v.Blocks {
// 			// Calculate the distance between the two blocks
// 			if i != j && b1.LightColor.A > 0 && b2.LightColor.A > 0 {
// 				dist := b1.Position.Sub(b2.Position).Length()
// 				if dist < b1.MinStepDist {
// 					b1.MinStepDist = dist
// 				}
// 			}
// 		}
// 	}
// }

// func (v *VoxelGrid) CalcualteSDF() {
// 	var wg sync.WaitGroup

// 	for i := range v.Blocks {
// 		wg.Add(1)
// 		go func(i int) {
// 			defer wg.Done()
// 			b1 := &v.Blocks[i]
// 			for j := range v.Blocks {
// 				if i != j {
// 					b2 := &v.Blocks[j]
// 					if b1.LightColor.A > 0 && b2.LightColor.A > 0 {
// 						dist := b1.Position.Sub(b2.Position).Length()
// 						minStepDistPtr := (*float32)(unsafe.Pointer(&b1.MinStepDist))
// 						if dist < *minStepDistPtr {
// 							*minStepDistPtr = dist
// 						}
// 					}
// 				}
// 			}
// 		}(i)
// 	}

// 	wg.Wait()
// }

func (v *VoxelGrid) ConvertBVHtoVoxelGrid(bvh *BVHNode) {
	for i := range v.Blocks {
		hit, t := bvh.PointInBoundingBox(v.Blocks[i].Position)
		if hit {
			v.Blocks[i].LightColor = t.color
		} else {
			v.Blocks[i].LightColor = ColorFloat32{0, 0, 0, 0}
		}
	}
}

func NewVoxelGrid(resolution int, minBB Vector, maxBB Vector, SmokeColor ColorFloat32, VolumeMaterial VolumeMaterial) *VoxelGrid {
	xDiff := maxBB.x - minBB.x
	yDiff := maxBB.y - minBB.y
	zDiff := maxBB.z - minBB.z

	xStep := xDiff / float32(resolution)
	yStep := yDiff / float32(resolution)
	zStep := zDiff / float32(resolution)

	v := VoxelGrid{Resolution: resolution, BBMin: minBB, BBMax: maxBB}

	for x := 0; x < resolution; x++ {
		for y := 0; y < resolution; y++ {
			for z := 0; z < resolution; z++ {
				v.Blocks = append(v.Blocks, Block{
					Position: Vector{
						minBB.x + (float32(x)+0.5)*xStep,
						minBB.y + (float32(y)+0.5)*yStep,
						minBB.z + (float32(z)+0.5)*zStep,
					},
					SmokeColor: SmokeColor,
				})
			}
		}
	}

	v.VolumeMaterial = VolumeMaterial

	v.BlocksPointer = unsafe.Pointer(&v.Blocks[0])

	// v.CalcualteSDF()

	return &v
}

func (v *VoxelGrid) SetBlockSmokeColor(color ColorFloat32) {
	for i := range v.Blocks {
		v.Blocks[i].SmokeColor = color
	}
}

func (v *VoxelGrid) ConvertVoxelsToSmoke() {
	for i := range v.Blocks {
		if v.Blocks[i].LightColor.A > 10 {
			ptrBlock := (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(i)*unsafe.Sizeof(Block{})))
			ptrBlock.SmokeColor = v.Blocks[i].LightColor
		}
	}
}

func (v *VoxelGrid) SetBlockSmokeColorUnsafe(color ColorFloat32) {
	for i := range v.Blocks {
		// Unsafe assignment
		smokeColorPtr := (*ColorFloat32)(unsafe.Pointer(&v.Blocks[i].SmokeColor))
		*smokeColorPtr = color
	}
}

func (v *VoxelGrid) SetBlockSmokeColorWithRandomnes(color ColorFloat32, randomness float32) {
	for i := range v.Blocks {
		rRandom := (rand.Float32() - 0.5) * randomness
		gRandom := (rand.Float32() - 0.5) * randomness
		bRandom := (rand.Float32() - 0.5) * randomness
		v.Blocks[i].SmokeColor = ColorFloat32{color.R + rRandom, color.G + gRandom, color.B + bRandom, color.A}
	}
}

func (v *VoxelGrid) SetBlockSmokeColorWithRandomnesUnsafe(color ColorFloat32, randomness float32) {
	for i := range v.Blocks {
		rRandom := (rand.Float32() - 0.5) * randomness
		gRandom := (rand.Float32() - 0.5) * randomness
		bRandom := (rand.Float32() - 0.5) * randomness

		// Unsafe assignment
		smokeColorPtr := (*ColorFloat32)(unsafe.Pointer(&v.Blocks[i].SmokeColor))
		*smokeColorPtr = ColorFloat32{color.R + rRandom, color.G + gRandom, color.B + bRandom, color.A}
	}
}

func (v *VoxelGrid) SetBlockLightColor(color ColorFloat32) {
	for i := range v.Blocks {
		v.Blocks[i].LightColor = color
	}
}

func (v *VoxelGrid) SetBlockLightColorUnsafe(color ColorFloat32) {
	for i := range v.Blocks {
		// Unsafe assignment
		lightColorPtr := (*ColorFloat32)(unsafe.Pointer(&v.Blocks[i].LightColor))
		*lightColorPtr = color
	}
}

func (v *VoxelGrid) SetBlockLightColorWithRandomnes(color ColorFloat32, randomness float32) {
	for i := range v.Blocks {
		rRandom := (rand.Float32() - 0.5) * randomness
		gRandom := (rand.Float32() - 0.5) * randomness
		bRandom := (rand.Float32() - 0.5) * randomness
		v.Blocks[i].LightColor = ColorFloat32{color.R + rRandom, color.G + gRandom, color.B + bRandom, color.A}
	}
}

func (v *VoxelGrid) SetBlockLightColorWithRandomnesUnsafe(color ColorFloat32, randomness float32) {
	for i := range v.Blocks {
		rRandom := (rand.Float32() - 0.5) * randomness
		gRandom := (rand.Float32() - 0.5) * randomness
		bRandom := (rand.Float32() - 0.5) * randomness

		// Unsafe assignment
		lightColorPtr := (*ColorFloat32)(unsafe.Pointer(&v.Blocks[i].LightColor))
		*lightColorPtr = ColorFloat32{color.R + rRandom, color.G + gRandom, color.B + bRandom, color.A}
	}
}

func (v *VoxelGrid) SetBlockLightColorWhereColorExistsWithRandomnesUnsafe(c ColorFloat32, r float32) {
	for i := range v.Blocks {
		if (v.Blocks[i].LightColor.R > 25 || v.Blocks[i].LightColor.G > 25 || v.Blocks[i].LightColor.B > 25) && v.Blocks[i].LightColor.A > 128 {
			rRandom := (rand.Float32() - 0.5) * r
			gRandom := (rand.Float32() - 0.5) * r
			bRandom := (rand.Float32() - 0.5) * r

			// Unsafe assignment
			lightColorPtr := (*ColorFloat32)(unsafe.Pointer(&v.Blocks[i].LightColor))
			*lightColorPtr = ColorFloat32{c.R + rRandom, c.G + gRandom, c.B + bRandom, c.A}
		}
	}
}

func (v *VoxelGrid) SetBlockLightColorWhereColorExistsUnsafe(c ColorFloat32) {
	for i := range v.Blocks {
		if (v.Blocks[i].LightColor.R > 25 || v.Blocks[i].LightColor.G > 25 || v.Blocks[i].LightColor.B > 25) && v.Blocks[i].LightColor.A > 128 {
			// Unsafe assignment
			lightColorPtr := (*ColorFloat32)(unsafe.Pointer(&v.Blocks[i].LightColor))
			*lightColorPtr = ColorFloat32{c.R, c.G, c.B, c.A}
		}
	}
}

func (v *VoxelGrid) SetRandomSmokeColor() {
	for i := range v.Blocks {
		v.Blocks[i].SmokeColor = ColorFloat32{rand.Float32() * 255, rand.Float32() * 255, rand.Float32() * 255, 255}
	}
}

func (v *VoxelGrid) SetRandomLightColor() {
	for i := range v.Blocks {
		if rand.Float32() < 0.1 {
			v.Blocks[i].LightColor = ColorFloat32{rand.Float32() * 255, rand.Float32() * 255, rand.Float32() * 255, 255}
		} else {
			v.Blocks[i].LightColor = ColorFloat32{0, 0, 0, 0}
		}
	}
}

func (v *VoxelGrid) ConvertTriangleToVoxels(triangle TriangleSimple) {
	V1 := triangle.v1
	V2 := triangle.v2
	V3 := triangle.v3

	if v.BBMin.x > v.BBMax.x {
		v.BBMin.x, v.BBMax.x = v.BBMax.x, v.BBMin.x
	}
	if v.BBMin.y > v.BBMax.y {
		v.BBMin.y, v.BBMax.y = v.BBMax.y, v.BBMin.y
	}
	if v.BBMin.z > v.BBMax.z {
		v.BBMin.z, v.BBMax.z = v.BBMax.z, v.BBMin.z
	}

	// normalize the triangle position to the voxel grid position
	V1 = V1.Sub(v.BBMin)
	V2 = V2.Sub(v.BBMin)
	V3 = V3.Sub(v.BBMin)

	xStep := (v.BBMax.x - v.BBMin.x) / float32(v.Resolution)
	yStep := (v.BBMax.y - v.BBMin.y) / float32(v.Resolution)
	zStep := (v.BBMax.z - v.BBMin.z) / float32(v.Resolution)

	// Calculate axis-aligned bounding box of the triangle
	minX := math32.Min(V1.x, math32.Min(V2.x, V3.x))
	minY := math32.Min(V1.y, math32.Min(V2.y, V3.y))
	minZ := math32.Min(V1.z, math32.Min(V2.z, V3.z))

	maxX := math32.Max(V1.x, math32.Max(V2.x, V3.x))
	maxY := math32.Max(V1.y, math32.Max(V2.y, V3.y))
	maxZ := math32.Max(V1.z, math32.Max(V2.z, V3.z))

	// Convert to voxel indices
	startX := int(minX / xStep)
	startY := int(minY / yStep)
	startZ := int(minZ / zStep)

	endX := int(maxX/xStep) + 1
	endY := int(maxY/yStep) + 1
	endZ := int(maxZ/zStep) + 1

	// Clamp to grid boundaries
	startX = max(0, min(startX, v.Resolution-1))
	startY = max(0, min(startY, v.Resolution-1))
	startZ = max(0, min(startZ, v.Resolution-1))

	endX = max(0, min(endX, v.Resolution))
	endY = max(0, min(endY, v.Resolution))
	endZ = max(0, min(endZ, v.Resolution))

	// Create triangle planes for intersection test
	edge1 := V2.Sub(V1)
	edge2 := V3.Sub(V1)
	triangleNormal := edge1.Cross(edge2).Normalize()

	// Loop through each potential voxel
	for x := startX; x < endX; x++ {
		for y := startY; y < endY; y++ {
			for z := startZ; z < endZ; z++ {
				// Get voxel center in normalized space
				voxelCenter := Vector{
					x: (float32(x) + 0.5) * xStep,
					y: (float32(y) + 0.5) * yStep,
					z: (float32(z) + 0.5) * zStep,
				}

				// Triangle-box overlap test
				// Using a simplified approach: check if the voxel center is close enough to the triangle plane
				distToPlane := triangleNormal.Dot(voxelCenter.Sub(V1))

				// If the voxel is close to the triangle plane
				if math32.Abs(distToPlane) <= math32.Max(xStep, math32.Max(yStep, zStep))*0.5 {
					// Use more accurate test: point in triangle projection
					// Project point onto triangle plane
					projectedPoint := voxelCenter.Sub(triangleNormal.Mul(distToPlane))

					// Check if the projected point is inside the triangle
					if pointInTriangle(projectedPoint, V1, V2, V3) {
						// Calculate index in the 1D grid
						index := x + y*v.Resolution + z*v.Resolution*v.Resolution

						// Check bounds
						if index >= 0 && index < len(v.Blocks) {
							// Set voxel properties (make it visible)
							randomR := rand.Float32() * 255
							randomG := rand.Float32() * 255
							randomB := rand.Float32() * 255
							v.Blocks[index].LightColor = ColorFloat32{randomR, randomG, randomB, 255}
						}
					}
				}
			}
		}
	}
}

// Helper function to check if a point is inside a triangle (using barycentric coordinates)
func pointInTriangle(p, a, b, c Vector) bool {
	// Compute vectors
	v0 := c.Sub(a)
	v1 := b.Sub(a)
	v2 := p.Sub(a)

	// Compute dot products
	dot00 := v0.Dot(v0)
	dot01 := v0.Dot(v1)
	dot02 := v0.Dot(v2)
	dot11 := v1.Dot(v1)
	dot12 := v1.Dot(v2)

	// Compute barycentric coordinates
	invDenom := 1.0 / (dot00*dot11 - dot01*dot01)
	u := (dot11*dot02 - dot01*dot12) * invDenom
	v := (dot00*dot12 - dot01*dot02) * invDenom

	// Check if point is inside triangle
	return (u >= 0) && (v >= 0) && (u+v <= 1)
}

func (v *VoxelGrid) CalculateLighting(samples int, depth int, light Light) {
	wg := sync.WaitGroup{}

	// Create mutex array for each block to prevent race conditions
	// mutexes := make([]sync.Mutex, len(v.Blocks))

	for i := range v.Blocks {
		wg.Add(1)
		go func(blockIndex int) {
			defer wg.Done()

			// Local accumulator for thread safety
			localColor := ColorFloat32{0, 0, 0, 0}

			for j := 0; j < samples; j++ {
				// Generate normalized random direction
				randomVector := Vector{
					x: rand.Float32()*2 - 1,
					y: rand.Float32()*2 - 1,
					z: rand.Float32()*2 - 1,
				}.Normalize()

				ray := Ray{
					origin:    v.Blocks[blockIndex].Position,
					direction: randomVector,
				}

				// Accumulate color locally
				localColor = localColor.Add(TraceRayV2(ray, depth, light, samples))
			}

			vBlocksBlockIndex := (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(blockIndex*48)))
			lightColorPtr := (*ColorFloat32)(unsafe.Pointer(&vBlocksBlockIndex.LightColor))
			// Unsafe assignment
			*lightColorPtr = localColor

		}(i) // Pass i directly to avoid closure issues
	}

	wg.Wait()
}

func (v *VoxelGrid) GetBlock(pos Vector) (Block, bool) {
	xStep := (v.BBMax.x - v.BBMin.x) / float32(v.Resolution)
	yStep := (v.BBMax.y - v.BBMin.y) / float32(v.Resolution)
	zStep := (v.BBMax.z - v.BBMin.z) / float32(v.Resolution)

	x := int((pos.x - v.BBMin.x) / xStep)
	y := int((pos.y - v.BBMin.y) / yStep)
	z := int((pos.z - v.BBMin.z) / zStep)

	// Ensure indices are within bounds
	if x < 0 || x >= v.Resolution || y < 0 || y >= v.Resolution || z < 0 || z >= v.Resolution {
		return Block{}, false
	}

	// Calculate the 1D index for the 3D grid
	index := x + y*v.Resolution + z*v.Resolution*v.Resolution

	// Return the block at the calculated index
	return v.Blocks[index], true
}

func (v *VoxelGrid) GetBlockUnsafe(pos Vector) (Block, bool) {
	xStep := (v.BBMax.x - v.BBMin.x) / float32(v.Resolution)
	yStep := (v.BBMax.y - v.BBMin.y) / float32(v.Resolution)
	zStep := (v.BBMax.z - v.BBMin.z) / float32(v.Resolution)

	x := int((pos.x - v.BBMin.x) / xStep)
	y := int((pos.y - v.BBMin.y) / yStep)
	z := int((pos.z - v.BBMin.z) / zStep)

	// Ensure indices are within bounds
	if x < 0 || x >= v.Resolution || y < 0 || y >= v.Resolution || z < 0 || z >= v.Resolution {
		return Block{}, false
	}

	// Calculate the 1D index for the 3D grid
	index := x + y*v.Resolution + z*v.Resolution*v.Resolution

	return *(*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(index*44))), true
}

func (v *VoxelGrid) GetVoxelUnsafe(pos Vector) (Block, bool) {
	xStep := (v.BBMax.x - v.BBMin.x) / float32(v.Resolution)
	yStep := (v.BBMax.y - v.BBMin.y) / float32(v.Resolution)
	zStep := (v.BBMax.z - v.BBMin.z) / float32(v.Resolution)

	x := int((pos.x - v.BBMin.x) / xStep)
	y := int((pos.y - v.BBMin.y) / yStep)
	z := int((pos.z - v.BBMin.z) / zStep)

	// Ensure indices are within bounds
	if x < 0 || x >= v.Resolution || y < 0 || y >= v.Resolution || z < 0 || z >= v.Resolution {
		return Block{}, false
	}

	// Calculate the 1D index for the 3D grid
	index := x + y*v.Resolution + z*v.Resolution*v.Resolution

	block := *(*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(index*44)))
	if block.LightColor.A == 0 {
		return Block{}, false
	}
	return block, true
}

// func (v *VoxelGrid) IntersectVoxels(ray Ray, light Light, steps int, Intensity float32) ColorFloat32 {
// 	hit, entry, exit := BoundingBoxCollisionEntryExitPoint(v.BBMax, v.BBMin, ray)
// 	if !hit {
// 		return ColorFloat32{}
// 	}

// 	maxDist := exit.Sub(entry).Length()
// 	currentDist := float32(0)

// 	for currentDist < maxDist && steps > 0 {
// 		block, exists := v.GetBlockUnsafe(entry)
// 		if !exists {
// 			// Step by minimum safe distance or stepSize, whichever is larger
// 			stepDist := block.MinStepDist
// 			if stepDist < float32(0.1) {
// 				stepDist = 0.1 // Minimum step to avoid getting stuck
// 			}
// 			entry = entry.Add(ray.direction.Mul(stepDist))
// 			currentDist += stepDist
// 			steps--
// 			continue
// 		}

// 		// Hit something - calculate lighting
// 		// distanceToLight := Vector{
// 		// 	light.Position.x - block.Position.x,
// 		// 	light.Position.y - block.Position.y,
// 		// 	light.Position.z - block.Position.z,
// 		// }.Length()
// 		// lightIntensity := 1.0 / (distanceToLight * distanceToLight / (light.intensity * light.intensity))

// 		return block.LightColor
// 	}

// 	return ColorFloat32{}
// }

type VolumeMaterial struct {
	transmittance float32
	density       float32
}

func ExpDecay(x float32) float32 {
	const k = 1.0 / (math.MaxFloat32 / 64) // Adjusting k so that f(MaxFloat32) ≈ 0
	z := float32(math.Exp(-float64(k) * float64(x)))
	if z > 1 {
		return 1
	}
	return z
}

func (v *VoxelGrid) IntersectVoxel(ray Ray, steps int, light Light) (ColorFloat32, bool) {
	hit, entry, exit := BoundingBoxCollisionEntryExitPoint(v.BBMax, v.BBMin, ray)
	if !hit {
		return ColorFloat32{}, false
	}

	stepSize := exit.Sub(entry).Mul(1.0 / float32(steps))

	currentPos := entry
	for i := 0; i < steps; i++ {
		block, exists := v.GetVoxelUnsafe(currentPos)
		if exists {
			// calculate shadows
			lightStep := light.Position.Sub(currentPos).Mul(1.0 / float32(steps*2))
			lightPos := currentPos.Add(lightStep)
			for j := 0; j < steps; j++ {
				_, exists := v.GetVoxelUnsafe(lightPos)
				if exists {
					return block.LightColor.MulScalar(0.05), true
				}
				lightPos = lightPos.Add(lightStep)
			}
			lightDistamce := light.Position.Sub(currentPos).Length()
			k := ExpDecay(lightDistamce)
			blockColor := block.LightColor.MulScalar(k)
			// blockColor.R *= light.Color[0]
			// blockColor.G *= light.Color[1]
			// blockColor.B *= light.Color[2]
			return blockColor, true
		}
		currentPos = currentPos.Add(stepSize)
	}
	return ColorFloat32{}, false
}

func (v *VoxelGrid) SetVoxelColor(color ColorFloat32, ray Ray, steps int) {
	_, hit, blockPtr := v.IntersectVoxelGetRefrence(ray, steps, Light{})
	if hit {
		// fmt.Println("Hit Color Before:", c)
		// blockPtr.LightColor = color
		// Assign the color to the block unsafely
		lightColorPtr := (*ColorFloat32)(unsafe.Pointer(&blockPtr.LightColor))
		*lightColorPtr = color
		// fmt.Println("Hit Color After:", *lightColorPtr)
	}
}

func (v *VoxelGrid) RemoveVoxel(ray Ray, steps int) {
	_, hit, blockPtr := v.IntersectVoxelGetRefrence(ray, steps, Light{})
	if hit {
		// blockPtr.LightColor = ColorFloat32{0, 0, 0, 0}
		// Assign the color to the block unsafely
		lightColorPtr := (*ColorFloat32)(unsafe.Pointer(&blockPtr.LightColor))
		*lightColorPtr = ColorFloat32{0, 0, 0, 0}
	}
}

func (v *VoxelGrid) AddVoxel(ray Ray, steps int, color ColorFloat32) {
	_, _, _ = v.IntersectVoxelAdd(ray, steps, Light{}, color)
}

func (v *VoxelGrid) IntersectVoxelAdd(ray Ray, steps int, light Light, col ColorFloat32) (ColorFloat32, bool, *Block) {
	hit, entry, exit := BoundingBoxCollisionEntryExitPoint(v.BBMax, v.BBMin, ray)
	if !hit {
		return ColorFloat32{}, false, nil
	}

	stepSize := exit.Sub(entry).Mul(1.0 / float32(steps))

	currentPos := entry
	for i := 0; i < steps; i++ {
		blockPtr, exists := v.GetVoxelUnsafeAdd(currentPos, col)
		if exists {
			// calculate shadows
			lightStep := light.Position.Sub(currentPos).Mul(1.0 / float32(steps*2))
			lightPos := currentPos.Add(lightStep)
			for j := 0; j < steps; j++ {
				_, exists := v.GetVoxelUnsafeRefrence(lightPos)
				if exists {
					return blockPtr.LightColor.MulScalar(0.05), true, blockPtr
				}
				lightPos = lightPos.Add(lightStep)
			}
			lightDistamce := light.Position.Sub(currentPos).Length()
			k := ExpDecay(lightDistamce)
			blockColor := blockPtr.LightColor.MulScalar(k)
			// blockColor.R *= light.Color[0]
			// blockColor.G *= light.Color[1]
			// blockColor.B *= light.Color[2]
			return blockColor, true, blockPtr
		}
		currentPos = currentPos.Add(stepSize)
	}
	return ColorFloat32{}, false, nil
}

func (v *VoxelGrid) IntersectVoxelGetRefrence(ray Ray, steps int, light Light) (ColorFloat32, bool, *Block) {
	hit, entry, exit := BoundingBoxCollisionEntryExitPoint(v.BBMax, v.BBMin, ray)
	if !hit {
		return ColorFloat32{}, false, nil
	}

	stepSize := exit.Sub(entry).Mul(1.0 / float32(steps))

	currentPos := entry
	for i := 0; i < steps; i++ {
		blockPtr, exists := v.GetVoxelUnsafeRefrence(currentPos)
		if exists {
			// calculate shadows
			lightStep := light.Position.Sub(currentPos).Mul(1.0 / float32(steps*2))
			lightPos := currentPos.Add(lightStep)
			for j := 0; j < steps; j++ {
				_, exists := v.GetVoxelUnsafeRefrence(lightPos)
				if exists {
					return blockPtr.LightColor.MulScalar(0.05), true, blockPtr
				}
				lightPos = lightPos.Add(lightStep)
			}
			lightDistamce := light.Position.Sub(currentPos).Length()
			k := ExpDecay(lightDistamce)
			blockColor := blockPtr.LightColor.MulScalar(k)
			// blockColor.R *= light.Color[0]
			// blockColor.G *= light.Color[1]
			// blockColor.B *= light.Color[2]
			return blockColor, true, blockPtr
		}
		currentPos = currentPos.Add(stepSize)
	}
	return ColorFloat32{}, false, nil
}

func (v *VoxelGrid) GetVoxelUnsafeRefrence(pos Vector) (*Block, bool) {
	xStep := (v.BBMax.x - v.BBMin.x) / float32(v.Resolution)
	yStep := (v.BBMax.y - v.BBMin.y) / float32(v.Resolution)
	zStep := (v.BBMax.z - v.BBMin.z) / float32(v.Resolution)

	x := int((pos.x - v.BBMin.x) / xStep)
	y := int((pos.y - v.BBMin.y) / yStep)
	z := int((pos.z - v.BBMin.z) / zStep)

	// Ensure indices are within bounds
	if x < 0 || x >= v.Resolution || y < 0 || y >= v.Resolution || z < 0 || z >= v.Resolution {
		return nil, false
	}

	// Calculate the 1D index for the 3D grid
	index := x + y*v.Resolution + z*v.Resolution*v.Resolution

	// Get a direct pointer to the block in the underlying array
	blockPtr := (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(index*44)))
	if blockPtr.LightColor.A == 0 {
		return nil, false
	}

	// Return the pointer directly, not the address of a local copy
	return blockPtr, true
}

func (v *VoxelGrid) GetVoxelUnsafeAdd(pos Vector, col ColorFloat32) (*Block, bool) {
	xStep := (v.BBMax.x - v.BBMin.x) / float32(v.Resolution)
	yStep := (v.BBMax.y - v.BBMin.y) / float32(v.Resolution)
	zStep := (v.BBMax.z - v.BBMin.z) / float32(v.Resolution)

	x := int((pos.x - v.BBMin.x) / xStep)
	y := int((pos.y - v.BBMin.y) / yStep)
	z := int((pos.z - v.BBMin.z) / zStep)

	// Ensure indices are within bounds
	if x < 0 || x >= v.Resolution || y < 0 || y >= v.Resolution || z < 0 || z >= v.Resolution {
		return nil, false
	}

	// Calculate the 1D index for the 3D grid
	index := x + y*v.Resolution + z*v.Resolution*v.Resolution

	// Get a direct pointer to the block in the underlying array
	blockPtr := (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(index*44)))
	if blockPtr.LightColor.A == 0 {
		return nil, false
	}

	for i := -1; i < 2; i++ {
		for j := -1; j < 2; j++ {
			for k := -1; k < 2; k++ {
				index := (x + i) + (y+j)*v.Resolution + (z+k)*v.Resolution*v.Resolution
				if index >= 0 && index < len(v.Blocks) {
					blockPtr := (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(index*44)))
					lightColorPtr := (*ColorFloat32)(unsafe.Pointer(&blockPtr.LightColor))
					*lightColorPtr = col
				}
			}
		}
	}

	// Return the pointer directly, not the address of a local copy
	return blockPtr, true
}

func (v *VoxelGrid) GetVoxelUnsafeRefrenceNoCheck(pos Vector) *Block {
	xStep := (v.BBMax.x - v.BBMin.x) / float32(v.Resolution)
	yStep := (v.BBMax.y - v.BBMin.y) / float32(v.Resolution)
	zStep := (v.BBMax.z - v.BBMin.z) / float32(v.Resolution)

	x := int((pos.x - v.BBMin.x) / xStep)
	y := int((pos.y - v.BBMin.y) / yStep)
	z := int((pos.z - v.BBMin.z) / zStep)

	// Ensure indices are within bounds
	if x < 0 || x >= v.Resolution || y < 0 || y >= v.Resolution || z < 0 || z >= v.Resolution {
		return nil
	}

	// Calculate the 1D index for the 3D grid
	index := x + y*v.Resolution + z*v.Resolution*v.Resolution

	// Get a direct pointer to the block in the underlying array
	blockPtr := (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(index*44)))

	// Return the pointer directly, not the address of a local copy
	return blockPtr
}

// func (v *VoxelGrid) GetVoxelUnsafeRefrenceNeighbors(pos Vector) (Top *Block, Bottom *Block, Left *Block, Right *Block, Front *Block, Back *Block) {
// 	// Calculate base grid indices for the position
// 	xStep := (v.BBMax.x - v.BBMin.x) / float32(v.Resolution)
// 	yStep := (v.BBMax.y - v.BBMin.y) / float32(v.Resolution)
// 	zStep := (v.BBMax.z - v.BBMin.z) / float32(v.Resolution)

// 	x := int((pos.y - v.BBMin.y) / yStep)
// 	y := int((pos.z - v.BBMin.z) / zStep)
// 	z := int((pos.x - v.BBMin.x) / xStep)

// 	// Calculate neighbor indices directly from the base indices
// 	x1 := x - 1 // left
// 	x2 := x + 1 // right
// 	y1 := y - 1 // front
// 	y2 := y + 1 // back
// 	z1 := z - 1 // bottom
// 	z2 := z + 1 // top

// 	// Create variables for each neighbor
// 	var topBlock, bottomBlock, leftBlock, rightBlock, frontBlock, backBlock *Block

// 	// Check each neighbor individually and assign if it's in bounds
// 	if z2 >= 0 && z2 < v.Resolution {
// 		topIndex := x + y*v.Resolution + z2*v.Resolution*v.Resolution
// 		topBlock = (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(topIndex*44)))
// 	}

// 	if z1 >= 0 && z1 < v.Resolution {
// 		bottomIndex := x + y*v.Resolution + z1*v.Resolution*v.Resolution
// 		bottomBlock = (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(bottomIndex*44)))
// 	}

// 	if x1 >= 0 && x1 < v.Resolution {
// 		leftIndex := x1 + y*v.Resolution + z*v.Resolution*v.Resolution
// 		leftBlock = (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(leftIndex*44)))
// 	}

// 	if x2 >= 0 && x2 < v.Resolution {
// 		rightIndex := x2 + y*v.Resolution + z*v.Resolution*v.Resolution
// 		rightBlock = (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(rightIndex*44)))
// 	}

// 	if y1 >= 0 && y1 < v.Resolution {
// 		frontIndex := x + y1*v.Resolution + z*v.Resolution*v.Resolution
// 		frontBlock = (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(frontIndex*44)))
// 	}

// 	if y2 >= 0 && y2 < v.Resolution {
// 		backIndex := x + y2*v.Resolution + z*v.Resolution*v.Resolution
// 		backBlock = (*Block)(unsafe.Pointer(uintptr(v.BlocksPointer) + uintptr(backIndex*44)))
// 	}

// 	// Return all neighbors
// 	return topBlock, bottomBlock, leftBlock, rightBlock, frontBlock, backBlock
// }

// func (v *VoxelGrid) IntersectVoxel(ray Ray, steps int, light Light) (ColorFloat32, bool) {
// 	hit, entry, exit := BoundingBoxCollisionEntryExitPoint(v.BBMax, v.BBMin, ray)
// 	if !hit {
// 		return ColorFloat32{}, false
// 	}

// 	stepSize := exit.Sub(entry).Mul(1.0 / float32(steps))

// 	currentPos := entry
// 	for i := 0; i < steps; i++ {
// 		block, exists := v.GetVoxelUnsafe(currentPos)
// 		if exists {
// 			// Calculate basic lighting components
// 			// lightDir := light.Position.Sub(currentPos).Normalize()
// 			lightDistance := light.Position.Sub(currentPos).Length()

// 			// Inverse square falloff for more physically accurate lighting
// 			attenuationFactor := light.intensity / (1.0 + 0.01*lightDistance*lightDistance)

// 			// Add ambient component to prevent completely dark shadows
// 			const ambientFactor = float32(0.2)

// 			// Shadow calculation
// 			shadowIntensity := float32(1.0)
// 			lightStep := light.Position.Sub(currentPos).Mul(1.0 / float32(steps*2))
// 			lightPos := currentPos.Add(lightStep)
// 			for j := 0; j < steps; j++ {
// 				_, exists := v.GetVoxelUnsafe(lightPos)
// 				if exists {
// 					// Reduce light intensity but don't eliminate it completely
// 					shadowIntensity = 0.25
// 					break
// 				}
// 				lightPos = lightPos.Add(lightStep)
// 			}

// 			// Apply light color to voxel color for more realistic colored lighting
// 			finalColor := ColorFloat32{
// 				R: block.LightColor.R * (ambientFactor + (1.0-ambientFactor)*attenuationFactor*shadowIntensity),
// 				G: block.LightColor.G * (ambientFactor + (1.0-ambientFactor)*attenuationFactor*shadowIntensity),
// 				B: block.LightColor.B * (ambientFactor + (1.0-ambientFactor)*attenuationFactor*shadowIntensity),
// 				A: block.LightColor.A,
// 			}

// 			return finalColor, true
// 		}
// 		currentPos = currentPos.Add(stepSize)
// 	}
// 	return ColorFloat32{}, false
// }

// Original intersection method with fixes for color and transparency
func (v *VoxelGrid) Intersect(ray Ray, steps int, light Light, volumeMaterail VolumeMaterial) ColorFloat32 {
	hit, entry, exit := BoundingBoxCollisionEntryExitPoint(v.BBMax, v.BBMin, ray)
	if !hit {
		return ColorFloat32{}
	}

	// Physical constants - adjusted for better visibility
	const (
		extinctionCoeff  = 0.5          // Reduced from 0.5 for less extinction
		scatteringAlbedo = 0.9          // Single scattering albedo
		asymmetryParam   = float32(0.3) // Henyey-Greenstein asymmetry parameter
		temperatureScale = 0.001        // Temperature influence on density
	)

	stepSize := exit.Sub(entry).Mul(1.0 / float32(steps))
	stepLength := stepSize.Length()

	var accumColor ColorFloat32
	transmittance := volumeMaterail.transmittance

	currentPos := entry
	for i := 0; i < steps; i++ {
		block, exists := v.GetBlockUnsafe(currentPos)
		if !exists {
			currentPos = currentPos.Add(stepSize)
			continue
		}

		density := volumeMaterail.density
		extinction := density * extinctionCoeff

		// Calculate light direction and phase function
		lightDir := light.Position.Sub(currentPos).Normalize()
		cosTheta := ray.direction.Dot(lightDir)
		g := asymmetryParam
		phaseFunction := (1.0 - g*g) / (4.0 * math32.Pi * math32.Pow(1.0+g*g-2.0*g*cosTheta, 1.5))

		// Calculate light contribution through volume
		lightRay := Ray{origin: currentPos, direction: lightDir}
		lightTransmittance := v.calculateLightTransmittance(lightRay, light, density)

		// Increased scattering for better visibility
		scattering := extinction * scatteringAlbedo * phaseFunction * 2.0

		// Apply Beer-Lambert law with adjusted extinction
		sampleExtinction := math32.Exp(-extinction * stepLength)
		transmittance *= sampleExtinction

		// Calculate color contribution with enhanced intensity
		lightContribution := ColorFloat32{
			R: block.SmokeColor.R * light.Color[0] * lightTransmittance * scattering,
			G: block.SmokeColor.G * light.Color[1] * lightTransmittance * scattering,
			B: block.SmokeColor.B * light.Color[2] * lightTransmittance * scattering,
			A: block.SmokeColor.A * density, // Tie alpha to density
		}

		// Accumulate color with transmittance
		accumColor = accumColor.Add(lightContribution.MulScalar(transmittance))

		// Adjusted early exit threshold
		if transmittance < 0.001 {
			break
		}

		currentPos = currentPos.Add(stepSize)
	}

	// Ensure final color has some opacity
	accumColor.A = math32.Min(accumColor.A, 1.0)
	return accumColor
}

func (v *VoxelGrid) calculateLightTransmittance(ray Ray, light Light, intensity float32) float32 {
	hit, entry, exit := BoundingBoxCollisionEntryExitPoint(v.BBMax, v.BBMin, ray)
	if !hit {
		return 1.0
	}

	const lightSamples = 16 // Increased from 8 for better quality
	stepSize := exit.Sub(entry).Mul(1.0 / float32(lightSamples))
	stepLength := stepSize.Length()

	transmittance := float32(1.0)
	currentPos := entry

	for i := 0; i < lightSamples; i++ {
		_, exists := v.GetBlockUnsafe(currentPos)
		if exists {
			extinction := intensity * 0.05 // Reduced extinction coefficient
			transmittance *= math32.Exp(-extinction * stepLength)
		}
		currentPos = currentPos.Add(stepSize)
	}

	return transmittance
}

func DrawRaysBlockVoxelGrid(camera Camera, scaling int, samples int, blocks []BlocksImage, voxelGrid *VoxelGrid, light Light, volumeMaterial VolumeMaterial) {
	var wg sync.WaitGroup
	for i := 0; i < len(blocks); i++ {
		wg.Add(1)
		go func(blockIndex int) {
			defer wg.Done()
			block := blocks[blockIndex]
			for y := block.startY; y < block.endY; y += 1 {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x += 1 {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c := voxelGrid.Intersect(Ray{origin: camera.Position, direction: rayDir}, samples, light, volumeMaterial)

					// Write the pixel color to the pixel buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.pixelBuffer[index] = clampUint8(c.R)
					block.pixelBuffer[index+1] = clampUint8(c.G)
					block.pixelBuffer[index+2] = clampUint8(c.B)
					block.pixelBuffer[index+3] = clampUint8(c.A)
				}
			}
			block.image.WritePixels(block.pixelBuffer)
		}(i)
	}

	if performanceOptions.Selected == 0 {
		wg.Wait()
	}
}

func DrawRaysBlockVoxels(camera Camera, scaling int, samples int, blocks []BlocksImage, voxelGrid *VoxelGrid, light Light, volumeMaterial VolumeMaterial) {
	var wg sync.WaitGroup
	for i := 0; i < len(blocks); i++ {
		wg.Add(1)
		go func(blockIndex int) {
			defer wg.Done()
			block := blocks[blockIndex]
			for y := block.startY; y < block.endY; y += 1 {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x += 1 {
					if x*scaling >= screenWidth {
						continue
					}
					rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					c, _ := voxelGrid.IntersectVoxel(Ray{origin: camera.Position, direction: rayDir}, 1024, light)

					// Write the pixel color to the pixel buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					block.pixelBuffer[index] = clampUint8(c.R)
					block.pixelBuffer[index+1] = clampUint8(c.G)
					block.pixelBuffer[index+2] = clampUint8(c.B)
					block.pixelBuffer[index+3] = clampUint8(c.A)
				}
			}
			block.image.WritePixels(block.pixelBuffer)
		}(i)
	}

	if performanceOptions.Selected == 0 {
		wg.Wait()
	}
}

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
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"image/png"

	"github.com/chewxy/math32"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

const screenWidth = 800
const screenHeight = 608
const rowSize = screenHeight / numCPU
const FOV = 90

const maxDepth = 16
const numCPU = 16

type Material struct {
	name  string
	color color.RGBA
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
			mat.color = color.RGBA{
				R: uint8(r * 255),
				G: uint8(g * 255),
				B: uint8(b * 255),
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
					triangle := Triangle{
						v1:         vertices[indices[0]],
						v2:         vertices[indices[i]],
						v3:         vertices[indices[i+1]],
						reflection: 0.09,
					}

					// Apply the current material color if available
					if mat, exists := materials[currentMaterial]; exists {
						triangle.color = mat.color
						triangle.color.A = 255 // Ensure alpha is fully opaque
					} else {
						triangle.color = color.RGBA{255, 125, 0, 255} // Default color
					}

					triangle.CalculateBoundingBox()
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

func (v Vector) Length() float32 {
	return math32.Sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
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
	magnitude := math32.Sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
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

type Ray struct {
	origin, direction Vector
}

type Triangle struct {
	v1, v2, v3  Vector
	color       color.RGBA
	BoundingBox [2]Vector
	Normal      Vector
	reflection  float32
}

func (t *Triangle) CalculateNormal() {
	edge1 := t.v2.Sub(t.v1)
	edge2 := t.v3.Sub(t.v1)
	t.Normal = edge1.Cross(edge2).Normalize()
}

func BoundingBoxCollision(BoundingBox *[2]Vector, ray *Ray) bool {
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
	return tmax >= max(0.0, tmin)
}

func BoundingBoxCollisionDistance(BoundingBox *[2]Vector, ray *Ray) (bool, float32) {
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

func (triangle *Triangle) Rotate(xAngle, yAngle, zAngle float32) {
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

func CreateCube(center Vector, size float32, color color.RGBA, refection float32) []Triangle {
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

func CreatePlane(center Vector, normal Vector, width, height float32, color color.RGBA, reflection float32) []Triangle {
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

	return []Triangle{
		NewTriangle(v1, v2, v3, color, reflection),
		NewTriangle(v1, v3, v4, color, reflection),
	}
}

func CreateSphere(center Vector, radius float32, color color.RGBA, reflection float32) []Triangle {
	var triangles []Triangle
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

			triangles = append(triangles, NewTriangle(Vector{x0 + center.x, y0 + center.y, z0 + center.z}, Vector{x1 + center.x, y1 + center.y, z0 + center.z}, Vector{x2 + center.x, y2 + center.y, z1 + center.z}, color, reflection))
			triangles = append(triangles, NewTriangle(Vector{x1 + center.x, y1 + center.y, z0 + center.z}, Vector{x3 + center.x, y3 + center.y, z1 + center.z}, Vector{x2 + center.x, y2 + center.y, z1 + center.z}, color, reflection))
		}
	}

	return triangles
}

func (triangle *Triangle) CalculateBoundingBox() {
	// Compute the minimum and maximum coordinates using float32 functions
	minX := math32.Min(triangle.v1.x, math32.Min(triangle.v2.x, triangle.v3.x))
	minY := math32.Min(triangle.v1.y, math32.Min(triangle.v2.y, triangle.v3.y))
	minZ := math32.Min(triangle.v1.z, math32.Min(triangle.v2.z, triangle.v3.z))
	maxX := math32.Max(triangle.v1.x, math32.Max(triangle.v2.x, triangle.v3.x))
	maxY := math32.Max(triangle.v1.y, math32.Max(triangle.v2.y, triangle.v3.y))
	maxZ := math32.Max(triangle.v1.z, math32.Max(triangle.v2.z, triangle.v3.z))

	// Set the BoundingBox with computed min and max values
	triangle.BoundingBox[0] = Vector{minX, minY, minZ}
	triangle.BoundingBox[1] = Vector{maxX, maxY, maxZ}
}

func NewTriangle(v1, v2, v3 Vector, color color.RGBA, reflection float32) Triangle {
	triangle := Triangle{v1: v1, v2: v2, v3: v3, color: color, reflection: reflection}
	triangle.CalculateBoundingBox()
	triangle.CalculateNormal()
	return triangle
}

func (triangle *Triangle) IntersectBoundingBox(ray Ray) bool {
	// Precompute the inverse direction
	invDirX := 1.0 / ray.direction.x
	invDirY := 1.0 / ray.direction.y
	invDirZ := 1.0 / ray.direction.z

	// Compute the tmin and tmax for each axis directly
	tx1 := (triangle.BoundingBox[0].x - ray.origin.x) * invDirX
	tx2 := (triangle.BoundingBox[1].x - ray.origin.x) * invDirX
	tmin := min(tx1, tx2)
	tmax := max(tx1, tx2)

	ty1 := (triangle.BoundingBox[0].y - ray.origin.y) * invDirY
	ty2 := (triangle.BoundingBox[1].y - ray.origin.y) * invDirY
	tmin = max(tmin, min(ty1, ty2))
	tmax = min(tmax, max(ty1, ty2))

	tz1 := (triangle.BoundingBox[0].z - ray.origin.z) * invDirZ
	tz2 := (triangle.BoundingBox[1].z - ray.origin.z) * invDirZ
	tmin = max(tmin, min(tz1, tz2))
	tmax = min(tmax, max(tz1, tz2))

	// Final intersection check
	return tmax >= max(0.0, tmin)
}

type Intersection struct {
	PointOfIntersection Vector
	Color               color.RGBA
	Normal              Vector
	Direction           Vector
	Distance            float32
	reflection          float32
}

type Light struct {
	Position  *Vector
	Color     *[3]float32
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
func (ray *Ray) IntersectBVH(nodeBVH *BVHNode) (Intersection, bool) {
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
		if currentNode.Triangles != nil {
			intersection, intersects := IntersectTriangles(*ray, *currentNode.Triangles)
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

func (ray *Ray) IntersectTriangle(triangle Triangle) (Intersection, bool) {
	// Check if the ray intersects the bounding box of the triangle first
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
		return Intersection{PointOfIntersection: ray.origin.Add(ray.direction.Mul(t)), Color: triangle.color, Normal: triangle.Normal, Direction: ray.direction, Distance: t, reflection: triangle.reflection}, true
	}
	return Intersection{}, false
}

func IntersectTriangles(ray Ray, triangles []Triangle) (Intersection, bool) {
	// Initialize the closest intersection and hit status
	closestIntersection := Intersection{Distance: math32.MaxFloat32}
	hasIntersection := false

	// Iterate over each triangle for the given ray
	for _, triangle := range triangles {
		// Check if the ray intersects the bounding box of the triangle first
		if !triangle.IntersectBoundingBox(ray) {
			continue
		}

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
	Position            Vector
	xAxis, yAxis, zAxis float32
}

type Pixel struct {
	x, y  int
	color color.RGBA
}

func (intersection *Intersection) Scatter(samples int, light Light, bvh *BVHNode) color.RGBA {
	var Red, Green, Blue float32

	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	reflectDir := intersection.Normal.Mul(2 * lightDir.Dot(intersection.Normal)).Sub(lightDir).Normalize()
	reflectRay := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: reflectDir}

	reflectedIntersection, _ := reflectRay.IntersectBVH(bvh)

	for i := 0; i < samples; i++ {
		u := rand.Float32()
		v := rand.Float32()
		r := math32.Sqrt(u)
		theta := 2 * math32.Pi * v

		var uVec Vector
		if math32.Abs(intersection.Normal.x) > 0.1 {
			uVec = Vector{0.0, 1.0, 0.0}
		} else {
			uVec = Vector{1.0, 0.0, 0.0}
		}
		uVec = uVec.Cross(intersection.Normal).Normalize()
		vVec := intersection.Normal.Cross(uVec)

		directionLocal := uVec.Mul(r * math32.Cos(theta)).Add(vVec.Mul(r * math32.Sin(theta))).Add(intersection.Normal.Mul(math32.Sqrt(1 - u)))
		direction := directionLocal.Normalize()

		ray := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: direction}

		if bvhIntersection, intersect := ray.IntersectBVH(bvh); intersect && bvhIntersection.Distance != math32.MaxFloat32 {
			Red += float32(bvhIntersection.Color.R)
			Green += float32(bvhIntersection.Color.G)
			Blue += float32(bvhIntersection.Color.B)
		}
	}

	if samples > 0 {
		s := float32(samples)
		Red /= s
		Green /= s
		Blue /= s

		ratioScatterToDirect := 1 - intersection.reflection
		return color.RGBA{
			R: clampUint8(Red*ratioScatterToDirect + float32(reflectedIntersection.Color.R)*intersection.reflection),
			G: clampUint8(Green*ratioScatterToDirect + float32(reflectedIntersection.Color.G)*intersection.reflection),
			B: clampUint8(Blue*ratioScatterToDirect + float32(reflectedIntersection.Color.B)*intersection.reflection),
			A: intersection.Color.A,
		}
	}
	return color.RGBA{}
}

func TraceRay(ray Ray, depth int, light Light, scatter int) color.RGBA {
	if depth == 0 {
		return color.RGBA{}
	}

	intersection, intersect := ray.IntersectBVH(BVH)
	if !intersect {
		return color.RGBA{}
	}

	scatteredColor := intersection.Scatter(scatter, light, BVH)

	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	reflectDir := intersection.Normal.Mul(2 * lightDir.Dot(intersection.Normal)).Sub(lightDir).Normalize()
	reflectRay := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: reflectDir}

	reflectedColor := color.RGBA{}
	if tempIntersection, intersect := reflectRay.IntersectBVH(BVH); intersect {
		reflectedColor.R = uint8(tempIntersection.Color.R)
		reflectedColor.G = uint8(tempIntersection.Color.G)
		reflectedColor.B = uint8(tempIntersection.Color.B)
	}

	shadowRay := Ray{
		origin:    intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)),
		direction: light.Position.Sub(intersection.PointOfIntersection).Normalize(),
	}
	_, inShadow := shadowRay.IntersectBVH(BVH)

	var directColor color.RGBA
	if !inShadow {
		lightIntensity := light.intensity * math32.Max(0.0, lightDir.Dot(intersection.Normal))
		directColor = color.RGBA{
			R: clampUint8((float32(scatteredColor.R) + float32(intersection.Color.R)) * lightIntensity * float32(light.Color[0])),
			G: clampUint8((float32(scatteredColor.G) + float32(intersection.Color.G)) * lightIntensity * float32(light.Color[1])),
			B: clampUint8((float32(scatteredColor.B) + float32(intersection.Color.B)) * lightIntensity * float32(light.Color[2])),
			A: intersection.Color.A,
		}
	} else {
		lightIntensity := float32(0.05)
		directColor = color.RGBA{
			R: clampUint8((float32(scatteredColor.R) + float32(intersection.Color.R)) * lightIntensity * float32(light.Color[0])),
			G: clampUint8((float32(scatteredColor.G) + float32(intersection.Color.G)) * lightIntensity * float32(light.Color[1])),
			B: clampUint8((float32(scatteredColor.B) + float32(intersection.Color.B)) * lightIntensity * float32(light.Color[2])),
			A: intersection.Color.A,
		}
	}

	ratioScatterToDirect := 1 - intersection.reflection
	finalColor := color.RGBA{
		R: clampUint8(float32(directColor.R)*ratioScatterToDirect + float32(reflectedColor.R)*intersection.reflection),
		G: clampUint8(float32(directColor.G)*ratioScatterToDirect + float32(reflectedColor.G)*intersection.reflection),
		B: clampUint8(float32(directColor.B)*ratioScatterToDirect + float32(reflectedColor.B)*intersection.reflection),
		A: uint8(intersection.Color.A),
	}

	bounceRay := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: reflectDir}
	bouncedColor := TraceRay(bounceRay, depth-1, light, scatter)

	finalColor.R = clampUint8((float32(finalColor.R) + float32(bouncedColor.R)) / 2)
	finalColor.G = clampUint8((float32(finalColor.G) + float32(bouncedColor.G)) / 2)
	finalColor.B = clampUint8((float32(finalColor.B) + float32(bouncedColor.B)) / 2)

	return finalColor
}

type object struct {
	triangles   []Triangle
	BoundingBox [2]Vector
}

var triangles []Triangle

func ConvertObjectsToBVH(objects []object, maxDepth int) *BVHNode {
	for _, object := range objects {
		triangles = append(triangles, object.triangles...)
	}
	return buildBVHNode(triangles, 0, maxDepth)
}

type BVHNode struct {
	Left, Right *BVHNode
	BoundingBox *[2]Vector
	Triangles   *[]Triangle
}

func (object *object) BuildBVH(maxDepth int) *BVHNode {
	return buildBVHNode(object.triangles, 0, maxDepth)
}

func calculateSurfaceArea(bbox [2]Vector) float32 {
	dx := bbox[1].x - bbox[0].x
	dy := bbox[1].y - bbox[0].y
	dz := bbox[1].z - bbox[0].z
	return 2 * (dx*dy + dy*dz + dz*dx)
}

func buildBVHNode(triangles []Triangle, depth int, maxDepth int) *BVHNode {
	if len(triangles) == 0 {
		return nil
	}

	// Calculate the bounding box of the node
	boundingBox := [2]Vector{
		{math32.MaxFloat32, math32.MaxFloat32, math32.MaxFloat32},
		{-math32.MaxFloat32, -math32.MaxFloat32, -math32.MaxFloat32},
	}

	for _, triangle := range triangles {
		boundingBox[0].x = math32.Min(boundingBox[0].x, triangle.BoundingBox[0].x)
		boundingBox[0].y = math32.Min(boundingBox[0].y, triangle.BoundingBox[0].y)
		boundingBox[0].z = math32.Min(boundingBox[0].z, triangle.BoundingBox[0].z)

		boundingBox[1].x = math32.Max(boundingBox[1].x, triangle.BoundingBox[1].x)
		boundingBox[1].y = math32.Max(boundingBox[1].y, triangle.BoundingBox[1].y)
		boundingBox[1].z = math32.Max(boundingBox[1].z, triangle.BoundingBox[1].z)
	}

	// If the node is a leaf or we've reached the maximum depth
	if len(triangles) <= 2 || depth >= maxDepth {
		// Allocate the slice with the exact capacity needed
		node := &BVHNode{
			BoundingBox: &boundingBox,
			Triangles:   &triangles,
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
				return triangles[i].BoundingBox[0].x < triangles[j].BoundingBox[0].x
			})
		case 1:
			sort.Slice(triangles, func(i, j int) bool {
				return triangles[i].BoundingBox[0].y < triangles[j].BoundingBox[0].y
			})
		case 2:
			sort.Slice(triangles, func(i, j int) bool {
				return triangles[i].BoundingBox[0].z < triangles[j].BoundingBox[0].z
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
				leftBBox[0].x = math32.Min(leftBBox[0].x, triangles[j].BoundingBox[0].x)
				leftBBox[0].y = math32.Min(leftBBox[0].y, triangles[j].BoundingBox[0].y)
				leftBBox[0].z = math32.Min(leftBBox[0].z, triangles[j].BoundingBox[0].z)
				leftBBox[1].x = math32.Max(leftBBox[1].x, triangles[j].BoundingBox[1].x)
				leftBBox[1].y = math32.Max(leftBBox[1].y, triangles[j].BoundingBox[1].y)
				leftBBox[1].z = math32.Max(leftBBox[1].z, triangles[j].BoundingBox[1].z)
			}

			for j := i; j < len(triangles); j++ {
				rightBBox[0].x = math32.Min(rightBBox[0].x, triangles[j].BoundingBox[0].x)
				rightBBox[0].y = math32.Min(rightBBox[0].y, triangles[j].BoundingBox[0].y)
				rightBBox[0].z = math32.Min(rightBBox[0].z, triangles[j].BoundingBox[0].z)
				rightBBox[1].x = math32.Max(rightBBox[1].x, triangles[j].BoundingBox[1].x)
				rightBBox[1].y = math32.Max(rightBBox[1].y, triangles[j].BoundingBox[1].y)
				rightBBox[1].z = math32.Max(rightBBox[1].z, triangles[j].BoundingBox[1].z)
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
			return triangles[i].BoundingBox[0].x < triangles[j].BoundingBox[0].x
		})
	case 1:
		sort.Slice(triangles, func(i, j int) bool {
			return triangles[i].BoundingBox[0].y < triangles[j].BoundingBox[0].y
		})
	case 2:
		sort.Slice(triangles, func(i, j int) bool {
			return triangles[i].BoundingBox[0].z < triangles[j].BoundingBox[0].z
		})
	}

	// Create the BVH node with the best split
	node := &BVHNode{BoundingBox: &boundingBox}
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

func CreateObject(triangles []Triangle) *object {
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
		object.BoundingBox[0].x = math32.Min(object.BoundingBox[0].x, triangle.BoundingBox[0].x)
		object.BoundingBox[0].y = math32.Min(object.BoundingBox[0].y, triangle.BoundingBox[0].y)
		object.BoundingBox[0].z = math32.Min(object.BoundingBox[0].z, triangle.BoundingBox[0].z)

		// Update maximum coordinates (BoundingBox[1])
		object.BoundingBox[1].x = math32.Max(object.BoundingBox[1].x, triangle.BoundingBox[1].x)
		object.BoundingBox[1].y = math32.Max(object.BoundingBox[1].y, triangle.BoundingBox[1].y)
		object.BoundingBox[1].z = math32.Max(object.BoundingBox[1].z, triangle.BoundingBox[1].z)
	}
}

func GenerateRandomSpheres(numSpheres int) []object {
	spheres := make([]object, numSpheres)
	for i := 0; i < numSpheres; i++ {
		radius := rand.Float32()*50 + 10
		color := color.RGBA{uint8(rand.Intn(255)), uint8(rand.Intn(255)), uint8(rand.Intn(255)), 255}
		reflection := rand.Float32()
		position := Vector{rand.Float32()*400 - 200, rand.Float32()*400 - 200, rand.Float32()*400 - 200}
		sphere := CreateSphere(position, radius, color, reflection)
		spheres[i] = *CreateObject(sphere)
	}
	return spheres
}

func GenerateRandomCubes(numCubes int) []object {
	cubes := make([]object, numCubes)
	for i := 0; i < numCubes; i++ {
		size := rand.Float32()*50 + 10
		color := color.RGBA{uint8(rand.Intn(255)), uint8(rand.Intn(255)), uint8(rand.Intn(255)), 255}
		reflection := rand.Float32()
		position := Vector{rand.Float32()*400 - 200, rand.Float32()*400 - 200, rand.Float32()*400 - 200}
		cube := CreateCube(position, size, color, reflection)
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

func (object *object) ConvertToTriangles() []Triangle {
	triangles := []Triangle{}
	triangles = append(triangles, object.triangles...)
	return triangles
}

func PrecomputeScreenSpaceCoordinates(screenWidth, screenHeight int, FOV float32, camera Camera) [][]Vector {
	aspectRatio := float32(screenWidth) / float32(screenHeight)
	scale := math32.Tan(FOV * 0.5 * math32.Pi / 180.0) // Convert FOV to radians

	screenSpaceCoordinates := make([][]Vector, screenWidth)
	for width := 0; width < screenWidth; width++ {
		screenSpaceCoordinates[width] = make([]Vector, screenHeight)
		for height := 0; height < screenHeight; height++ {
			// Normalize screen coordinates to [-1, 1]
			pixelNDCX := (float32(width) + 0.5) / float32(screenWidth)
			pixelNDCY := (float32(height) + 0.5) / float32(screenHeight)

			// Screen space coordinates [-1, 1]
			pixelScreenX := 2.0*pixelNDCX - 1.0
			pixelScreenY := 1.0 - 2.0*pixelNDCY

			// Apply aspect ratio and FOV scale
			pixelCameraX := pixelScreenX * aspectRatio * scale
			pixelCameraY := pixelScreenY * scale

			screenSpaceCoordinates[width][height] = Vector{pixelCameraX, pixelCameraY, -1}.Normalize()

			// Rotate the screen space coordinates based on the direction of the camera
			screenSpaceCoordinates[width][height] = screenSpaceCoordinates[width][height].Rotate(camera.xAxis, camera.yAxis, camera.zAxis)
		}
	}

	return screenSpaceCoordinates
}

func RotationMatrix(angleX, angleY, angleZ float32) [3][3]float32 {
	cx := math32.Cos(angleX)
	sx := math32.Sin(angleX)
	cy := math32.Cos(angleY)
	sy := math32.Sin(angleY)
	cz := math32.Cos(angleZ)
	sz := math32.Sin(angleZ)

	return [3][3]float32{
		{cy * cz, cy * sz, -sy},
		{sx*sy*cz - cx*sz, sx*sy*sz + cx*cz, sx * cy},
		{cx*sy*cz + sx*sz, cx*sy*sz - sx*cz, cx * cy},
	}
}

func UpdateScreenSpaceCoordinates(screenSpaceCoordinates [][]Vector, camera Camera) [][]Vector {
	rotationMatrix := RotationMatrix(camera.xAxis, camera.yAxis, camera.zAxis)

	for width := range screenSpaceCoordinates {
		for height := range screenSpaceCoordinates[width] {
			v := screenSpaceCoordinates[width][height]

			// Apply the rotation matrix
			screenSpaceCoordinates[width][height] = Vector{
				x: rotationMatrix[0][0]*v.x + rotationMatrix[0][1]*v.y + rotationMatrix[0][2]*v.z,
				y: rotationMatrix[1][0]*v.x + rotationMatrix[1][1]*v.y + rotationMatrix[1][2]*v.z,
				z: rotationMatrix[2][0]*v.x + rotationMatrix[2][1]*v.y + rotationMatrix[2][2]*v.z,
			}
		}
	}
	return screenSpaceCoordinates
}



func DrawRays(camera Camera, light Light, scaling int, samples int, screenSpaceCoordinates [][]Vector, depth int, subImages []*ebiten.Image) {
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
					rayDirection := screenSpaceCoordinates[x][y]
					ray := Ray{origin: camera.Position, direction: rayDirection}
					c := TraceRay(ray, depth, light, samples)

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
	// Wait for all workers to finish
	wg.Wait()
}

// // UpdateImage writes pixels from pixelChan to the screen image efficiently
// func UpdateImage(screen *ebiten.Image, pixelChan <-chan Pixel) {
// 	width, height := screen.Bounds().Dx(), screen.Bounds().Dy()
// 	pixelBuffer := make([]uint8, width*height*4)

// 	for pixel := range pixelChan {
// 		index := (pixel.y*width + pixel.x) * 4
// 		pixelBuffer[index] = pixel.color.R
// 		pixelBuffer[index+1] = pixel.color.G
// 		pixelBuffer[index+2] = pixel.color.B
// 		pixelBuffer[index+3] = pixel.color.A // Alpha channel
// 	}
// 	screen.WritePixels(pixelBuffer)
// }

func findIntersectionAndSetColor(node *BVHNode, ray Ray, newColor color.RGBA) bool {
	if node == nil {
		return false
	}

	// Check if ray intersects the bounding box of the node
	if !BoundingBoxCollision(node.BoundingBox, &ray) {
		return false
	}

	// If this is a leaf node, check the triangles for intersection
	if node.Triangles != nil {
		for i, triangle := range *node.Triangles {
			if _, hit := ray.IntersectTriangle(triangle); hit {
				// fmt.Println("Triangle hit", triangle.color)
				triangle.color = newColor
				// Update the triangle in the slice
				(*node.Triangles)[i] = triangle // Dereference the pointer to modify the slice
				return true
			}
		}
		return false
	}

	// Traverse the left and right child nodes
	leftHit := findIntersectionAndSetColor(node.Left, ray, newColor)
	rightHit := findIntersectionAndSetColor(node.Right, ray, newColor)

	return leftHit || rightHit
}

func (g *Game) Update() error {
	// check if the mouse is pressed
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		// get position of the mouse
		x, y := ebiten.CursorPosition()
		// fmt.Println("Mouse position", x, y)
		if x < screenWidth && y < screenHeight {
			findIntersectionAndSetColor(g.BVHobjects, Ray{origin: g.camera.Position, direction: g.screenSpaceCoordinates[x][y]}, color.RGBA{255, 0, 0, 255})
		}
	}

	// Rotate the camera around the object
	angle := float64(g.updateFreq) * 2 * math.Pi / 600
	g.camera.Position.x = float32(math.Cos(angle)) * 300
	g.camera.Position.z = float32(math.Sin(angle)) * 300

	// g.camera.yAxis += 0.01
	// g.camera.xAxis += 0.01
	// g.camera.zAxis += 0.01

	// Update the screen space coordinates
	// start := time.Now()
	// g.screenSpaceCoordinates = PrecomputeScreenSpaceCoordinates(screenWidth, screenHeight, FOV, g.camera)
	// fmt.Println("PrecomputeScreenSpaceCoordinates:", time.Since(start))
	g.screenSpaceCoordinates = UpdateScreenSpaceCoordinates(g.screenSpaceCoordinates, g.camera)

	g.updateFreq++

	// Check if 30 seconds have passed
	if time.Since(g.startTime).Seconds() >= 30 {
		fmt.Println("Average FPS:", averageFPS/float64(Frames))
		// Close the program
		os.Exit(0)
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

var averageFPS float64
var Frames int

func (g *Game) Draw(screen *ebiten.Image) {
	// Display frame rate
	fps := ebiten.ActualFPS()

	averageFPS += fps
	Frames++

	g.currentFrame.Clear()

	// Perform path tracing and draw rays into the current frame
	DrawRays(g.camera, g.light, g.scaleFactor, g.samples, g.screenSpaceCoordinates, g.depth, g.subImages)
	for i, subImage := range g.subImages {
		op := &ebiten.DrawImageOptions{}
		op.GeoM.Translate(0, float64(subImageHeight)*float64(i)) // Use the outer loop variable directly
		g.currentFrame.DrawImage(subImage, op)
	}

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(float64(screen.Bounds().Dx())/float64(g.currentFrame.Bounds().Dx()), float64(screen.Bounds().Dy())/float64(g.currentFrame.Bounds().Dy()))
	screen.DrawImage(g.currentFrame, op)

	// Create a temporary image for bloom shader
	// bloomImage := ebiten.NewImageFromImage(g.currentFrame)

	// Apply Bloom shader
	// bloomOpts := &ebiten.DrawRectShaderOptions{}
	// bloomOpts.Images[0] = g.currentFrame
	// bloomOpts.Uniforms = map[string]interface{}{
	// 	"screenSize":     []float32{float32(bloomImage.Bounds().Dx()), float32(bloomImage.Bounds().Dy())},
	// 	"bloomThreshold": 1,
	// }

	// // Apply the bloom shader
	// bloomImage.DrawRectShader(
	// 	bloomImage.Bounds().Dx(),
	// 	bloomImage.Bounds().Dy(),
	// 	g.bloomShader,
	// 	bloomOpts,
	// )

	// Apply Dither shader
	// ditherImage := ebiten.NewImageFromImage(bloomImage)
	// ditherOpts := &ebiten.DrawRectShaderOptions{}
	// ditherOpts.Images[0] = g.currentFrame
	// ditherOpts.Uniforms = map[string]interface{}{
	// 	"screenSize":  []float32{float32(ditherImage.Bounds().Dx()), float32(ditherImage.Bounds().Dy())},
	// 	"BayerMatrix": bayerMatrix,
	// }

	// // Apply the dither shader
	// ditherImage.DrawRectShader(
	// 	ditherImage.Bounds().Dx(),
	// 	ditherImage.Bounds().Dy(),
	// 	g.ditherColor,
	// 	ditherOpts,
	// )

	// Draw the current frame (bloomImage) to the screen, scaling it to screen size
	// Draw bloomImage to the screen, scaling it to screen size
	// Draw bloomImage to the screen, scaling it to screen size
	// op1 := &ebiten.DrawImageOptions{}
	// op1.GeoM.Scale(float64(screen.Bounds().Dx())/float64(bloomImage.Bounds().Dx()),
	// 	float64(screen.Bounds().Dy())/float64(bloomImage.Bounds().Dy()))
	// screen.DrawImage(bloomImage, op1)

	// Create Triangle Rendering Shader
	// triangleImage := ebiten.NewImageFromImage(ditherImage)
	// triangleOpts := &ebiten.DrawRectShaderOptions{}
	// triangleOpts.Images[0] = ditherImage
	// triangleOpts.Uniforms = map[string]interface{}{
	// 	"cameraPos": []float32{g.camera.Position.x, g.camera.Position.y, g.camera.Position.z},
	// 	"cameraDir": []float32{g.camera.xAxis, g.camera.yAxis, g.camera.zAxis},
	// 	"cameraPitch" : g.camera.xAxis,
	// 	"cameraYaw" : g.camera.yAxis,
	// 	"cameraRoll" : g.camera.zAxis,
	// 	"cameraFoe" : FOV,

	// 	"TriangleV1": TrianglesV1,
	// 	"TriangleV2": TrianglesV2,
	// 	"TriangleV3": TrianglesV3,

	// 	"epsilon": 0.0001,
	// 	"maxDist ": 1000.0,
	// }

	// // Apply the triangle shader
	// triangleImage.DrawRectShader(
	// 	triangleImage.Bounds().Dx(),
	// 	triangleImage.Bounds().Dy(),
	// 	g.TriangleShader,
	// 	triangleOpts,
	// )

	// screen.DrawImage(triangleImage, op1)
	// // Prepare the options for ditherImage with darker blending
	// op2 := &ebiten.DrawImageOptions{}
	// op2.GeoM.Scale(float64(screen.Bounds().Dx())/float64(ditherImage.Bounds().Dx()),
	// 	float64(screen.Bounds().Dy())/float64(ditherImage.Bounds().Dy()))
	// op2.CompositeMode = ebiten.CompositeModeMultiply // Set to multiply for darker blending

	// // Draw ditherImage with darker blending
	// screen.DrawImage(ditherImage, op2)

	// Show the current FPS
	ebitenutil.DebugPrint(screen, fmt.Sprintf("FPS: %.2f", fps))
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 800, 608
}

var BVH *BVHNode

type Game struct {
	subImages              []*ebiten.Image
	camera                 Camera
	light                  Light
	scaleFactor            int
	samples                int
	screenSpaceCoordinates [][]Vector
	BVHobjects             *BVHNode
	startTime              time.Time
	updateFreq             int
	currentFrame           *ebiten.Image
	depth                  int
	ditherColor            *ebiten.Shader
	ditherGrayScale        *ebiten.Shader
	bloomShader            *ebiten.Shader
	// TriangleShader         *ebiten.Shader
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

var subImageHeight int
var subImageWidth int

func main() {
	src, err := LoadShader("shaders/ditherColor.kage")
	if err != nil {
		panic(err)
	}
	ditherShaderColor, err := ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	src, err = LoadShader("shaders/ditherGray.kage")
	if err != nil {
		panic(err)
	}
	ditherGrayShader, err := ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

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
	bloomShader, err := ebiten.NewShader(src)
	if err != nil {
		panic(err)
	}

	fmt.Println("Shader:", bloomShader)
	// fmt.Println("Shader:", ditherGrayShader)

	fmt.Println("Number of CPUs:", numCPU)

	runtime.GOMAXPROCS(numCPU)

	ebiten.SetVsyncEnabled(false)
	ebiten.SetTPS(24)

	// spheres := GenerateRandomSpheres(15)
	// cubes := GenerateRandomCubes(30)

	obj, err := LoadOBJ("T 90.obj")
	if err != nil {
		panic(err)
	}
	obj.Scale(65)

	objects := []object{}
	objects = append(objects, obj)

	camera := Camera{Position: Vector{0, 100, 0}, xAxis: 0, yAxis: 0, zAxis: 0}
	light := Light{Position: &Vector{0, 1500, 1000}, Color: &[3]float32{1, 1, 1}, intensity: 1}

	// bestDepth := OptimizeBVHDepth(objects, camera, light)

	// objects = append(objects, spheres...)
	// objects = append(objects, cubes...)

	bvh := ConvertObjectsToBVH(objects, maxDepth)

	BVH = bvh

	// Optimize the block size
	// minBlockSize := 16
	// maxBlockSize := 512
	// maxIteration := 8
	// bestBlockSize := OptimizeBlockSize(objects, camera, light, bvh, minBlockSize, maxBlockSize, maxIteration)
	// BlockSize: 181, FPS: 10.30 | BlockSize: 347, FPS: 9.30
	// BlockSize: 126, FPS: 21.50 | BlockSize: 237, FPS: 9.30
	// BlockSize: 89, FPS: 18.70 | BlockSize: 164, FPS: 11.50
	// BlockSize: 65, FPS: 19.30 | BlockSize: 115, FPS: 20.80
	// BlockSize: 98, FPS: 16.30 | BlockSize: 131, FPS: 20.80
	// BlockSize: 120, FPS: 15.80 | BlockSize: 142, FPS: 15.60
	// BlockSize: 112, FPS: 21.20 | BlockSize: 128, FPS: 19.20
	// BlockSize: 108, FPS: 19.10 | BlockSize: 118, FPS: 22.20
	// BlockSize: 114, FPS: 20.70 | BlockSize: 122, FPS: 20.70
	// BlockSize: 118, FPS: 16.80 | BlockSize: 124, FPS: 24.90

	scale := 2
	screenSpaceCoordinates := PrecomputeScreenSpaceCoordinates(screenWidth, screenHeight, FOV, camera)

	subImages := make([]*ebiten.Image, numCPU)

	subImageHeight = screenHeight / numCPU / scale
	subImageWidth = screenWidth / scale

	for i := range numCPU {
		subImages[i] = ebiten.NewImage(int(subImageWidth), int(subImageHeight))
	}

	game := &Game{
		subImages:              subImages,
		camera:                 camera,
		light:                  light,
		scaleFactor:            scale,
		updateFreq:             0,
		samples:                0,
		startTime:              time.Now(),
		screenSpaceCoordinates: screenSpaceCoordinates,
		BVHobjects:             bvh,
		depth:                  2,
		ditherColor:            ditherShaderColor,
		ditherGrayScale:        ditherGrayShader,
		bloomShader:            bloomShader,
		currentFrame:           ebiten.NewImage(screenWidth/scale, screenHeight/scale),
		// TriangleShader: 	   rayCasterShader,
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Ebiten Benchmark")

	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}

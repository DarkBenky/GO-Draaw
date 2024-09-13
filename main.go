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

package main

import (
	"bufio"
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/chewxy/math32"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const screenWidth = 800
const screenHeight = 600
const FOV = 90
const maxDepth = 12

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
						reflection: 0.25,
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

	// Final intersection check
	return tmax >= max(0.0, tmin)
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

func CreateSphere(center Vector, radius float32, color color.RGBA, reflection float32) []Triangle {
	var triangles []Triangle
	latitudeBands := 15
	longitudeBands := 15

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
	Position  Vector
	Color     color.RGBA
	intensity float32
}

func (light *Light) CalculateLighting(intersection Intersection, bvh *BVHNode) color.RGBA {
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	shadowRay := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: lightDir}

	// Check if the point is in shadow
	inShadow := false
	if _, intersect := shadowRay.IntersectBVH(bvh); intersect {
		inShadow = true
	}

	// Ambient light contribution
	ambientFactor := 0.05 // Adjust ambient factor as needed
	ambientColor := color.RGBA{
		uint8(float64(light.Color.R) * ambientFactor),
		uint8(float64(light.Color.G) * ambientFactor),
		uint8(float64(light.Color.B) * ambientFactor),
		light.Color.A,
	}

	if inShadow {
		// If in shadow, return ambient color
		return ambientColor
	}

	// Calculate diffuse lighting

	lightIntensity := light.intensity * math32.Max(0.0, lightDir.Dot(intersection.Normal))
	finalColor := color.RGBA{
		clampUint8(float32(ambientColor.R) + lightIntensity*float32(intersection.Color.R)),
		clampUint8(float32(ambientColor.G) + lightIntensity*float32(intersection.Color.G)),
		clampUint8(float32(ambientColor.B) + lightIntensity*float32(intersection.Color.B)),
		ambientColor.A,
	}

	return finalColor
}

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

func (ray *Ray) IntersectBVH(nodeBVH *BVHNode) (Intersection, bool) {
	// If the ray doesn't hit the bounding box, return immediately
	if !BoundingBoxCollision(nodeBVH.BoundingBox, ray) {
		return Intersection{}, false
	}

	// If the node is a leaf, check intersections with all triangles
	if len(nodeBVH.Triangles) > 0 {
		closestIntersection := Intersection{Distance: math32.MaxFloat32}
		hasIntersection := false

		for _, triangle := range nodeBVH.Triangles {
			tempIntersection, intersect := ray.IntersectTriangle(triangle)
			if intersect && tempIntersection.Distance < closestIntersection.Distance {
				closestIntersection = tempIntersection
				hasIntersection = true
			}
		}

		return closestIntersection, hasIntersection
	}

	// Recursively check child nodes
	leftHit := nodeBVH.Left != nil && BoundingBoxCollision(nodeBVH.Left.BoundingBox, ray)
	rightHit := nodeBVH.Right != nil && BoundingBoxCollision(nodeBVH.Right.BoundingBox, ray)

	if leftHit && rightHit {
		// Traverse both children and return the closest intersection
		leftIntersection, leftIntersect := ray.IntersectBVH(nodeBVH.Left)
		rightIntersection, rightIntersect := ray.IntersectBVH(nodeBVH.Right)

		if leftIntersect && rightIntersect {
			if leftIntersection.Distance < rightIntersection.Distance {
				return leftIntersection, true
			}
			return rightIntersection, true
		} else if leftIntersect {
			return leftIntersection, true
		} else if rightIntersect {
			return rightIntersection, true
		}
	} else if leftHit {
		return ray.IntersectBVH(nodeBVH.Left)
	} else if rightHit {
		return ray.IntersectBVH(nodeBVH.Right)
	}

	return Intersection{}, false
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

type Camera struct {
	Position            Vector
	xAxis, yAxis, zAxis float32
}

type Pixel struct {
	x, y  int
	color color.RGBA
}

func (intersection *Intersection) Scatter(samples int, light Light, bvh *BVHNode) color.RGBA {
	var finalColor color.RGBA64

	// Calculate the direct reflection
	lightDir := light.Position.Sub(intersection.PointOfIntersection).Normalize()
	reflectDir := intersection.Normal.Mul(2 * lightDir.Dot(intersection.Normal)).Sub(lightDir).Normalize()
	reflectRay := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: reflectDir}

	// Find intersection with the scene
	reflectedIntersection := Intersection{}
	tempIntersection, intersect := reflectRay.IntersectBVH(bvh)
	if intersect {
		reflectedIntersection = tempIntersection
	}

	for i := 0; i < samples; i++ {
		// Generate random direction in hemisphere around the normal (cosine-weighted distribution)
		u := rand.Float32()
		v := rand.Float32()
		r := math32.Sqrt(u)
		theta := 2 * math32.Pi * v

		// Construct local coordinate system based on the precomputed normal
		w := intersection.Normal // Use the precomputed normal vector from the intersection
		var uVec, vVec Vector
		if math32.Abs(w.x) > 0.1 {
			uVec = Vector{0.0, 1.0, 0.0}
		} else {
			uVec = Vector{1.0, 0.0, 0.0}
		}
		uVec = uVec.Cross(w).Normalize()
		vVec = w.Cross(uVec)

		// Calculate direction in local coordinates
		directionLocal := uVec.Mul(float32(r * math32.Cos(theta))).Add(vVec.Mul(float32(r * math32.Sin(theta)))).Add(w.Mul(float32(math32.Sqrt(1 - u))))

		// Transform direction to global coordinates
		direction := directionLocal.Normalize()

		// Create ray starting from intersection point
		ray := Ray{origin: intersection.PointOfIntersection.Add(intersection.Normal.Mul(0.001)), direction: direction}

		// Find intersection with the scene
		scatteredIntersection := Intersection{Distance: math32.MaxFloat32}
		bvhIntersection, intersect := ray.IntersectBVH(bvh)
		if intersect {
			scatteredIntersection = bvhIntersection
		}

		if scatteredIntersection.Distance != math32.MaxFloat32 {
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

	// Mix the direct reflection with the scattered color
	rationScatterToDirect := 1 - intersection.reflection

	return color.RGBA{
		clampUint8(float32(finalColor.R)*rationScatterToDirect + float32(reflectedIntersection.Color.R)*intersection.reflection),
		clampUint8(float32(finalColor.G)*rationScatterToDirect + float32(reflectedIntersection.Color.G)*intersection.reflection),
		clampUint8(float32(finalColor.B)*rationScatterToDirect + float32(reflectedIntersection.Color.B)*intersection.reflection),
		uint8(finalColor.A),
	}
}

type object struct {
	triangles   []Triangle
	BoundingBox [2]Vector
	materials   map[string]Material
}

func ConvertObjectsToBVH(objects []object, maxDepth int) *BVHNode {
	triangles := []Triangle{}
	for _, object := range objects {
		triangles = append(triangles, object.triangles...)
	}
	return buildBVHNode(triangles, 0, maxDepth)
}

type BVHNode struct {
	Left, Right *BVHNode
	BoundingBox *[2]Vector
	Triangles   []Triangle
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
		node := &BVHNode{BoundingBox: &boundingBox}
		node.Triangles = triangles
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

func DrawRays(bvh *BVHNode, screen *ebiten.Image, camera Camera, light Light, scaling int, samples int, screenSpaceCoordinates [][]Vector, blockSize int) {
	pixelChan := make(chan Pixel, screenWidth*screenHeight)
	var wg sync.WaitGroup

	for startX := 0; startX < screenWidth; startX += blockSize * scaling {
		for startY := 0; startY < screenHeight; startY += blockSize * scaling {
			wg.Add(1)
			go func(startX, startY int) {
				defer wg.Done()

				endX := min(startX+blockSize*scaling, screenWidth)
				endY := min(startY+blockSize*scaling, screenHeight)

				for width := startX; width < endX; width += scaling {
					for height := startY; height < endY; height += scaling {
						// Use precomputed screen space coordinates
						rayDirection := screenSpaceCoordinates[width][height]
						ray := Ray{origin: camera.Position, direction: rayDirection}

						intersection := Intersection{Distance: math32.MaxFloat32}

						// Find intersection with scene
						intersection, intersect := ray.IntersectBVH(bvh)

						if intersection.Distance != math32.MaxFloat32 && intersect {
							// Calculate the final color with lighting
							Scatter := intersection.Scatter(samples, light, bvh)
							light := light.CalculateLighting(intersection, bvh)

							c := color.RGBA{
								G: clampUint8(float32(Scatter.G) + float32(light.G)),
								R: clampUint8(float32(Scatter.R) + float32(light.R)),
								B: clampUint8(float32(Scatter.B) + float32(light.B)),
								A: 255,
							}

							pixelChan <- Pixel{x: width, y: height, color: c}
						}
					}
				}
			}(startX, startY)
		}
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

type ColorInt16 struct {
	R uint16
	G uint16
	B uint16
	A uint16
}

func (g *Game) Update() error {

	// Rotate the camera around the object
	angle := float64(g.updateFreq) * 2 * math.Pi / 600
	g.camera.Position.x = float32(math.Cos(angle)) * 300
	g.camera.Position.y = 200
	g.camera.Position.z = float32(math.Sin(angle)) * 300

	g.camera.yAxis += 0.01

	// Update the screen space coordinates
	g.screenSpaceCoordinates = PrecomputeScreenSpaceCoordinates(screenWidth, screenHeight, FOV, g.camera)

	g.updateFreq++

	// Check if 60 seconds have passed
	if time.Since(g.startTime).Seconds() >= 30 {
		fmt.Println("Average FPS:", averageFPS/float64(Frames))
		// Close the program
		os.Exit(0)
	}

	return nil
}

// Mutex to synchronize access to shared resources
func (g *Game) InterpolateFrames(numInterpolations int) *ebiten.Image {
	if g.prevFrame == nil || g.currentFrame == nil {
		return g.currentFrame
	}

	// Calculate interpolation factor based on the number of frames to interpolate
	t := math.Min(1.0, ebiten.ActualFPS()/float64(numInterpolations*60))

	// Create a new image to hold the interpolated frame
	interpolatedFrame := ebiten.NewImageFromImage(g.prevFrame)

	// Blend between the previous and current frames
	op := &ebiten.DrawImageOptions{}
	op.ColorM.Scale(1.0-t, 1.0-t, 1.0-t, 1.0)
	interpolatedFrame.DrawImage(g.currentFrame, op)

	return interpolatedFrame
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Display frame rate
	fps := ebiten.ActualFPS()
	ebitenutil.DebugPrint(screen, fmt.Sprintf("FPS: %v", fps))

	// Move current frame to previous frame
	if g.currentFrame != nil {
		g.prevFrame = g.currentFrame
	}

	// Create a new image for the current frame
	g.currentFrame = ebiten.NewImage(800, 600)

	// Perform path tracing and draw rays into the current frame
	DrawRays(g.BVHobjects, g.currentFrame, g.camera, g.light, int(g.scaleFactor), g.samples, g.screenSpaceCoordinates, g.blockSize)
	averageFPS += fps
	Frames++

	// If there's no previous frame, just draw the current frame
	if g.prevFrame == nil {
		screen.DrawImage(g.currentFrame, nil)
		return
	}

	interpolatedFrame := g.InterpolateFrames(10) // Specify the number of frames to interpolate
	screen.DrawImage(interpolatedFrame, nil)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 800, 600
}

type Game struct {
	camera                 Camera
	light                  Light
	scaleFactor            float32
	samples                int
	screenSpaceCoordinates [][]Vector
	BVHobjects             *BVHNode
	startTime              time.Time
	updateFreq             int
	blockSize              int
	prevFrame              *ebiten.Image
	currentFrame           *ebiten.Image
}

var averageFPS = 0.0
var Frames = 0

func main() {
	numCPU := runtime.NumCPU()
	fmt.Println("Number of CPUs:", numCPU)

	runtime.GOMAXPROCS(numCPU)

	ebiten.SetVsyncEnabled(false)
	ebiten.SetTPS(24)

	// spheres := GenerateRandomSpheres(15)
	// cubes := GenerateRandomCubes(15)

	obj, err := LoadOBJ("Room.obj")
	if err != nil {
		panic(err)
	}
	obj.Scale(100)

	objects := []object{}
	objects = append(objects, obj)

	bvh := ConvertObjectsToBVH(objects, maxDepth)

	camera := Camera{Position: Vector{0, 200, 0}, xAxis: 0, yAxis: 0, zAxis: 0}

	game := &Game{
		camera:                 camera,
		light:                  Light{Position: Vector{0, 400, 10000}, Color: color.RGBA{255, 255, 255, 255}, intensity: 0.5},
		scaleFactor:            1,
		updateFreq:             0,
		samples:                2,
		startTime:              time.Now(),
		screenSpaceCoordinates: PrecomputeScreenSpaceCoordinates(screenWidth, screenHeight, FOV, camera),
		BVHobjects:             bvh,
		blockSize:              32,
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Ebiten Benchmark")

	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}

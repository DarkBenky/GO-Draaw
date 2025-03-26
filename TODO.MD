# TODO List

- [X] APIs Updated
- [ ] Add part about this benchmark
  - Results: BoolArray check time: 18.178246ms
    BitArray check time: 17.968679ms
    Direct check time: 17.94769ms

  ```go
    // BoolArray stores each bit as a separate bool
  type BoolArray struct {
    data []bool
    size int
  }

  // NewBoolArray initializes a bool array of given size
  func NewBoolArray(size int) *BoolArray {
    return &BoolArray{
      data: make([]bool, size),
      size: size,
    }
  }

  // IsSet checks if the n-th bit is 1 in BoolArray
  func (b *BoolArray) IsSet(n int) bool {
    if n >= b.size || n < 0 {
      return false
    }
    return b.data[n]
  }

  // BitArray stores bits efficiently using uint64
  type BitArray struct {
    data []uint64
    size int
  }

  // NewBitArray initializes a bit array of given size
  func NewBitArray(size int) *BitArray {
    return &BitArray{
      data: make([]uint64, (size+63)/64),
      size: size,
    }
  }

  // IsSet checks if the n-th bit is 1 in BitArray
  func (b *BitArray) IsSet(n int) bool {
    if n >= b.size || n < 0 {
      return false
    }
    return (b.data[n/64] & (1 << (n % 64))) != 0
  }

  // Benchmark function to compare BoolArray vs BitArray
  func BenchmarkCheckSpeed() {
    const size = 32*32*32
    const numChecks = 128*128*128

    // Initialize BoolArray and set some random bits
    boolArr := NewBoolArray(size)
    for i := 0; i < size/10; i++ {
      boolArr.data[rand.Intn(size)] = true
    }

    // initialize random blocks
    blocks := make([]Block, size)
    // set randomly block to true (LightColor.A > 25)
    for i := 0; i < size/10; i++ {
      blocks[rand.Intn(size)] = Block{LightColor: ColorFloat32{A: 26}}
    }

    // Initialize BitArray and set some random bits
    bitArr := NewBitArray(size)
    for i := 0; i < size/10; i++ {
      pos := rand.Intn(size)
      bitArr.data[pos/64] |= (1 << (pos % 64))
    }

    // Benchmark BoolArray
    start := time.Now()
    for i := 0; i < numChecks; i++ {
      _ = boolArr.IsSet(rand.Intn(size))
    }
    boolTime := time.Since(start)

    // Benchmark BitArray
    start = time.Now()
    for i := 0; i < numChecks; i++ {
      _ = bitArr.IsSet(rand.Intn(size))
    }
    bitTime := time.Since(start)

    // Benchmark direct access
    start = time.Now()
    for i := 0; i < numChecks; i++ {
      _ = blocks[rand.Intn(size)].LightColor.A > 25
    }
    directTime := time.Since(start)

    // Print results
    fmt.Println("BoolArray check time:", boolTime)
    fmt.Println("BitArray check time:", bitTime)
    fmt.Println("Direct check time:", directTime)
  }
  ```

- [ ] Add option to bake lighting
- [ ] Update documentation about new version of V4-Linear-Optim-V2 that check bounding box intersection for both boxes usually the node has 2 nodes so we can optimize with one func call not two
  Whole function
  - IntersectBVHLean_TextureLeanOptim: 484.632972ms
  - IntersectBVHLean_TextureLean: 532.122386ms
  - IntersectBVHLean-V1: 587.8628ms
  BBox Check
  - BoundingBoxCollisionVector: 291.248548ms
  - BoundingBoxCollisionPair: 215.934921ms
- [X] Test Bach processing of BBOX and Triangles
- [X] Add material properties type like specular etc.

  ```golang
  type Texture struct {
    texture [128][128]ColorFloat32
    normals [128][128]Vector
    reflection      float32
    directToScatter float32
    specular        float32
    Roughness       float32
    Metallic        float32
  }

  ```

  - BVH array concept

    ```golang
    type TriangleBBOX struct {
      V1orBBoxMin , V2orBBoxMax , V3  // 12 * 3
      normal Vector // 12
      id int16 // 2 id for texture
    }
    ```

    - if id == -1 => BBox
    - else => triangle

    ```golang
      var i uint16
      for i = 0; i < 52; i++ {
        fmt.Println(i)
    }
    ```

    [n]TriangleBBOX{}
    - [ ] Fix Render Version V4
- [ ] New BVH for Textures using TriangleBBOX

- [ ] Remove GUI
- [X] Vec 32 is pretty much same as float64 ([link](https://github.com/DarkBenky/Go-Vec32-Vec64))
- [ ] Update submitTexture tu submit specular
- [ ] Test BVH.SetPropertiesWithID
- [ ] Add Multiplayer for Textures In Frontend
- [X] Fix Benchmarking
- [ ] GOAMD64
  - v1: Base AMD64 features (default).
  - v2: Adds support for SSE3, SSSE3, and SSE4.1 instructions.
  - v3: Includes all v2 features plus SSE4.2, AVX, and AVX2.
  - v4: Enables all v3 features along with AVX512.
  - GOAMD64=v3 go build main.go
  - [ ] float 32 Vs float 64
- [X] add support for normal maps
  - [X] check if it does not have api ([link](https://cpetry.github.io/NormalMap-Online/))
- [ ] Why V2 is faster than V2-LOG...
  - [ ] Profiling
- [X] Color Mapping Shader
- [X] IntersectTriangleTexture fix add textures
- [ ] Movement options
- [X] Fix Draw Voxels Method\
- [ ] BVH densify
  - [ ] bvh Array method
- [ ] Optimize struct to be memory aligned
- [ ] Fix Distance Buffer
- [ ] Add Textures for triangles ([link](https://chatgpt.com/c/6799fedf-17ac-8004-a51b-2bfaacdf7b98))
  - [ ] Triangle Struct for Material
    - [ ] Methods
  - [ ] BVH
  - [ ] Draw Method
- [ ] Finish Voxel renderer
  - [X] Implement Render Method
  - [x] Implement Ray Between Sun and Point in Volume
  - [ ] Add Baked Lighting from Volume
  - [ ] Particle System
- [ ] Finish Backend + Frontend connection
  - [X] Color
  - [X] Volume
  - [ ] Voxels
  - [ ] Shaders
- [ ] Ability to save BVH / Voxel Volume
- [ ] Fix Render frame
- [ ] Crete benchmark to test optimal BVH structure for example (all in one array or only bbox and pointer or int to array of triangles)
- [X] Create new GUI using Kage shader
  - [X] or use this shader ([link](https://github.com/AllenDang/giu))
    - It is not compatable with enbiten
- [X] HW info for benchmark
- [x] Bloom based on gaussian blur
- [ ] Depth of field effect (shader)
- [ ] Rain Snow Shader
- [ ] Shaders implementation of normal to rendered image ([Link](https://chatgpt.com/c/675abf8e-582c-8004-816f-6544a1c45231))
- [ ] Hot reloading of shaders
- [X] ***Important*** Use standard sqrt instead of math32 sqrt
  - *Sources*
    - ([link](https://github.com/DarkBenky/Vec3-vs-Vec4-bench))
    - ([link](https://github.com/DarkBenky/FastSqrt-Test))
  - [ ] More test like speed of math32 sin cos tan vs math lib
- [ ] Shader Menu
  - [ ] BlendModes ([link](https://www.sketch.com/blog/blend-modes/))
  - [ ] Fix dither
- [ ] Depth Map
- [X] Normal Map
- [X] HDR for normal renderer (Linear, Logarithmic)
- [X] Fix the scaling of the Block based renderer
- [ ] Enable to down scale number of triangles in mesh
- [ ] *Important* RayTrace Enhance ([link](https://claude.ai/chat/d6805f99-cbcd-460d-89e4-c99f2b7493dc))
- [X] *Important* Modes of rendering ( old renderer , physically accurate renderer , HDR renderer)
- [X] *Optimization* (easy) sub image is size of full screen and each go routine will have random chunks of image
  - [X] test this optimization
    - Old : Current run - Average FPS: 189.80, Min FPS: 0.05, Max FPS: 234.29, Min 15% FPS: 10.60
    - New : Current run - Average FPS: 201.73, Min FPS: 0.06, Max FPS: 235.60, Min 15% FPS: 53.32
- [ ] ***Important*** fix camera projection
  - [ ] Screen position calculate vector for center cross to ring and left and cross to top and bottom and you will get the position of screen
- [ ] Draw UI in different process (use sub images for GUI to enable go routines)
  - [ ] ***Performance*** Implement Gui to be separate Go routine
- [ ] *Important* fix Saving and rendering image
- [ ] ***improvements*** Use pointers not globals ([link](https://github.com/DarkBenky/globals-vs-poiters))
- [ ] ***Important*** fix lighting intensity and color is not changing anything
- [X] ***Performance*** Implement Go routine for RayMarching
  - [X] ***Performance*** Use BVH for speed up
- [ ] ***Important*** Implement Voxel based rendering for smoke fog etc. ([link](https://www.youtube.com/watch?v=ryB8hT5TMSg))
- [X] Try converting lighting to type int16 or int32 to be able have emissive material and remove the light sours
  - [X] Add option to color Picker To have colors bigger then 255 to be able to have emissive materials
- [ ] Fix Native Resolution
- [X] Do Benchmarks about changes
  - ***Note*** Currently i am happy with performance i didn't get any noticeable drop
- [ ] *Optional* Crete assets for GUI
- [ ] Try go ASM or SIMD for vector operations
- [ ] Convert BVH to array based on the Benchmarks ([link](https://github.com/DarkBenky/testBinaryTree))
- [X] Implement Ray marching
  - [ ] **Important** Implement shader based on ([link](https://www.shadertoy.com/view/Xds3zN))
    - [ ] Fix Shader
  - [ ] Implement Intersection Union ([link](https://youtu.be/Cp5WWtMoeKg?si=g1ZjOuq3KzzEET3s&t=128))
  - [ ] Implement Different shapes (Cube...)
  - [X] Optimize it
    - [x] Test splitting screen based on BVH and process only small amount of objects
- [X] Add shader for averaging images
- [X] Convert All Triangles to TriangleSimple
- [ ] Fix overlapping images ( this problem is mainly seen when alpha is not 100%)
- ***NOTE*** Spliting array for each go routine does not make sense besed on the Benchmark ([link](https://github.com/DarkBenky/Benchmark-GO-multiple-arrays))
- [X] Add option to move light to camera position
- [ ] Split Code to multiple files
- [X] *Optimizations* Not allocating new slices in function
  - *Note* Did not help

  ```go
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

  func DrawRaysPerformance(camera Camera, light Light, scaling int, samples int, depth int, subImages []*ebiten.Image, pixelBuffers []*[]uint8) {
    var wg sync.WaitGroup

    // Create a pool of worker goroutines, each handling a portion of the image
    for i := 0; i < numCPU; i++ {
      wg.Add(1)
      go func(startY int, endIndex int, subImage *ebiten.Image, pixelBuffer []uint8) {
        defer wg.Done()
        yRow := 0
        for y := startY; y < endIndex; y += scaling {
          xColumn := 0
          for x := 0; x < screenWidth; x += scaling {
            rayDir := ScreenSpaceCoordinates[x][y]
            c := TraceRay(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples)

            // Write the pixel color to the pixel buffer
            index := ((yRow*screenWidth + xColumn) * 4)
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
      }(i*rowSize, (i+1)*rowSize, subImages[i], *pixelBuffers[i])
    }
    if performanceOptions.Selected == 0 {
      // Wait for all workers to finish
      wg.Wait()
    }
  } 

  ```

  ```go
    func DrawRaysBlockV2(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImage) {
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

      if performanceOptions.Selected == 0 {
          wg.Wait()
      }
  }

  // Time taken for V2:  8.789734794642856e+06
  // Time taken for V2-Unsafe:  9.21041036607143e+06
  func DrawRaysBlockUnsafe(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImage) {
      var wg sync.WaitGroup
      ScreenSpaceCoordinatesPtr := unsafe.Pointer(&ScreenSpaceCoordinates[0][0])
      for _, block := range blocks {
          wg.Add(1)
          go func(block BlocksImage) {
              pixelBufferPtr := unsafe.Pointer(&block.pixelBuffer[0])
              defer wg.Done()
              for y := block.startY; y < block.endY; y += 1 {
                  if y*scaling >= screenHeight {
                      continue
                  }
                  for x := block.startX; x < block.endX; x += 1 {
                      if x*scaling >= screenWidth {
                          continue
                      }
                      rayDir := *(*Vector)(unsafe.Pointer(uintptr(ScreenSpaceCoordinatesPtr) + uintptr((x*scaling*screenHeight+y*scaling)*12)))
                      c := TraceRayV3(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples)

                      // Write the pixel color to the pixel buffer
                      index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
                      *(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index))) = clampUint8(c.R)
                      *(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index+1))) = clampUint8(c.G)
                      *(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index+2))) = clampUint8(c.B)
                      *(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index+3))) = clampUint8(c.A)
                  }
              }
              block.image.WritePixels(block.pixelBuffer)
          }(block)
      }

      if performanceOptions.Selected == 0 {
          wg.Wait()
      }
  }
  ```

- [X] Create Universal benchmarking function to judge the changes of performance
- [X] Remove Triangle Bounding Box out of struct

  ```go
  type Triangle struct {
      v1, v2, v3  Vector
      color       color.RGBA
      BoundingBox [2]Vector
      Normal      Vector
      reflection  float32
      specular    float32
  }
  
  # Simple Triangle
  type Triangle struct {
      v1, v2, v3  Vector
      color       color.RGBA
      Normal      Vector
      reflection  float32
      specular    float32
  }
  ```

  - Test The changes Crete function for benchmarking
  
- [X] Try Multiple BOX of Triangle intersection in on function using SIMD
  - Example ([link](https://claude.ai/chat/b6e783ca-1044-4c96-a412-845b8b56465f))
  - Test didn't show any improvements
    - Normal 80ns
      SIMD 992ns
      Normal 51ns
      SIMD 731ns
      Normal 50ns
      SIMD 652ns
      Normal 50ns

    ```go
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

    func BoundingBoxCollisionDistanceSIMD(BoundingBox [2]Vector, ray *Ray) (bool, float32) {
        // Precompute the inverse direction
        invDir := vek32.Inv([]float32{ray.direction.x, ray.direction.y, ray.direction.z})

        // Compute the tmin and tmax for each axis directly
        tx := vek32.Minimum(
            []float32{(BoundingBox[0].x - ray.origin.x) * invDir[0]},
            []float32{(BoundingBox[1].x - ray.origin.x) * invDir[0]},
        )
        ty := vek32.Minimum(
            []float32{(BoundingBox[0].y - ray.origin.y) * invDir[1]},
            []float32{(BoundingBox[1].y - ray.origin.y) * invDir[1]},
        )
        tz := vek32.Minimum(
            []float32{(BoundingBox[0].z - ray.origin.z) * invDir[2]},
            []float32{(BoundingBox[1].z - ray.origin.z) * invDir[2]},
        )

        tmin := vek32.Maximum(tx, vek32.Maximum(ty, tz))

        tx = vek32.Maximum(
            []float32{(BoundingBox[0].x - ray.origin.x) * invDir[0]},
            []float32{(BoundingBox[1].x - ray.origin.x) * invDir[0]},
        )
        ty = vek32.Maximum(
            []float32{(BoundingBox[0].y - ray.origin.y) * invDir[1]},
            []float32{(BoundingBox[1].y - ray.origin.y) * invDir[1]},
        )
        tz = vek32.Maximum(
            []float32{(BoundingBox[0].z - ray.origin.z) * invDir[2]},
            []float32{(BoundingBox[1].z - ray.origin.z) * invDir[2]},
        )

        tmax := vek32.Minimum(tx, vek32.Minimum(ty, tz))

        // Final intersection check
        tminPositive := vek32.Maximum([]float32{0.0}, tmin)
        intersects := vek32.Gte(tmax, tminPositive)
        if vek32.Any(intersects) {
            return true, tmin[0]
        }

        return false, 0.0
    }
    ```

- [X] Try BVH leaf notes with different number of triangles inside
  - **NOTE** Leaf Notes Contain only one Triangle For Easier Representation As Array
- [X] Implement GUI ([link](https://github.com/zeozeozeo/ebitengine-microui-go))
  - **NOTE**: Scraped for now. Currently working on my own custom GUI.
- [ ] Fix Shaders and Test Shaders
- [X] Implement SIMD instructions ([link](https://github.com/viterin/vek))
  - **Important** It does not improve performance base on the test

  ```go
    vec := []float32{1, 2, 3}
    add := []float32{3, 2, 1}

    start := time.Now()

    for i := 0; i < 1_000_000; i++ {
      vek32.Add_Inplace(vec, add)
    }

    fmt.Println("Result:", vec, "it took", time.Since(start))
    // Result: [3.000001e+06 2.000002e+06 1.000003e+06] it took 13.20764ms

    dot := float32(0.0)

    start = time.Now()

    for i := 0; i < 1_000_000; i++ {
      dot = vek32.Dot(vec, add)
    }

    fmt.Println("Result:", dot, "it took", time.Since(start))
    // Result: 1.400001e+07 it took 11.798035ms
    

    v := Vector{1, 2, 3}
    a := Vector{3, 2, 1}

    start = time.Now()

    for i := 0; i < 1_000_000; i++ {
      v = v.Add(a)
    }

    fmt.Println("Result:", v, "it took", time.Since(start))
    // Result: {3.000001e+06 2.000002e+06 1.000003e+06} it took 1.349328ms

    dot = 0.0

    start = time.Now()

    for i := 0; i < 1_000_000; i++ {
      dot = v.Dot(a)
    }
  ```

- [X] Merge the latest branch with main
- [X] Implement bloom and other post-processing effects
- [X] Implement shaders for color grading, contrast, etc.
- [X] Unsafe implementation of Drawing method
  - *Note* This implementation is not working

  ```golang
    func DrawRaysBlockV2(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImage) {
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

	if performanceOptions.Selected == 0 {
		wg.Wait()
	}
}

// Time taken for V2:  8.789734794642856e+06
// Time taken for V2-Unsafe:  9.21041036607143e+06
func DrawRaysBlockUnsafe(camera Camera, light Light, scaling int, samples int, depth int, blocks []BlocksImage) {
	var wg sync.WaitGroup
	ScreenSpaceCoordinatesPtr := unsafe.Pointer(&ScreenSpaceCoordinates[0][0])
	for _, block := range blocks {
		wg.Add(1)
		go func(block BlocksImage) {
			pixelBufferPtr := unsafe.Pointer(&block.pixelBuffer[0])
			defer wg.Done()
			for y := block.startY; y < block.endY; y += 1 {
				if y*scaling >= screenHeight {
					continue
				}
				for x := block.startX; x < block.endX; x += 1 {
					if x*scaling >= screenWidth {
						continue
					}
					// rayDir := ScreenSpaceCoordinates[x*scaling][y*scaling]
					rayDir := *(*Vector)(unsafe.Pointer(uintptr(ScreenSpaceCoordinatesPtr) + uintptr((x*scaling*screenHeight+y*scaling)*12)))
					c := TraceRayV3(Ray{origin: camera.Position, direction: rayDir}, depth, light, samples)

					// Write the pixel color to the pixel buffer
					index := ((y-block.startY)*(block.endX-block.startX) + (x - block.startX)) * 4
					*(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index))) = clampUint8(c.R)
					*(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index+1))) = clampUint8(c.G)
					*(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index+2))) = clampUint8(c.B)
					*(*uint8)(unsafe.Pointer(uintptr(pixelBufferPtr) + uintptr(index+3))) = clampUint8(c.A)

				}
			}
			block.image.WritePixels(block.pixelBuffer)
		}(block)
	}

	if performanceOptions.Selected == 0 {
		wg.Wait()
	}
}

  ```
- [ ] *Optional* Optimize for speed
- [ ] *Optional* Create flame graphs
- [ ] Trace rays to corners and draw triangles:
  - Create a box/cone representing the camera's viewing area
  - Find all triangles inside it
  - Order them based on distance
  - Draw them (consider adding shading based on normals)
- [ ] Fix ray tracer:
  - [X] Fix uniforms ([link](https://github.com/tinne26/kage-desk/blob/main/docs/tutorials/intro/06_uniforms.md))
  - **Note**: Uniforms start with capital letters.
- [X] Implement specular lighting
- [X] Add coloring of triangles
- [X] Fix camera movement (it works, but not perfectly)
  - [X] Add the previous camera projection - Done, but might need revisiting
  - [ChatGPT Link](https://chatgpt.com/c/6709b2d9-4770-8004-a46b-59e572c0b0cc)
- [X] Optimize `PrecomputeScreenSpaceCoordinatesSphere` (goroutines; if no movement, no change)
- [X] Implement Custom GUI
  - [X] Shader Menu
  - [X] Color Menu
    - [X] Add menu for Specular lighting etc.
  - [ ] Painter Menu
- [ ] Add painting utilities to the renderer
- [ ] Update Split os Screen and Num of Go routines based on number of CPUs
- [X] Fix pointer dereference issue after long runs
  - **Note** Test on more system maybe it is only problem with instability when OC
  - **Error**:

    ```plaintext
    panic: runtime error: invalid memory address or nil pointer dereference
    [signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x473b71]
    
    goroutine 91998 [running]:
    math/rand.(*runtimeSource).Int63(0x0?)
            /snap/go/10730/src/math/rand/rand.go:360 +0xf
    math/rand.(*Rand).Int63(...)
            /snap/go/10730/src/math/rand/rand.go:96
    math/rand.(*Rand).Float64(...)
            /snap/go/10730/src/math/rand/rand.go:207
    math/rand.(*Rand).Float32(0xc000628390)
            /snap/go/10730/src/math/rand/rand.go:220 +0x46
    math/rand.Float32()
            /snap/go/10730/src/math/rand/rand.go:461 +0x18
    main.TraceRay({{0x42678478, 0x439347d8, 0x418f8f2e}, {0xbed36efb, 0xbf460b12, 0xbc58c440}}, 0x2, {0xc00011d450?, 0xc00011d460?, 0xff4e4e4e?}, ...)
            /home/user/Desktop/GO-Draaw/main.go:826 +0x1ff
    main.DrawRays.func1(0x17c, 0x1a2, 0xc0005a6600)
            /home/user/Desktop/GO-Draaw/main.go:1277 +0x2ae
    created by main.DrawRays in goroutine 11
            /home/user/Desktop/GO-Draaw/main.go:1267 +0xb0
    exit status 2
    ```


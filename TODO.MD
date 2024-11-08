# TODO List

- [ ] Do Benchmarks about changes
- [ ] Crete assets for GUI
- [ ] Try go ASM or SIMD for vector operations
- [ ] Convert BVH to array based on the Benchmarks ([link](https://github.com/DarkBenky/testBinaryTree))
- [ ] Add shader for averaging images
- [ ] Convert All Triangles to TriangleSimple
- [ ] Fix overlapping images ( this problem is mainly seen when alpha is not 100%)
- ***NOTE*** Spliting array for each go routine does not make sense besed on the Benchmark ([link](https://github.com/DarkBenky/Benchmark-GO-multiple-arrays))
- [ ] Add option to move light to camera position
- [ ] Split Code to multiple files
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

- [ ] Try BVH leaf notes with different number of triangles inside
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
- [ ] Optimize for speed
- [ ] Create flame graphs
- [X] Trace rays to corners and draw triangles:
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
- [ ] Implement Custom GUI
  - [ ] Shader Menu
  - [X] Color Menu
    - [X] Add menu for Specular lighting etc.
  - [ ] Painter Menu
- [ ] Add painting utilities to the renderer
- [ ] Update Split os Screen and Num of Go routines based on number of CPUs
- [ ] Fix pointer dereference issue after long runs
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

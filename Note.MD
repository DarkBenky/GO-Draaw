# Vector Addition Benchmarking

This document compares the performance of vector addition using classic `float64` calculations versus `float32` calculations. 

## Vector Structures

```go
type Vector struct {
    x, y, z float64
}

type Vector32 struct {
    x, y, z float32
}

start := time.Now()
v := Vector{1, 2, 3}
testVec := Vector{0.1, 0.2, 0.3}
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
fmt.Println(v32, time.Since(start), "Vector SIMD Calculation")

{1.0000001641244583e+09 2.0000003282489166e+09 3.0000003040739307e+09} 8.090329462s
{1.0000001641244583e+09 2.0000003282489166e+09 3.0000003040739307e+09} 2.717395562s

```

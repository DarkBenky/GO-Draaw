# Performance of Vector Operations

**System Information:**

```go
{amd64 [AVX AES ADX OSXSAVE RDRAND RDSEED PCLMULQDQ FMA SSE2 SSE41 SSSE3 AVX2 SSE42 BMI1 POPCNT BMI2 CX16 SSE3] true}

v := Vector{1, 2, 3}
v2 := Vector{4, 5, 6}

start := time.Now()
for i := 0; i < 100000; i++ {
    v = v.Add(v2)
}
fmt.Println(time.Since(start), "result is Add", v)

80.372µs result is Add {400001 500002 600003}


start = time.Now()
for i := 0; i < 100000; i++ {
    v = v.Sub(v2)
}
fmt.Println(time.Since(start), "result is Sub", v)

91.343µs result is Sub {1 2 3}

dot := float32(0.0)
start = time.Now()
for i := 0; i < 100000; i++ {
    dot += v.Dot(v2)
}
fmt.Println(time.Since(start), "result is Dot", dot)

80.212µs result is Dot 3.2e+06

v3 := []float32{1, 2, 3}
v4 := []float32{4, 5, 6}

start = time.Now()
for i := 0; i < 10000; i++ {
    vek32.Add_Inplace(v3, v4)
}
fmt.Println(time.Since(start), "result is Add", v3)

64.542µs result is Add [40001 50002 60003]

start = time.Now()
for i := 0; i < 10000; i++ {
    vek32.Sub_Inplace(v3, v4)
}
fmt.Println(time.Since(start), "result is Sub", v3)

72.277µs result is Sub [1 2 3]

dot = float32(0.0)
start = time.Now()
for i := 0; i < 10000; i++ {
    dot -= vek32.Dot(v3, v4)
}
fmt.Println(time.Since(start), "result is Dot", v3, dot)

62.147µs result is Dot [1 2 3] -320000

```

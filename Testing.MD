addV := Vector{1, 1, 1}

func (v Vector) Add(v2 Vector) Vector {
	return Vector{v.x + v2.x, v.y + v2.y, v.z + v2.z}
}

start := time.Now()
v := Vector{1, 1, 1}
for i := 0; i < 1000000; i++ {
    v = v.Add(addV)
}
fmt.Println("Time taken:", time.Since(start), "v:", v)
Time taken: 959.414µs v: {1.000001e+06 1.000001e+06 1.000001e+06}

==========================================================================

func (v *Vector) AddP(v2 Vector) {
	v.x += v2.x
	v.y += v2.y
	v.z += v2.z
}

start = time.Now()
v1 := Vector{1, 1, 1}
for i := 0; i < 1000000; i++ {
    v1.AddP(addV)
}
fmt.Println("Time taken:", time.Since(start), "v1:", v1)
Time taken: 2.202097ms v1: {1.000001e+06 1.000001e+06 1.000001e+06}

==========================================================================

func (v *Vector) AddPP(v2 *Vector) {
	v.x += v2.x
	v.y += v2.y
	v.z += v2.z
}

start = time.Now()
v2 := Vector{1, 1, 1}
for i := 0; i < 1000000; i++ {
    v2.AddPP(&addV)
}
fmt.Println("Time taken:", time.Since(start), "v2:", v2)
Time taken: 2.392046ms v2: {1.000001e+06 1.000001e+06 1.000001e+06}

==========================================================================

func (v Vector) AddCP(v2 *Vector) Vector {
	return Vector{v.x + v2.x, v.y + v2.y, v.z + v2.z}
}

start = time.Now()
v3 := Vector{1, 1, 1}
for i := 0; i < 1000000; i++ {
    v3 = v3.AddCP(&addV)
}
fmt.Println("Time taken:", time.Since(start), "v3:", v3)
Time taken: 959.472µs v3: {1.000001e+06 1.000001e+06 1.000001e+06}

==========================================================================

start = time.Now()
v4 := Vector{1, 1, 1}
for i := 0; i < 1000000; i++ {
    v4 = Vector{v4.x + addV.x, v4.y + addV.y, v4.z + addV.z}
}
fmt.Println("Time taken:", time.Since(start), "v4:", v4)
Time taken: 962.99µs v4: {1.000001e+06 1.000001e+06 1.000001e+06}







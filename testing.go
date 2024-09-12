// main.go
package main

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart
#include <cuda_runtime.h>
#include "kernel.h"
*/
import "C"
import (
    "fmt"
    "unsafe"
)

const (
    N = 1024
)

func main() {
    // Allocate memory on host
    a := make([]float32, N*N)
    b := make([]float32, N*N)
    c := make([]float32, N*N)

    // Initialize matrices
    for i := 0; i < N*N; i++ {
        a[i] = float32(i)
        b[i] = float32(i)
    }

    // Allocate memory on device
    var d_a, d_b, d_c C.float*
    C.cudaMalloc(unsafe.Pointer(&d_a), C.size_t(N*N*4))
    C.cudaMalloc(unsafe.Pointer(&d_b), C.size_t(N*N*4))
    C.cudaMalloc(unsafe.Pointer(&d_c), C.size_t(N*N*4))

    // Copy data from host to device
    C.cudaMemcpy(unsafe.Pointer(d_a), unsafe.Pointer(&a[0]), C.size_t(N*N*4), C.cudaMemcpyHostToDevice)
    C.cudaMemcpy(unsafe.Pointer(d_b), unsafe.Pointer(&b[0]), C.size_t(N*N*4), C.cudaMemcpyHostToDevice)

    // Launch kernel
    C.matrixMultiply(d_a, d_b, d_c, C.int(N))

    // Copy result from device to host
    C.cudaMemcpy(unsafe.Pointer(&c[0]), unsafe.Pointer(d_c), C.size_t(N*N*4), C.cudaMemcpyDeviceToHost)

    // Free device memory
    C.cudaFree(unsafe.Pointer(d_a))
    C.cudaFree(unsafe.Pointer(d_b))
    C.cudaFree(unsafe.Pointer(d_c))

    // Print a small portion of the result
    for i := 0; i < 10; i++ {
        fmt.Printf("%f ", c[i])
    }
    fmt.Println()
}
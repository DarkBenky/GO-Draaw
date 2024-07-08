#include <stdio.h>
#include <cuda.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return;}} while(0)


__global__ void vecAdd(float *x, float *y, float *z, float *x1, float *y1, float *z1, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        x[idx] = x[idx] + x1[idx];
        y[idx] = y[idx] + y1[idx];
        z[idx] = z[idx] + z1[idx];
    }
}

__global__ void vecSub(float *x, float *y, float *z, float *x1, float *y1, float *z1, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        x[idx] = x[idx] - x1[idx];
        y[idx] = y[idx] - y1[idx];
        z[idx] = z[idx] - z1[idx];
    }
}


extern "C" {

   void vectorAdd(float *x, float *y, float *z, float *x1, float *y1, float *z1, int numElements) {
        // Allocate device memory
        float *gpu_x, *gpu_y, *gpu_z, *gpu_x1, *gpu_y1, *gpu_z1;
        int memSize = numElements * sizeof(float);
        cudaMalloc((void**)&gpu_x, memSize);
        cudaMalloc((void**)&gpu_y, memSize);
        cudaMalloc((void**)&gpu_z, memSize);
        cudaMalloc((void**)&gpu_x1, memSize);
        cudaMalloc((void**)&gpu_y1, memSize);
        cudaMalloc((void**)&gpu_z1, memSize);

        // Copy input vectors from host memory to device memory
        cudaMemcpy(gpu_x, x, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y, y, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_z, z, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_x1, x1, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y1, y1, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_z1, z1, memSize, cudaMemcpyHostToDevice);

        // Calculate grid and block dimensions
        int threadsPerBlock = 32;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the vector addition kernel
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(gpu_x, gpu_y, gpu_z, gpu_x1, gpu_y1, gpu_z1, numElements);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        }

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // Copy result from device memory to host memory
        cudaMemcpy(x, gpu_x, memSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(y, gpu_y, memSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(z, gpu_z, memSize, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(gpu_x);
        cudaFree(gpu_y);
        cudaFree(gpu_z);
        cudaFree(gpu_x1);
        cudaFree(gpu_y1);
        cudaFree(gpu_z1);
    }

     void vectorSub(float *x, float *y, float *z, float *x1, float *y1, float *z1, int numElements) {
        // Allocate device memory
        float *gpu_x, *gpu_y, *gpu_z, *gpu_x1, *gpu_y1, *gpu_z1;
        int memSize = numElements * sizeof(float);
        cudaMalloc((void**)&gpu_x, memSize);
        cudaMalloc((void**)&gpu_y, memSize);
        cudaMalloc((void**)&gpu_z, memSize);
        cudaMalloc((void**)&gpu_x1, memSize);
        cudaMalloc((void**)&gpu_y1, memSize);
        cudaMalloc((void**)&gpu_z1, memSize);

        // Copy input vectors from host memory to device memory
        cudaMemcpy(gpu_x, x, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y, y, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_z, z, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_x1, x1, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y1, y1, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_z1, z1, memSize, cudaMemcpyHostToDevice);

        // Calculate grid and block dimensions
        int threadsPerBlock = 32;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the vector subtraction kernel
        vecSub<<<blocksPerGrid, threadsPerBlock>>>(gpu_x, gpu_y, gpu_z, gpu_x1, gpu_y1, gpu_z1, numElements);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        }

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // Copy result from device memory to host memory
        cudaMemcpy(x, gpu_x, memSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(y, gpu_y, memSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(z, gpu_z, memSize, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(gpu_x);
        cudaFree(gpu_y);
        cudaFree(gpu_z);
        cudaFree(gpu_x1);
        cudaFree(gpu_y1);
        cudaFree(gpu_z1);
    }
}

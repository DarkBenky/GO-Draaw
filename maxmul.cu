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

__global__ void Normalize(float *x, float *y, float *z, int numVectors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numVectors) {
        float magnitude = sqrt(x[idx] * x[idx] + y[idx] * y[idx] + z[idx] * z[idx]);
        if (magnitude > 0.0f) {
            x[idx] /= magnitude;
            y[idx] /= magnitude;
            z[idx] /= magnitude;
        }
    }
}

__global__ void vecDot(float *x, float *y, float *z, float *x1, float *y1, float *z1, float *dotProduct, int numElements) {
    __shared__ float cache[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIdx = threadIdx.x;

    float temp = 0;
    if (idx < numElements) {
        temp = x[idx] * x1[idx] + y[idx] * y1[idx] + z[idx] * z1[idx];
    }

    cache[cacheIdx] = temp;

    __syncthreads();

    // Reduction
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIdx == 0) {
        atomicAdd(dotProduct, cache[0]);
    }
}

__global__ void vecCross(float *x, float *y, float *z, float *x1, float *y1, float *z1, float *cx, float *cy, float *cz, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        cx[idx] = y[idx] * z1[idx] - z[idx] * y1[idx];
        cy[idx] = z[idx] * x1[idx] - x[idx] * z1[idx];
        cz[idx] = x[idx] * y1[idx] - y[idx] * x1[idx];
    }
}


extern "C" {

    void vectorNormalize(float *x, float *y, float *z, int numElements) {
        float *gpu_x, *gpu_y, *gpu_z;
        int memSize = numElements * sizeof(float);

        cudaMalloc((void**)&gpu_x, memSize);
        cudaMalloc((void**)&gpu_y, memSize);
        cudaMalloc((void**)&gpu_z, memSize);

        cudaMemcpy(gpu_x, x, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y, y, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_z, z, memSize, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        Normalize<<<blocksPerGrid, threadsPerBlock>>>(gpu_x, gpu_y, gpu_z, numElements);
        cudaDeviceSynchronize();

        cudaMemcpy(x, gpu_x, memSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(y, gpu_y, memSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(z, gpu_z, memSize, cudaMemcpyDeviceToHost);

        cudaFree(gpu_x);
        cudaFree(gpu_y);
        cudaFree(gpu_z);
    }

    void vectorDot(float *x, float *y, float *z, float *x1, float *y1, float *z1, float *result, int numElements) {
        float *gpu_x, *gpu_y, *gpu_z, *gpu_x1, *gpu_y1, *gpu_z1, *gpu_result;
        int memSize = numElements * sizeof(float);
        cudaMalloc((void**)&gpu_x, memSize);
        cudaMalloc((void**)&gpu_y, memSize);
        cudaMalloc((void**)&gpu_z, memSize);
        cudaMalloc((void**)&gpu_x1, memSize);
        cudaMalloc((void**)&gpu_y1, memSize);
        cudaMalloc((void**)&gpu_z1, memSize);
        cudaMalloc((void**)&gpu_result, sizeof(float));

        cudaMemcpy(gpu_x, x, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y, y, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_z, z, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_x1, x1, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y1, y1, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_z1, z1, memSize, cudaMemcpyHostToDevice);

        cudaMemset(gpu_result, 0, sizeof(float));

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        vecDot<<<blocksPerGrid, threadsPerBlock>>>(gpu_x, gpu_y, gpu_z, gpu_x1, gpu_y1, gpu_z1, gpu_result, numElements);
        cudaDeviceSynchronize();

        cudaMemcpy(result, gpu_result, sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(gpu_x);
        cudaFree(gpu_y);
        cudaFree(gpu_z);
        cudaFree(gpu_x1);
        cudaFree(gpu_y1);
        cudaFree(gpu_z1);
        cudaFree(gpu_result);
    }

    void vectorCross(float *x, float *y, float *z, float *x1, float *y1, float *z1, float *cx, float *cy, float *cz, int numElements) {
        float *gpu_x, *gpu_y, *gpu_z, *gpu_x1, *gpu_y1, *gpu_z1, *gpu_cx, *gpu_cy, *gpu_cz;
        int memSize = numElements * sizeof(float);
        cudaMalloc((void**)&gpu_x, memSize);
        cudaMalloc((void**)&gpu_y, memSize);
        cudaMalloc((void**)&gpu_z, memSize);
        cudaMalloc((void**)&gpu_x1, memSize);
        cudaMalloc((void**)&gpu_y1, memSize);
        cudaMalloc((void**)&gpu_z1, memSize);
        cudaMalloc((void**)&gpu_cx, memSize);
        cudaMalloc((void**)&gpu_cy, memSize);
        cudaMalloc((void**)&gpu_cz, memSize);

        cudaMemcpy(gpu_x, x, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y, y, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_z, z, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_x1, x1, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y1, y1, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_z1, z1, memSize, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        vecCross<<<blocksPerGrid, threadsPerBlock>>>(gpu_x, gpu_y, gpu_z, gpu_x1, gpu_y1, gpu_z1, gpu_cx, gpu_cy, gpu_cz, numElements);
        cudaDeviceSynchronize();

        cudaMemcpy(cx, gpu_cx, memSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(cy, gpu_cy, memSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(cz, gpu_cz, memSize, cudaMemcpyDeviceToHost);

        cudaFree(gpu_x);
        cudaFree(gpu_y);
        cudaFree(gpu_z);
        cudaFree(gpu_x1);
        cudaFree(gpu_y1);
        cudaFree(gpu_z1);
        cudaFree(gpu_cx);
        cudaFree(gpu_cy);
        cudaFree(gpu_cz);
    }

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
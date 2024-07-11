#include <stdio.h>
#include <cuda.h>

// Define a small constant for floating-point comparisons
#define EPSILON 0.00001f

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

// Structure to represent a ray
struct Ray {
    float origin[3];
    float direction[3];
};

// Structure to represent a triangle
struct Triangle {
    float v1[3];
    float v2[3];
    float v3[3];
    float color[3];
};

// Structure to represent an intersection
struct Intersection {
    float PointOfIntersection[3];
    float Color[3];
    float Normal[3];
    float Direction[3];
    float Distance;
};

// CUDA kernel to perform ray-triangle intersection test
__global__ void IntersectTriangleKernel(Ray *rays, Triangle *triangles, Intersection *intersections, bool *hits, int numRays, int numTriangles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numRays * numTriangles) {
        int rayIdx = idx / numTriangles;
        int triIdx = idx % numTriangles;

        Ray ray = rays[rayIdx];
        Triangle triangle = triangles[triIdx];

        // Implement the Möller–Trumbore intersection algorithm
        float edge1[3], edge2[3], h[3], s[3], q[3];
        float a, f, u, v, t;

        for (int i = 0; i < 3; ++i) {
            edge1[i] = triangle.v2[i] - triangle.v1[i];
            edge2[i] = triangle.v3[i] - triangle.v1[i];
        }

        // Cross product of ray direction and edge2
        h[0] = ray.direction[1] * edge2[2] - ray.direction[2] * edge2[1];
        h[1] = ray.direction[2] * edge2[0] - ray.direction[0] * edge2[2];
        h[2] = ray.direction[0] * edge2[1] - ray.direction[1] * edge2[0];

        a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

        if (a > -EPSILON && a < EPSILON) {
            hits[idx] = false;
            return;
        }

        f = 1.0f / a;

        for (int i = 0; i < 3; ++i) {
            s[i] = ray.origin[i] - triangle.v1[i];
        }

        u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);

        if (u < 0.0f || u > 1.0f) {
            hits[idx] = false;
            return;
        }

        q[0] = s[1] * edge1[2] - s[2] * edge1[1];
        q[1] = s[2] * edge1[0] - s[0] * edge1[2];
        q[2] = s[0] * edge1[1] - s[1] * edge1[0];

        v = f * (ray.direction[0] * q[0] + ray.direction[1] * q[1] + ray.direction[2] * q[2]);

        if (v < 0.0f || u + v > 1.0f) {
            hits[idx] = false;
            return;
        }

        t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);

        if (t > 0.00001) {
            for (int i = 0; i < 3; ++i) {
                intersections[idx].PointOfIntersection[i] = ray.origin[i] + ray.direction[i] * t;
                intersections[idx].Color[i] = triangle.color[i];
                intersections[idx].Normal[i] = edge1[(i+1)%3] * edge2[(i+2)%3] - edge1[(i+2)%3] * edge2[(i+1)%3];
                intersections[idx].Direction[i] = ray.direction[i];
            }

            intersections[idx].Distance = t;
            hits[idx] = true;
        } else {
            hits[idx] = false;
        }
    }
}

// Function to call the CUDA kernel for intersection tests
extern "C" void IntersectTriangles(Ray *rays, Triangle *triangles, Intersection *intersections, bool *hits, int numRays, int numTriangles) {
    Ray *gpu_rays;
    Triangle *gpu_triangles;
    Intersection *gpu_intersections;
    bool *gpu_hits;
    int raySize = numRays * sizeof(Ray);
    int triSize = numTriangles * sizeof(Triangle);
    int intSize = numRays * numTriangles * sizeof(Intersection);
    int boolSize = numRays * numTriangles * sizeof(bool);

    cudaMalloc((void**)&gpu_rays, raySize);
    cudaMalloc((void**)&gpu_triangles, triSize);
    cudaMalloc((void**)&gpu_intersections, intSize);
    cudaMalloc((void**)&gpu_hits, boolSize);

    cudaMemcpy(gpu_rays, rays, raySize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_triangles, triangles, triSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numRays * numTriangles + threadsPerBlock - 1) / threadsPerBlock;

    IntersectTriangleKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_rays, gpu_triangles, gpu_intersections, gpu_hits, numRays, numTriangles);
    cudaMemcpy(intersections, gpu_intersections, intSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hits, gpu_hits, boolSize, cudaMemcpyDeviceToHost);

    cudaFree(gpu_rays);
    cudaFree(gpu_triangles);
    cudaFree(gpu_intersections);
    cudaFree(gpu_hits);
}
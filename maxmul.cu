#include <stdio.h>
#include <cuda.h>

// Define a small constant for floating-point comparisons
#define EPSILON 0.00001f
// Define an invalid distance value for rays to infinity
#define INVALID_DISTANCE 0.0f

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

        int threadsPerBlock = 2048;
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

        int threadsPerBlock = 2048;
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

        int threadsPerBlock = 2048;
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
        int threadsPerBlock = 2048;
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
        int threadsPerBlock = 2048;
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
    int x;
    int y;
};

// Structure to represent a triangle
struct Triangle {
    float v1[3];
    float v2[3];
    float v3[3];
    float color[3];
    float boundingBox[6];
};

struct Object {
    Triangle *triangles;
    int numTriangles;
    float boundingBox[6];
    char padding[4]; // 4 bytes of padding
};

// Structure to represent an intersection result
struct IntersectionResult {
    float intersectionPoint[3];
    float triangleColor[3];
    float distance;
    int x;
    int y;
};

__device__ bool RayIntersectsBox(float3 origin, float3 dir, float3 boxMin, float3 boxMax, float *tmin, float *tmax) {
    float t1 = (boxMin.x - origin.x) / dir.x;
    float t2 = (boxMax.x - origin.x) / dir.x;
    float t3 = (boxMin.y - origin.y) / dir.y;
    float t4 = (boxMax.y - origin.y) / dir.y;
    float t5 = (boxMin.z - origin.z) / dir.z;
    float t6 = (boxMax.z - origin.z) / dir.z;

    *tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    *tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

    return *tmax >= *tmin;
}

__global__ void IntersectTrianglesKernel(Ray *rays, Object *objects, int numObjects, IntersectionResult *results, int numRays) {
    int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rayIdx < numRays) {
        Ray ray = rays[rayIdx];
        IntersectionResult closestIntersection;
        closestIntersection.distance = INFINITY;

        for (int objIdx = 0; objIdx < numObjects; ++objIdx) {
            Object obj = objects[objIdx];

            // Check if the ray intersects the bounding box of the object
            float tmin, tmax;
            if (!RayIntersectsBox(make_float3(ray.origin[0], ray.origin[1], ray.origin[2]),
                                  make_float3(ray.direction[0], ray.direction[1], ray.direction[2]),
                                  make_float3(obj.boundingBox[0], obj.boundingBox[1], obj.boundingBox[2]),
                                  make_float3(obj.boundingBox[3], obj.boundingBox[4], obj.boundingBox[5]),
                                  &tmin, &tmax)) {
                continue; // Skip this object if no intersection with its bounding box
            }

            for (int triIdx = 0; triIdx < obj.numTriangles; ++triIdx) {
                Triangle tri = obj.triangles[triIdx];
                
                // Check if the ray intersects the bounding box of the triangle
                if (!RayIntersectsBox(make_float3(ray.origin[0], ray.origin[1], ray.origin[2]),
                                      make_float3(ray.direction[0], ray.direction[1], ray.direction[2]),
                                      make_float3(tri.boundingBox[0], tri.boundingBox[1], tri.boundingBox[2]),
                                      make_float3(tri.boundingBox[3], tri.boundingBox[4], tri.boundingBox[5]),
                                      &tmin, &tmax)) {
                    continue; // Skip this triangle if no intersection with its bounding box
                }

                // Implement the Möller–Trumbore intersection algorithm
                float edge1[3], edge2[3], h[3], s[3], q[3];
                float a, f, u, v, t;

                for (int i = 0; i < 3; ++i) {
                    edge1[i] = tri.v2[i] - tri.v1[i];
                    edge2[i] = tri.v3[i] - tri.v1[i];
                }

                // Cross product of ray direction and edge2
                h[0] = ray.direction[1] * edge2[2] - ray.direction[2] * edge2[1];
                h[1] = ray.direction[2] * edge2[0] - ray.direction[0] * edge2[2];
                h[2] = ray.direction[0] * edge2[1] - ray.direction[1] * edge2[0];

                a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

                if (a > -EPSILON && a < EPSILON) {
                    continue;
                }

                f = 1.0f / a;

                for (int i = 0; i < 3; ++i) {
                    s[i] = ray.origin[i] - tri.v1[i];
                }

                u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);

                if (u < 0.0f || u > 1.0f) {
                    continue;
                }

                q[0] = s[1] * edge1[2] - s[2] * edge1[1];
                q[1] = s[2] * edge1[0] - s[0] * edge1[2];
                q[2] = s[0] * edge1[1] - s[1] * edge1[0];

                v = f * (ray.direction[0] * q[0] + ray.direction[1] * q[1] + ray.direction[2] * q[2]);

                if (v < 0.0f || u + v > 1.0f) {
                    continue;
                }

                t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);

                if (t > EPSILON && t < closestIntersection.distance) {
                    closestIntersection.distance = t;
                    closestIntersection.intersectionPoint[0] = ray.origin[0] + t * ray.direction[0];
                    closestIntersection.intersectionPoint[1] = ray.origin[1] + t * ray.direction[1];
                    closestIntersection.intersectionPoint[2] = ray.origin[2] + t * ray.direction[2];
                    closestIntersection.triangleColor[0] = tri.color[0];
                    closestIntersection.triangleColor[1] = tri.color[1];
                    closestIntersection.triangleColor[2] = tri.color[2];
                    closestIntersection.x = ray.x;
                    closestIntersection.y = ray.y;
                }
            }
        }

        results[rayIdx] = closestIntersection;
    }
}

extern "C" void IntersectTriangles(Ray *rays, Object *objects, IntersectionResult *results, int numRays, int numObjects) {
    Ray *gpu_rays;
    Object *gpu_objects;
    Triangle *gpu_triangles;
    IntersectionResult *gpu_results;
    int raySize = numRays * sizeof(Ray);
    int objectSize = numObjects * sizeof(Object);
    int totalTriangles = 0;

    for (int i = 0; i < numObjects; ++i) {
        totalTriangles += objects[i].numTriangles;
    }

    int triangleSize = totalTriangles * sizeof(Triangle);
    int resultsSize = numRays * sizeof(IntersectionResult);

    cudaMalloc((void**)&gpu_rays, raySize);
    cudaMalloc((void**)&gpu_objects, objectSize);
    cudaMalloc((void**)&gpu_triangles, triangleSize);
    cudaMalloc((void**)&gpu_results, resultsSize);

    cudaMemcpy(gpu_rays, rays, raySize, cudaMemcpyHostToDevice);

    // Copy objects and their triangles to device
    Object *objectsOnHost = (Object*)malloc(objectSize);
    Triangle *trianglesOnHost = (Triangle*)malloc(triangleSize);

    int triangleOffset = 0;
    for (int i = 0; i < numObjects; ++i) {
        objectsOnHost[i] = objects[i];
        objectsOnHost[i].triangles = &gpu_triangles[triangleOffset];
        cudaMemcpy(&trianglesOnHost[triangleOffset], objects[i].triangles, objects[i].numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
        triangleOffset += objects[i].numTriangles;
    }

    cudaMemcpy(gpu_objects, objectsOnHost, objectSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_triangles, trianglesOnHost, triangleSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;

    IntersectTrianglesKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_rays, gpu_objects, numObjects, gpu_results, numRays);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(results, gpu_results, resultsSize, cudaMemcpyDeviceToHost);

    cudaFree(gpu_rays);
    cudaFree(gpu_objects);
    cudaFree(gpu_triangles);
    cudaFree(gpu_results);
    free(objectsOnHost);
    free(trianglesOnHost);
}


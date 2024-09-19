#ifndef RAY_INTERSECT_H
#define RAY_INTERSECT_H

#include <stdbool.h>

// Ray structure: defines a ray with an origin and direction
typedef struct {
    float origin[3];
    float direction[3];
} Ray;

// Bounding box (AABB) structure: defines the minimum and maximum extents of the box
typedef struct {
    float min[3];
    float max[3];
} BBox;

// Triangle structure: defines a triangle by its three vertices
typedef struct {
    float v0[3];
    float v1[3];
    float v2[3];
} Triangle;

// Function to check if a ray intersects a bounding box (AABB)
bool ray_bbox_intersect(const Ray* ray, const BBox* bbox, float* t_near, float* t_far);

// Function to check if a ray intersects a triangle
bool ray_triangle_intersect(const Ray* ray, const Triangle* triangle, float* t, float* u, float* v);

// Function to batch process ray-triangle intersections
bool ray_triangle_intersect_batch(const Ray* rays, int ray_count, const Triangle* triangles, int triangle_count, bool* results);

#endif // RAY_INTERSECT_H

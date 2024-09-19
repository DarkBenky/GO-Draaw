#include <immintrin.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>  // Include this for fmaxf and fminf

typedef struct {
    float origin[3];
    float direction[3];
} Ray;

typedef struct {
    float min[3];
    float max[3];
} BBox;

typedef struct {
    float v0[3];
    float v1[3];
    float v2[3];
} Triangle;

bool ray_bbox_intersect(const Ray* ray, const BBox* bbox, float* t_near, float* t_far) {
    __m128 origin = _mm_loadu_ps(ray->origin);
    __m128 direction = _mm_loadu_ps(ray->direction);
    __m128 bmin = _mm_loadu_ps(bbox->min);
    __m128 bmax = _mm_loadu_ps(bbox->max);

    __m128 inv_direction = _mm_div_ps(_mm_set1_ps(1.0f), direction);
    __m128 t1 = _mm_mul_ps(_mm_sub_ps(bmin, origin), inv_direction);
    __m128 t2 = _mm_mul_ps(_mm_sub_ps(bmax, origin), inv_direction);

    __m128 tmin = _mm_min_ps(t1, t2);
    __m128 tmax = _mm_max_ps(t1, t2);

    float tmin_arr[4], tmax_arr[4];
    _mm_storeu_ps(tmin_arr, tmin);
    _mm_storeu_ps(tmax_arr, tmax);

    *t_near = fmaxf(fmaxf(tmin_arr[0], tmin_arr[1]), tmin_arr[2]);
    *t_far = fminf(fminf(tmax_arr[0], tmax_arr[1]), tmax_arr[2]);

    return *t_near <= *t_far && *t_far > 0;
}

// Helper function to compute cross product using SIMD
__m128 cross_product_simd(__m128 a, __m128 b) {
    __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 c = _mm_sub_ps(
        _mm_mul_ps(a, b_yzx),
        _mm_mul_ps(a_yzx, b)
    );
    return _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1));
}

bool ray_triangle_intersect(const Ray* ray, const Triangle* triangle, float* t, float* u, float* v) {
    __m128 v0 = _mm_loadu_ps(triangle->v0);
    __m128 v1 = _mm_loadu_ps(triangle->v1);
    __m128 v2 = _mm_loadu_ps(triangle->v2);
    __m128 orig = _mm_loadu_ps(ray->origin);
    __m128 dir = _mm_loadu_ps(ray->direction);

    __m128 edge1 = _mm_sub_ps(v1, v0);
    __m128 edge2 = _mm_sub_ps(v2, v0);
    __m128 pvec = cross_product_simd(dir, edge2);
    __m128 det = _mm_dp_ps(edge1, pvec, 0x7F);

    __m128 epsilon = _mm_set1_ps(1e-8f);
    if (_mm_comile_ss(det, epsilon) && _mm_comige_ss(det, _mm_sub_ps(_mm_setzero_ps(), epsilon)))
        return false;

    __m128 inv_det = _mm_div_ps(_mm_set1_ps(1.0f), det);
    __m128 tvec = _mm_sub_ps(orig, v0);
    __m128 u_vec = _mm_mul_ps(_mm_dp_ps(tvec, pvec, 0x7F), inv_det);

    if (_mm_comilt_ss(u_vec, _mm_setzero_ps()) || _mm_comigt_ss(u_vec, _mm_set1_ps(1.0f)))
        return false;

    __m128 qvec = cross_product_simd(tvec, edge1);
    __m128 v_vec = _mm_mul_ps(_mm_dp_ps(dir, qvec, 0x7F), inv_det);

    if (_mm_comilt_ss(v_vec, _mm_setzero_ps()) || _mm_comigt_ss(_mm_add_ps(u_vec, v_vec), _mm_set1_ps(1.0f)))
        return false;

    __m128 t_vec = _mm_mul_ps(_mm_dp_ps(edge2, qvec, 0x7F), inv_det);

    *t = _mm_cvtss_f32(t_vec);
    *u = _mm_cvtss_f32(u_vec);
    *v = _mm_cvtss_f32(v_vec);

    return true;
}

bool ray_triangle_intersect_batch(const Ray* rays, int ray_count, const Triangle* triangles, int triangle_count, bool* results) {
    for (int i = 0; i < ray_count; i++) {
        for (int j = 0; j < triangle_count; j++) {
            float t, u, v;
            results[i * triangle_count + j] = ray_triangle_intersect(&rays[i], &triangles[j], &t, &u, &v);
        }
    }
    return true;
}

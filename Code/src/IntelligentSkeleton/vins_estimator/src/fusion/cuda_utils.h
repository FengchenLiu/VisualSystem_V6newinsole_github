// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cmath>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define DIV_UP(n,div) (((n) + (div) - 1) / (div))

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define KERNEL_ABORT(msg, ...) {printf("KERNEL_ABORT: %s %d: " msg, __FILE__, __LINE__, ##__VA_ARGS__);asm("trap;");}

#define DR_FUSION_DEBUG

#ifdef DR_FUSION_DEBUG
#define DEBUG_PRINT(...) {printf(__VA_ARGS__);}
#else
#define DEBUG_PRINT(...) {}
#endif

inline unsigned int eventFlags(){
return cudaEventBlockingSync | cudaEventDisableTiming;
}

static inline int DivUp(int total, int grain) { return (total + grain - 1) / grain; }

__host__ __device__ inline float dot(const float3 a, const float3 b) {
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__host__ __device__ inline float norm(const float3 &vec) {
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__host__ __device__ inline float3 normalize(const float3 &vec) {
    float vec_norm = norm(vec);
    return make_float3(vec.x / vec_norm, vec.y / vec_norm, vec.z / vec_norm);
}

__host__ __device__ inline float3 cross(const float3 &a, const float3 &b) {
    float3 res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;
    return res;
}

__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline int3 operator+(const int3 &a, const int3 &b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3 &a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline float3 operator*(float b, const float3 &a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline float3 operator/(const float3 &a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ inline float distance(const float3 &a, const float3 &b) {
    return norm(b - a);
}

__host__ __device__ inline int sign(float n) { return (n > 0) - (n < 0); }

__host__ __device__ inline float signf(float value) {
    return (value > 0) - (value < 0);
}

__host__ __device__ inline float3 ColorToFloat(uchar3 c) {
    return make_float3(static_cast<float>(c.x) / 255,
                        static_cast<float>(c.y) / 255,
                        static_cast<float>(c.z) / 255);
}


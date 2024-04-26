#include <stdio.h>
#include <stdlib.h>

#define THREAD_N 100
#define BLOCK_SIZE 1024
#define BLOCKS 4
#define THREADS (BLOCKS * BLOCK_SIZE)
#define INTERVAL (1.f / THREADS)

__global__ void g_compute_values(float2 *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float x0 = idx * INTERVAL;
    float x_interval = INTERVAL;
    float x, temp, s1 = 0.f, s2 = 0.f;
    for (int i = 0; i < THREAD_N; i++) {
        x = x0 + x_interval * i / THREAD_N;
        temp = sqrtf(1.f - x * x);
        s2 += temp;
        s1 += temp * x;
    }
    
    data[idx].x = s1 / THREAD_N;
    data[idx].y = s2 / THREAD_N;
}

int main() {
    float2 *points, *d_points;
    points = (float2*) malloc(sizeof(float2) * THREADS);
    cudaMalloc((void**)&d_points, sizeof(float2) * THREADS);

    g_compute_values<<<dim3(BLOCK_SIZE), dim3(BLOCKS)>>>(d_points);
    cudaMemcpy(points, d_points, sizeof(float2) * THREADS, cudaMemcpyDeviceToHost);
    cudaFree(d_points);

    float up = 0.f, down = 0.f;
    for (int i = 0; i < THREADS; i++) {
        up += points[i].x;
        down += points[i].y;
    }
    up *= INTERVAL;
    down *= INTERVAL;

    printf("PI / 4 = %f\nx_c = %f\n", down, up / down);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHUNK 32
#define INDEX(x, y, x_size) ((x) + (y) * (x_size))
#define X 10000
#define Y 10000
#define Z 10000

// Sizes of matrices are multiple 16x16
// size.x (in chunks) -- x size of A = y size of B
// size.y (in chunks) -- y size of A
// size.z (in chunks) -- x size of B
__global__ void g_matrix_multiply(float *left, float *right, int3 size, float *result) {
    __shared__ float s_left[CHUNK][CHUNK];
    __shared__ float s_right[CHUNK][CHUNK];

    float sum = 0.f;
    int result_x = CHUNK * blockIdx.x + threadIdx.x;
    int result_y = CHUNK * blockIdx.y + threadIdx.y;

    for (int i = 0; i < size.x; i++) {
        // load blocks into shared memory
        s_left[threadIdx.y][threadIdx.x] = left[INDEX(threadIdx.x + i * CHUNK, result_y, CHUNK * size.x)];
        s_right[threadIdx.y][threadIdx.x] = right[INDEX(result_x, threadIdx.y + i * CHUNK, size.z * CHUNK)];
        __syncthreads();
        
        // compute value
        for (int j = 0; j < CHUNK; j++)
            sum += s_left[threadIdx.y][j] * s_right[j][threadIdx.x];

        __syncthreads();
    }

    result[INDEX(result_x, result_y, CHUNK * size.z)] = sum;
}

float matrix_multiply(float *left, float *right, int3 size, float *result) {
    float *h_left, *h_right, *h_result, *d_left, *d_right, *d_result;
    int3 new_size = make_int3((size.x + CHUNK - 1) / CHUNK, (size.y + CHUNK - 1) / CHUNK, (size.z + CHUNK - 1) / CHUNK);

    const size_t LEFT_SIZE = sizeof(float) * new_size.x * new_size.y * CHUNK * CHUNK;
    const size_t RIGHT_SIZE = sizeof(float) * new_size.x * new_size.z * CHUNK * CHUNK;
    const size_t RESULT_SIZE = sizeof(float) * new_size.y * new_size.z * CHUNK * CHUNK;

    h_left = (float *) calloc(LEFT_SIZE / sizeof(float), sizeof(float));
    h_right = (float *) calloc(RIGHT_SIZE / sizeof(float), sizeof(float));
    h_result = (float *) calloc(RESULT_SIZE / sizeof(float), sizeof(float));
    
    cudaMalloc((void **) &d_left, LEFT_SIZE);
    cudaMalloc((void **) &d_right, RIGHT_SIZE);
    cudaMalloc((void **) &d_result, RESULT_SIZE);

    // copy matrices
    for (int i = 0; i < size.y; i++) memcpy(h_left + i * CHUNK * new_size.x, left + i * size.x, sizeof(float) * size.x);
    for (int i = 0; i < size.x; i++) memcpy(h_right + i * CHUNK * new_size.z, right + i * size.z, sizeof(float) * size.z);

    cudaMemcpy(d_left, h_left, LEFT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, h_right, RIGHT_SIZE, cudaMemcpyHostToDevice);

    // multiply
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    g_matrix_multiply<<<dim3(new_size.z, new_size.y), dim3(CHUNK, CHUNK)>>>(d_left, d_right, new_size, d_result);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ETA;
    cudaEventElapsedTime(&ETA, start, stop);

    // copy result
    cudaMemcpy(h_result, d_result, RESULT_SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size.y; i++) memcpy(result + i * size.z, h_result + i * CHUNK * new_size.z, sizeof(float) * size.z);

    // free memory
    free(h_left);
    free(h_right);
    free(h_result);
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_result);

    // return time, ms
    return ETA;
}

float *generate_matrix(int x, int y, float min = 0.f, float max = 10.f,  int seed = 0) {
    float *result = (float *) malloc(sizeof(float) * x * y);
    srand(seed);

    for (int i = 0; i < x * y; i++) result[i] = min + rand() * (max - min) / ((float) RAND_MAX);

    return result;
} 

void print_matrix(float *matrix, int x, int y, FILE *fp) {
    for (int row = 0; row < y; row++) {
        for (int column = 0; column < x; column++) {
            fprintf(fp, "%f ", matrix[INDEX(column, row, x)]);
        } 
        fprintf(fp, "\n");
    }
}

int main() {
    // generate 3 matrices
    // multiply
    // print
    // print time

    int3 size = make_int3(X, Y, Z);
    float *A = generate_matrix(size.x, size.y);
    float *B = generate_matrix(size.z, size.x);
    float *C = generate_matrix(size.z, size.y);

    double time = matrix_multiply(A, B, size, C);

    printf("Time: %f ms\n", time);

    FILE *f = fopen("A.matrix", "w");
    print_matrix(A, size.x, size.y, f);
    fclose(f);

    f = fopen("B.matrix", "w");
    print_matrix(B, size.z, size.x, f);
    fclose(f);

    f = fopen("C.matrix", "w");
    print_matrix(C, size.z, size.y, f);
    fclose(f);

    free(A);
    free(B);
    free(C);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// part = 0 or 1
// position -- uint3
// size -- uint3
#define INDEX(position, size) (position.x + position.y * size.x + position.z * size.x * size.y)
#define PBC(position, size) ((position) - (position) / (size) + (size) * ((position) < 0))

#define BLOCK_SIZE_1D 8
#define BLOCK_SIZE dim3(BLOCK_SIZE_1D, BLOCK_SIZE_1D, BLOCK_SIZE_1D)

texture<int, 1, cudaReadModeElementType> tex0;
texture<int, 1, cudaReadModeElementType> tex1;

__global__ void g_make_step_slow(int *in, int *out, int3 size) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    // get cell value
    int f = in[INDEX(make_int3(x, y, z), size)];

    // get sum of neighbour cell values
    int sigma = 0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++)
                sigma += in[INDEX(make_int3(PBC(x + i, size.x), PBC(y + j, size.y), PBC(z + k, size.z)), size)];

    sigma -= f;

    // set new cell value
    out[INDEX(make_int3(x, y, z), size)] = ((sigma / (2 * f + 2) - (3 - 2 * f)) == 0);

}

__global__ void g_make_step(int *out, int3 size, int offset) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    // get cell value
    int f = tex1Dfetch(offset ? tex1 : tex0, INDEX(make_int3(x, y, z), size));

    // get sum of neighbour cell values
    int sigma = 0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++)
                sigma += tex1Dfetch(offset ? tex1 : tex0, INDEX(make_int3(PBC(x + i, size.x), PBC(y + j, size.y), PBC(z + k, size.z)), size));

    sigma -= f;

    // set new cell value
    out[INDEX(make_int3(x, y, z), size)] = ((sigma / (2 * f + 2) - (3 - 2 * f)) == 0);
}


typedef struct {
    FILE *input;        // input file path
    int steps;          // total number of steps
    int output_every;   // dump once in this steps
    FILE *output;       // output file path
} InputArgs;

InputArgs parse_cli(int argc, char *argv[]) {
    InputArgs args;

    args.input = fopen(argv[1], "r");
    args.steps = atoi(argv[2]);
    args.output_every = atoi(argv[3]);
    args.output = fopen(argv[4], "w");

    return args;
}

typedef struct {
    int *array;
    int3 size;
} Array;

Array parse_input_file(InputArgs args) {
    Array result;
    
    // read first line
    int stype;
    fscanf(args.input, "%d %d %d %d\n", &result.size.x, &result.size.y, &result.size.z, &stype);    
    assert(stype == 1);
    assert(result.size.x > 0 && result.size.x % 8 == 0 &&
           result.size.y > 0 && result.size.y % 8 == 0 &&
           result.size.z > 0 && result.size.z % 8 == 0);

    // create array
    result.array = (int*) calloc(result.size.x * result.size.y * result.size.z, sizeof(int));

    // read other lines
    int4 numbers;
    do {
        fscanf(args.input, "%d %d %d %d\n", &numbers.x, &numbers.y, &numbers.z, &numbers.w);
        assert(numbers.x == 0 || numbers.x == 1);
        result.array[INDEX(make_int3(numbers.y, numbers.z, numbers.w), result.size)] = 1;
    } while (numbers.x != 0);
    result.array[INDEX(make_int3(0, 0, 0), result.size)] = 0;

    fclose(args.input);
    
    return result;
}

void dump_file(Array array, InputArgs args) {
    for (int i = 0; i < array.size.z; i++)
        for (int j = 0; j < array.size.y; j++)
            for (int k = 0; k < array.size.x; k++)
                if (array.array[INDEX(make_int3(k, j, i), array.size)] == 1)
                    fprintf(args.output, "1 %d %d %d\n", k, j, i);
    fprintf(args.output, "0 0 0 0\n");
}



void simulate_slow(Array array, InputArgs args) {
    size_t array_size = array.size.x * array.size.y * array.size.z;

    // allocate gpu memory
    int *d_arrays[2];
    cudaMalloc((void**) &(d_arrays[0]), array_size * sizeof(int));
    cudaMalloc((void**) &(d_arrays[1]), array_size * sizeof(int));

    cudaMemcpy(d_arrays[0], array.array, array_size * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0, offset = 0; i <= args.steps; i++, offset = 1 - offset) {
        // dump
        if (i % args.output_every == 0) {
            cudaMemcpy(array.array, d_arrays[offset], array_size * sizeof(int), cudaMemcpyDeviceToHost);
            dump_file(array, args);
        }

        // simulate 1 step
        g_make_step_slow<<<dim3(array.size.x / BLOCK_SIZE_1D, array.size.y / BLOCK_SIZE_1D, array.size.z / BLOCK_SIZE_1D), BLOCK_SIZE>>>(d_arrays[offset], d_arrays[1 - offset], array.size);
    }

    // free resources
    cudaFree(d_arrays[0]);
    cudaFree(d_arrays[1]);

    fclose(args.output);
}

void simulate(Array array, InputArgs args) {
    size_t array_size = array.size.x * array.size.y * array.size.z;

    // allocate gpu memory
    int *d_arrays[2];
    cudaMalloc((void**) &(d_arrays[0]), array_size * sizeof(int));
    cudaMalloc((void**) &(d_arrays[1]), array_size * sizeof(int));

    // bind textures
    cudaBindTexture(0, tex0, d_arrays[0], array_size * sizeof(int));
    cudaBindTexture(0, tex1, d_arrays[1], array_size * sizeof(int));

    cudaMemcpy(d_arrays[0], array.array, array_size * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0, offset = 0; i <= args.steps; i++, offset = 1 - offset) {
        // dump
        if (i % args.output_every == 0) {
            cudaMemcpy(array.array, d_arrays[offset], array_size * sizeof(int), cudaMemcpyDeviceToHost);
            dump_file(array, args);
        }

        // simulate 1 step
        g_make_step<<<dim3(array.size.x / BLOCK_SIZE_1D, array.size.y / BLOCK_SIZE_1D, array.size.z / BLOCK_SIZE_1D), BLOCK_SIZE>>>(d_arrays[1 - offset], array.size, offset);
    }

    // free resources
    cudaUnbindTexture(tex0);
    cudaUnbindTexture(tex1);
    cudaFree(d_arrays[0]);
    cudaFree(d_arrays[1]);

    fclose(args.output);
}


void print_limitations() {
    printf("1) Simulation box sizes must be multiple to 8\n");
}


int main(int argc, char *argv[]) {
    print_limitations();

    InputArgs args = parse_cli(argc, argv);
    Array input_array = parse_input_file(args);

#ifdef SLOW
    printf("Slow version\n");
    simulate_slow(input_array, args);
#else
    printf("Fast version\n");
    simulate(input_array, args);
#endif

    free(input_array.array);

    return 0;
}

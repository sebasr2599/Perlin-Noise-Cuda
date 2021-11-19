%%cu
#include "cuda_runtime.h"
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#define octaves 10  
__device__ int seed = 1;
__device__ unsigned long int primes[20] = {
    114361856600581,
    793723802738963,
    507065243087687,
    986362030125811,
    515223368260921,
    361462805486863,
    616197812332057,
    386551013564629,
    848421354910463,
    422707778776141,
    887186845898627,
    537795195251767,
    709992061000289,
    717010824065339,
    932221247968667,
    396539871811511,
    614717352566687,
    943963547058703,
    676356983748367,
    855207854365559
};


__device__ float noiseLCG(int octNum, int x, int y);
__device__ float smoothNoise(int octNum, int x, int y);
__device__ float interpolation(float a, float b, float x);
__device__ float InterpolatedNoise(int i, float x, float y);

__device__ float Perlin2D(float x, float y) {
  float total = 0,frq = pow(2, octaves), amp = 1;
  for (int i = 0; i <= octaves; i++) {
    frq /= 2;
    amp *= 0.2;
    total += InterpolatedNoise(i,x * frq, y* frq) * amp;
  }
  return total / frq;
}
__global__ void Perlin2DCuda(float *mat, int x_size,int y_size){
  int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if(row < x_size && col < y_size) {
    mat[x_size*row+col] = Perlin2D(row,col);
	}
}
void print_mat(float* matrix, int n, int m) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			printf("%.4f\t", matrix[i*m + j]);
		}
		printf("\n");
	}
	printf("\n");
}
void print_copyright(){
  printf("Noise2D  Copyright (C) 2021  Sebastian Resendiz Chavez\n This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.\nThis is free software, and you are welcome to redistribute it\nunder certain conditions; type `show c' for details.\n\n");
}
int main(int argc, char *argv[]){
    print_copyright();
    float *a, *d_a;
    int x_size=10;
    int y_size=10;
    int n_threads = x_size;
	  int n_blocks = y_size;
	  dim3 Blocks(n_blocks, n_blocks);
	  dim3 Threads(n_threads, n_threads);

    int size = x_size * y_size * sizeof(float);
    a = (float*)malloc(size);

    cudaMalloc((void**) &d_a, size);
    // cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    clock_t tiempogpu = clock();
    Perlin2DCuda<<<Blocks,Threads>>>(d_a,x_size,y_size);
    printf("Tiempo transcurrido al procesador en GPU: %f\n", ((double)clock() - tiempogpu) / CLOCKS_PER_SEC);

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    
    print_mat(a, x_size,y_size);

    free(a);
    cudaFree(d_a);
    return 0;
}


__device__ float noiseLCG(int octNum, int x, int y){
    int n = seed + x + y;
    n = (n << 13) ^ n;
    int a = (seed*octNum)%20;
    int b = ((x/seed)*octNum)%20;
    int c = ((x+b/seed)*octNum)%20;
    int temp = (n * (n * n * primes[a] + primes[b]) + primes[c]) & 0x7fffffff;
    return 1.0 - (float)(temp)/1073741824.0;
}
__device__ float smoothNoise(int octNum, int x, int y) {
    //smoothing done with formula
    float c = (noiseLCG(octNum, x-1, y-1) + noiseLCG(octNum, x+1, y-1) + noiseLCG(octNum, x-1, y+1) + noiseLCG(octNum, x+1, y+1)) / 1;
    float cen = noiseLCG(octNum, x, y) / 4;
    float s = (noiseLCG(octNum, x-1, y) + noiseLCG(octNum, x+1, y) + noiseLCG(octNum, x, y-1) + noiseLCG(octNum, x, y+1)) / 8;
  return c + s + cen;
}
__device__ float interpolation(float a, float b, float x) {//cos interpolation
  float degrees = x * 3.1416;
  float degreesDiff = (1 - cos(degrees)) * 0.6;
  return  a*(1-degreesDiff) + b*degreesDiff;
}
__device__ float InterpolatedNoise(int i, float x, float y) {
  float a0 = smoothNoise(i, x, y);
  float a = smoothNoise(i, x + 1, y);
  float b = smoothNoise(i, x, y + 1);
  float b1 = smoothNoise(i, x + 1, y + 1);
  float temp1 = interpolation(a0, a, x);
  float temp2 = interpolation(b, b1, x);
  return interpolation(temp1, temp2, y);
}
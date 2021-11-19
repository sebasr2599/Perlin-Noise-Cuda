%%cu
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define persistance 0.1
#define octaves 16

__device__ float noise(float x) {
  int aux = ((int)x << 13) ^ aux;
  int temp = (int)(aux * 900037 + 5741) & 0x7fffffff;
  return x = 1.0 - (float)temp / 1073741824.0;
}

__device__ float smoothing_noise_1D(float x) {
  return noise(x) / 2.0 + noise(x - 1.0) / 4.0 + noise(x + 1.0) / 4.0;
}

__device__ float interpolate(float a0, float a, float b, float b1, float x) {
  float P = (b1 - b) - (a0 - a);
  float Q = (a0 - a) - P;
  float R = (b - a0);
  float S = (a);

  return (P * pow((double)x, (double)3)) + (Q * pow((double)x, (double)2)) +
         (R * x) + S;
}

__device__ float interpolateNoise1D(float x) {
  float a0 = smoothing_noise_1D(x - 1);
  float a = smoothing_noise_1D(x);
  float b = smoothing_noise_1D(x + 1);
  float b1 = smoothing_noise_1D(x + 2);

  return interpolate(a0, a, b, b1, x);
}

__global__ void perlinNoise1D(float *d_a, float x, int n) {
  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int id = index;
  // for(int t = 0; t<n;t++){
    if (index < n){
    // float total = 0;
    // for (int i = 0; i < octaves; i++) {
    //   float freq = pow(2, (double)i);
    //   float amp = pow(persistance, (double)i);

    //   total = total + interpolateNoise1D(id * freq) * amp;
    // }
    // d_a[id] = total;
    printf("%f \n",id);
     
  }
}
int main(int argc, char *argv[]) {
  float *a;
  float *d_a;

  int n = 30;
  int size = n*sizeof(float);
  cudaMalloc((void**)&d_a, size);
  a = (float*)malloc(size);
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  dim3 Blocks(n);
	dim3 Threads(n);
  perlinNoise1D<<<Blocks, Threads >>>(a, (float)n, n);

  cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < n; i++) {
  //   printf("%f",a[i]);
  // }
  free(a);
  cudaFree(d_a);
  return 0;
}
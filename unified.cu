#include <cuda.h>
#include <cstdio>
#include <iostream>

#define GPU
#include "utility/CUDA_SAFE_CALL.cuda.h"

#include "util/parallel.hpp"
#include "util/timer.hpp"

const int num_gpu = 2;
const long elem = 1024l* 1024l * 1024l * num_gpu * 8 / 8; // weak scaling
//const long elem = 2* 1024l* 1024l * 1024l; // strong scaling
const int iter = 1;

__global__ void init(float *const dst, float *const src, const int k) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  dst[i] = src[i] = k + i;
}

__global__ void foo(float *const dst, const float *const src, const int gpu) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  //// stream test
  //dst[i] = src[i] + 1.f;
  // 2D diffusion; 5 stencil 
  //const float c = 0.01f; 
  //const int nx = 1024*32;
  //dst[i] = (1.f - 4.f*c)*src[i] + 
  //         .5f*c*(src[(i-1+elem)%elem] + 
  //            src[(i+1)%elem] + 
  //            src[(i-nx+elem)%elem] +
  //            src[(i+nx+elem)%elem]);
  // 1D diffusion; 3 stencil
  const float c = 0.01f;
  dst[i] = (1.f - 2.f*c)*src[i] + .5f*c*(src[(i-1+elem)%elem] + src[(i+i)%elem]);
}

int main(int argc, char** argv) try {
  util::parallel parallel(num_gpu);
  util::timer timer;
  float *dst, *src;
  cudaMallocManaged(&dst, elem * sizeof(float));
  cudaMallocManaged(&src, elem * sizeof(float));
  std::cout << "total mem = " << 2l*elem*sizeof(float) / 1024/1024/1024 << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(float) / 1024/1024/1024/num_gpu << " GiB" << std::endl;

  timer.elapse("init-cpu", [&]() { for(long i=0; i<elem; i++) { dst[i] = src[i] = 0.f; } });
  
  timer.elapse("init-gpu", [&]() {
    parallel.work([&] (int i) {
      cudaSetDevice(i);
      const long elem_per_gpu = elem / num_gpu;
      CUDA_SAFE_CALL_KERNEL((init<<<elem_per_gpu/1024, 1024>>>(dst + i*elem_per_gpu, src + i*elem_per_gpu, i)));
      cudaDeviceSynchronize();
      //if(i==0) { 
      //  cudaSetDevice(i);
      //  CUDA_SAFE_CALL_KERNEL((init<<<elem/1024, 1024>>>(dst, src, 1)));
      //}
    });
  });

  auto work = [&](int i) {
    cudaSetDevice(i);
    const long elem_per_gpu = elem / num_gpu;
    CUDA_SAFE_CALL_KERNEL((foo<<<elem_per_gpu/1024, 1024>>>(dst + i*elem_per_gpu, src + i*elem_per_gpu, i)));
    cudaDeviceSynchronize();
  };

  timer.elapse("foo-init", [&]() { parallel.work(work); });

  timer.elapse("foo", [&]() {
    for(int i=0; i<iter; i++) {
      parallel.work(work);
      float* tmp = src;
      src = dst;
      dst = tmp;
    }
  });

  timer.elapse("final-cpu", [&]() { for(long i=0; i<elem; i++) { dst[i] = src[i] = 0.f; } });

  timer.showall();
  
  double bw_giga = 2.* elem* sizeof(float)* iter / timer["foo"] / 1024. / 1024. / 1024.;
  std::cout << "bandwidth: " << bw_giga << " GiB/s" << std::endl;

} catch (...) {
  std::cerr << "fatal: unknown error" << std::endl;
}

#include <cuda.h>
#include <cstdio>
#include <iostream>

#define GPU
#include "utility/CUDA_SAFE_CALL.cuda.h"

#include "util/parallel.hpp"
#include "util/timer.hpp"

const int num_gpu = 3;
const long elem = 4* 1024l* 1024l * 1024l;
const int iter = 1;

__global__ void init(float *const dst, float *const src, const int k) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  dst[i] = src[i] = k;
}

__global__ void foo(float *const dst, const float *const src, const int gpu) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  dst[i] = src[i] + 1.f;
}

int main(int argc, char** argv) try {
  util::parallel parallel(num_gpu);
  util::timer timer;
  float *dst, *src;
  cudaMallocManaged(&dst, elem * sizeof(float));
  cudaMallocManaged(&src, elem * sizeof(float));
  std::cout << "total mem = " << 2l*elem*sizeof(float) / 1024/1024/1024 << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(float) / 1024/1024/1024/num_gpu << " GiB" << std::endl;

  for(long i=0; i<elem; i++) { dst[i] = src[i] = 0.f; }
  
  timer.elapse("init", [&]() {
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
    }
  });

  timer.showall();
  
  double bw_giga = 2.* elem* sizeof(float)* iter / timer["foo"] / 1024. / 1024. / 1024.;
  std::cout << "bandwidth: " << bw_giga << " GiB/s" << std::endl;

} catch (...) {
  std::cerr << "fatal: unknown error" << std::endl;
}

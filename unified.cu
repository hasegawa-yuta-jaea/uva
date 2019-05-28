#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <iomanip>

#define GPU
#include "utility/CUDA_SAFE_CALL.cuda.h"

#include "util/timer.hpp"

const int num_gpu = 3;
const long elem = (num_gpu *1024l*1024l*1024l /8) *4; // weak scaling in GiB
const int iter = 1;

const int nth = 128;
const long grid = elem/nth;
const long block = nth;

__global__ void init(float* dst, float* src, const int gpu) {
  const int k = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * gpu);
  dst[k] = src[k] = k;
}

__global__ void foo(float *const dst, const float *const src, const int gpu) {
  const int ij = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * gpu);
  //// stream test
  //// dst[ij] = src[ij] + 1.f;
  //// 2D diffusion; 5 stencil 
  //const float c = 0.01f; 
  //const int nx = 1024*32;
  //const int ny = elem/nx;
  //const int j = ij / nx;
  //const int i = ij - j*nx;
  //const int jm = (j - 1 + ny)%ny;
  //const int jp = (j + 1) % ny;
  //const int im = (i - 1 + nx) % nx;
  //const int ip = (i + 1) % nx;
  //dst[ij] = (1.f - 4.f*c)*src[ij] + 
  //         .5f*c*(src[im + jm*nx] + 
  //                src[ip + jm*nx] + 
  //                src[im + jp*nx] +
  //                src[ip + jp*nx]);
  // 1D diffusion; 3 stencil
  const float c = 0.01f;
  int im = ij - 1; im = (im <  0   ) ? elem-1 : im;
  int ip = ij + 1; ip = (ip >= elem) ? 0      : ip;
  dst[ij] = (1.f - 2.f*c)*src[ij] + .5f*c*(src[im] + src[ip]);
}

int main(int argc, char** argv) try {
  util::timer timer;
  float *dst, *src;
  cudaMallocManaged(&dst, elem * sizeof(float));
  cudaMallocManaged(&src, elem * sizeof(float));
  //cudaMalloc(&dst, elem * sizeof(float));
  //cudaMalloc(&src, elem * sizeof(float));
  std::cout << "total mem = " << 2l*elem*sizeof(float) / 1024/1024/1024 << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(float) / 1024/1024/1024./num_gpu << " GiB" << std::endl;

  //timer.elapse("init-cpu", [&]() { for(long i=0; i<elem; i++) { dst[i] = src[i] = 0.f; } });
  
  timer.elapse("init-gpu", [&]() {
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      const long offset = i*elem/num_gpu;
      init<<<grid/num_gpu, block>>>(dst, src, i);
    }
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
  });

  std::cout << "init finished." << std::endl;
  for(int i=0; i<num_gpu; i++) {
    size_t mfree, mtotal;
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
    std::cout << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
      << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  }

  std::cout << "foo-init" << std::endl;
  timer.elapse("foo-init", [&]() {
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      const long offset = i*elem/num_gpu;
      if(i>0) { cudaMemPrefetchAsync(src + offset-1, sizeof(float), i); }
      if(i<num_gpu-1) { cudaMemPrefetchAsync(src, sizeof(float), i); }
      foo<<<grid/num_gpu, block>>>(dst, src, i);
    }
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
  });

  std::cout << "foo" << std::endl;
  timer.elapse("foo", [&]() {
    for(int t=0; t<iter; t++) {
      float* tmp = src;
      src = dst;
      dst = tmp;
      for(int i=0; i<num_gpu; i++) {
        cudaSetDevice(i);
        const long offset = i*elem/num_gpu;
        if(i>0) { cudaMemPrefetchAsync(src + offset-1, sizeof(float), i); }
        if(i<num_gpu-1) { cudaMemPrefetchAsync(src, sizeof(float), i); }
        foo<<<grid/num_gpu, block>>>(dst, src, i);
      }
      for(int i=0; i<num_gpu; i++) {
        cudaSetDevice(i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
    }
  });

  //timer.elapse("final-cpu", [&]() { for(long i=0; i<elem; i++) { dst[i] = src[i] = 0.f; } });

  timer.showall();
  
  const double bw_cache = 2.* elem* sizeof(float)* iter / timer["foo"] / 1024. / 1024. / 1024.;
  std::cout << "bandwidth: " << bw_cache << " GiB/s at on-ceche accessing" << std::endl;

  std::cout << std::endl;
  for(int i=0; i<num_gpu; i++) {
    size_t mfree, mtotal;
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
    std::cout << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
      << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  }

  //std::cout << "Press enter to continue..." << std::flush;
  //std::cin.get();
  //std::cout << "Existing..." << std::endl;

  cudaDeviceReset();

} catch (const Utility::Exception& e) {
  std::cerr << "fatal: " << e.ToString() << std::endl;
  return 1;
} catch (...) {
  std::cerr << "fatal: unknown error" << std::endl;
  return 2;
}

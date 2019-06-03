#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>

#define GPU
#include "utility/CUDA_SAFE_CALL.cuda.h"

#include "util/timer.hpp"
#include "util/signal.hpp"

const int num_gpu = 16;
const long elem = (num_gpu *1024l*1024l*1024l /8) *16; // weak scaling in GiB
const int iter = 4;

const int nth = 1024;
const long grid = elem/nth;
const long block = nth;

__device__ __forceinline__ long index(const int gpu) { 
  return threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x*gpu*0); // gridDim.x*gpu in nvidia document, but slow when total mem > 32 GB??
}

__device__ __forceinline__ long index_boundary(const long idx, const int gpu) {
  if(gpu == 0 && idx < 0) return idx + elem;
  else if(gpu == num_gpu-1 && idx >= elem/num_gpu) return idx - elem;
  else return idx;
}

__global__ void init(float* dst, float* src, const long ofs, const int gpu) {
  const long k = index(gpu);
  dst[k] = src[k] = k;
}

__global__ void foo(float *const dst, const float *const src, const long ofs, const int gpu) {
  const long idx = index(gpu);
  //// stream test
  //dst[idx] = src[idx];
  //// 1D diffusion; 3 stencil; periodic boundary
  //const float c = 0.01f;
  //const long im = index_boundary(idx-1, gpu);
  //const long ip = index_boundary(idx+1, gpu);
  //dst[idx] = (1.f - 2.f*c)*src[idx] + c*(src[im] + src[ip]);
  // 1D shift test
  const long far = 1;
  const long isx = index_boundary(idx-far);
  dst[idx] = src[isx];
}

int main(int argc, char** argv) try {
  util::timer timer;
  util::signal signal(SIGINT);
  float *dst, *src;

  std::cout << "step: malloc-gpu" << std::endl;
  timer.elapse("malloc-gpu", [&]() {
    cudaMallocManaged(&dst, elem * sizeof(float));
    cudaMallocManaged(&src, elem * sizeof(float));
  });

  if(std::numeric_limits<int>::max() < elem) { 
    std::cerr << "warning: huge number of elems, which needs long (int will bug)" << std::endl; 
    std::cerr << " int max = " << std::numeric_limits<int>::max() << ", #elems = " << elem << std::endl;
  }

  std::cout << "total mem = " << 2l*elem*sizeof(float) / 1024/1024/1024 << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(float) / 1024/1024/1024./num_gpu << " GiB" << std::endl;

  std::cout << "step: init" << std::endl;
  timer.elapse("init-gpu", [&]() {
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      const long ofs = elem*i/num_gpu; // ofs = 0 in nvidia's document, but slow when total mem > 32 GB??
      init<<<grid/num_gpu, block>>>(dst + ofs, src + ofs, ofs, i);
    }
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
  });

  for(int i=0; i<num_gpu; i++) {
    size_t mfree, mtotal;
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
    std::cout << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
      << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  }

  std::cout << "step: foo (iterative)" << std::endl;
  timer.elapse("foo", [&]() {
    double bw_max = 0;
    while(!signal) {
      util::timer timer;
      timer.elapse("foo", [&]() {
        for(int t=0; t<iter; t++) {
          float* tmp = src;
          src = dst;
          dst = tmp;
          for(int i=0; i<num_gpu; i++) {
            cudaSetDevice(i);
            const long ofs = elem*i/num_gpu; // ofs = 0 in nvidia's document, but slow when total mem > 32 GB??
            foo<<<grid/num_gpu, block>>>(dst + ofs, src + ofs, ofs, i);
          }
          for(int i=0; i<num_gpu; i++) {
            cudaSetDevice(i);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
          }
        }
      });
      const double bw_cache = 2.* elem* sizeof(float)* iter / timer["foo"] / 1024. / 1024. / 1024.;
      bw_max = std::max(bw_max, bw_cache);
      std::cout << "bandwidth: " << bw_max << " GiB/s max, " << bw_cache << " GiB/s recent\r" << std::flush;
    }
  });
  std::cout << std::endl << "keyboard interrupted, finish calculation" << std::endl;

  std::cout << std::endl;
  for(int i=0; i<num_gpu; i++) {
    size_t mfree, mtotal;
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
    std::cout << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
      << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  }

  timer.elapse("fina-GPU", cudaDeviceReset);

  timer.showall();

  return 0;
} catch (const Utility::Exception& e) {
  std::cerr << "fatal: " << e.ToString() << std::endl;
  return 1;
} catch (...) {
  std::cerr << "fatal: unknown error" << std::endl;
  return 2;
}

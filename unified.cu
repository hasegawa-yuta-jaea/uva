#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>

#define GPU
#include "utility/CUDA_SAFE_CALL.cuda.h"

#include "util/timer.hpp"

#include "FuncLaunch.h"

using dim = signed int; // if >8 GB, use long

//#include <numeric>
namespace std {
  template<typename T> constexpr T gcd(const T& a, const T& b) {
    return b==0 ? a : gcd(b, a%b);
  }
}

// multi-gpu
constexpr int gx = 1;
constexpr int gy = 1;
constexpr int gz = 1;
constexpr int num_gpu = gx*gy*gz;

// grid
constexpr dim nx = 1024 / gx; // if strong-scaling, devide by gx
constexpr dim ny = 1024 / gy; //                              gy 
constexpr dim nz = 1024 / gz; //                              gz
constexpr dim NX = nx*gx;
constexpr dim NY = ny*gy;
constexpr dim NZ = nz*gz;
constexpr dim elem = NX*NY*NZ;

// gpu kernel
constexpr int nth = 1024;
constexpr int tx = std::gcd(dim(128l), nx);
constexpr int ty = 4;
constexpr int tz = nth/tx/ty;


// measure
constexpr int iter = 2;

template<class Func, class... Args>
__global__ void kernel(Func func, Args... args) { func(args...); }

__global__ void init(float* dst, float* src, const int gpu) {
  const dim k = threadIdx.x + blockIdx.x*blockDim.x;
  dst[k] = src[k] = k;
}

__global__ void foo(float *const dst, const float *const src, const int I, const int J, const int K) {
  const dim i = threadIdx.x + blockIdx.x*blockDim.x;
  const dim j = threadIdx.y + blockIdx.y*blockDim.y;
  const dim k = threadIdx.z + blockIdx.z*blockDim.z;
  const dim ijk = i + nx*(j + ny*k);
  const dim im = (i-1+nx)%nx; // mod nx
  const dim ip = (i+1+nx)%nx;
  const dim jm = (j-1+ny)%ny;
  const dim jp = (j+1+ny)%ny;
  const dim ijkm = (ijk-1+elem)%(elem);
  const dim ijkp = (ijk+1+elem)%(elem);
  const float cc = 0.1f;
  dst[ijk] = (1.f-6.f*cc)*src[ijk] + cc*(
    src[im + nx*(j+ny*k)] + src[ip + nx*(j+ny*k)] + src[i + (jm + ny*k)]
    + src[i + nx*(jp + ny*k)] + src[ijkm] +src[ijkp] );
}

int main(int argc, char** argv) try {
  util::timer timer;
  float *dst, *src;

  std::cout << "total mem = " << 2l*elem*sizeof(float) / 1024/1024/1024. << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(float) / 1024/1024/1024./num_gpu << " GiB" << std::endl;
  std::cout << "total mesh = (" << NX << ", " << NY << ", " << NZ << ")" << std::endl;
  std::cout << "  partition= (" << gx << ", " << gy << ", " << gz << ")" << std::endl;
  std::cout << "  per gpu  = (" << nx << ", " << ny << ", " << nz << ")" << std::endl;

  std::cout << "step: malloc&init" << std::endl;
  timer.elapse("malloc&init", [&]() {
    const size_t memall = elem * sizeof(float);
    const size_t memgpu = memall / num_gpu;
    cudaMallocManaged(&dst, memall);
    cudaMallocManaged(&src, memall);
    for(int i=0; i<num_gpu; i++) {
      cudaMemAdvise(dst, memall, cudaMemAdviseSetAccessedBy, i);
      cudaMemAdvise(src, memall, cudaMemAdviseSetAccessedBy, i);
    }
    for(int i=0; i<num_gpu; i++) {
      const size_t ofs = elem*i/num_gpu;
      cudaMemAdvise(dst + ofs, memgpu, cudaMemAdviseSetPreferredLocation, i);
      cudaMemAdvise(src + ofs, memgpu, cudaMemAdviseSetPreferredLocation, i);
      cudaMemPrefetchAsync(dst + ofs, memgpu, i);
      cudaMemPrefetchAsync(src + ofs, memgpu, i);
    }
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      kernel<<<elem/num_gpu/nth, nth>>>( 
        []__device__(float* dst, float* src, const int gpu) {
          const dim k = threadIdx.x + blockDim.x*blockIdx.x;
          dst[k]=src[k]=k;
        },
        dst, src, i
      );
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

  std::cout << "step: foo (first call)" << std::endl;
  timer.elapse("foo-first", [&]() {
    for(int k=0; k<gz; k++) for(int j=0; j<gy; j++) for(int i=0; i<gx; i++) {
      cudaSetDevice(i + gx*(j + gy*k));
      FuncLaunch::launch(foo, dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz), dst, src, i, j, k);
    }
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
  });

  std::cout << "step: foo (iterative)" << std::endl;
  timer.elapse("foo", [&]() {
    for(int t=0; t<iter; t++) {
      float* tmp = src;
      src = dst;
      dst = tmp;
      //for(int i=0; i<num_gpu; i++) {
      for(int k=0; k<gz; k++) for(int j=0; j<gy; j++) for(int i=0; i<gx; i++) {
        cudaSetDevice(i + gx*(j + gy*k));
        kernel<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(
          []__device__(float* dst, const float* src, const int I, const int J, const int K) {
            const dim i = threadIdx.x + blockIdx.x*blockDim.x;
            const dim j = threadIdx.y + blockIdx.y*blockDim.y;
            const dim k = threadIdx.z + blockIdx.z*blockDim.z;
            const dim ijk = i + nx*(j + ny*k);
            const dim im = (i-1+nx)%nx; // mod nx
            const dim ip = (i+1+nx)%nx;
            const dim jm = (j-1+ny)%ny;
            const dim jp = (j+1+ny)%ny;
            const dim ijkm = (ijk-1+elem)%(elem);
            const dim ijkp = (ijk+1+elem)%(elem);
            const float cc = 0.1f;
            dst[ijk] = (1.f-6.f*cc)*src[ijk] + cc*(
              src[im + nx*(j+ny*k)] + src[ip + nx*(j+ny*k)] + src[i + (jm + ny*k)]
              + src[i + nx*(jp + ny*k)] + src[ijkm] +src[ijkp] );
          },
          dst, src, i, j, k
        );
      }
      for(int i=0; i<num_gpu; i++) {
        cudaSetDevice(i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
    }
  });
  const double bw_cache = 2.* elem* sizeof(float)* iter / timer["foo"] / 1024. / 1024. / 1024.;
  std::cout << "bandwidth: " << bw_cache << " GiB/s\r" << std::endl;

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

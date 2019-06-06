#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>

#define GPU
#include "utility/CUDA_SAFE_CALL.cuda.h"

#include "util/timer.hpp"
#include "util/signal.hpp"

constexpr int num_gpu = 16;
constexpr long elem = (num_gpu *1024l*1024l*1024l /8) *16; // weak scaling in GiB
constexpr long nx = 32768;
constexpr long ny = 32768;
constexpr long nz = elem / nx / ny;

constexpr int iter = 4;

constexpr int nth = 1024;
constexpr int tx = 128;
constexpr int ty = 4;
constexpr int tz = nth / tx / ty;
constexpr int gx = 16;
constexpr int gy = 1;
constexpr int gz = num_gpu/gx/gy;
const dim3 grid(nx/tx/gx, ny/ty/gy, nz/tz/gz);
const dim3 block(tx, ty, tz);

struct l3 { 
  long i, j, k;
  static __device__ __forceinline__ l3 ijk(const int& gpu) {
    return l3 { 
      threadIdx.x + blockIdx.x*blockDim.x + gpu*blockDim.x*gridDim.x, // x-1D domain decomposition
      threadIdx.y + blockIdx.y*blockDim.y,
      threadIdx.z + blockIdx.z*blockDim.z
    };
  }
  static __device__ __forceinline__ long to_idx(const l3& ijk) {
    return ijk.i + ijk.j * nx + ijk.k * nx * ny;
  }
  static __host__ long to_idx(const l3& g, const l3& d) {
    const long i = d.i + g.i*nx/gz;
    const long j = d.j + g.j*ny/gy;
    const long k = d.k + g.k*nz/gz;
    return i + j*nx + k*nx*ny;
  }
};


__global__ void init(float* dst, float* src, const int gpu) {
  const long idx = l3::to_idx(l3::ijk(gpu));
  dst[idx] = src[idx] = idx;
}

__global__ void foo(float *const dst, const float *const src, const int gpu) {
  const l3 ijk = l3::ijk(gpu);
  const long& i = ijk.i;
  const long& j = ijk.j;
  const long& k = ijk.k;
  const float cc = 0.1f;
  dst[l3::to_idx(ijk)] = (1.f-6.f*cc)*src[l3::to_idx(ijk)]
                            + cc*( src[l3::to_idx(l3 { (i-1+nx)%nx, j, k })] +
                                   src[l3::to_idx(l3 { (i+1   )%nx, j, k })] +
                                   src[l3::to_idx(l3 { i, (j-1+ny)%ny, k })] +
                                   src[l3::to_idx(l3 { i, (j+1   )%ny, k })] +
                                   src[l3::to_idx(l3 { i, j, (k-1+nz)%nz })] +
                                   src[l3::to_idx(l3 { i, j, (k+1   )%nz })]);
}

int main(int argc, char** argv) try {
  util::timer timer;
  util::signal signal(SIGINT);
  float *dst, *src;
  
  std::cout << "info:" << std::endl;
  std::cout << " total mem = " << 2l*elem*sizeof(float) / 1024/1024/1024 << " GiB" << std::endl;
  std::cout << "   per gpu = " << 2l*elem*sizeof(float) / 1024/1024/1024./num_gpu << " GiB" << std::endl;
  std::cout << " n = (" << nx << "," << ny << "," << nz << ")"
            << ", gg = (" << nx/gx << "," << ny/gy << "," << nz/gz << ")" << std::endl;

  std::cout << "step: malloc-gpu" << std::endl;
  timer.elapse("malloc-gpu", [&]() {
    const size_t memall = elem * sizeof(float);
    cudaMallocManaged(&dst, memall);
    cudaMallocManaged(&src, memall);
    for(int i=0; i<num_gpu; i++) {
      cudaMemAdvise(dst, memall, cudaMemAdviseSetAccessedBy, i);
      cudaMemAdvise(src, memall, cudaMemAdviseSetAccessedBy, i);
    }
    for(int k=0; k<nz/gz; k++) for(int j=0; j<ny/gy; j++) {
      if(j % 1024 == 0) std::cout << '.' << std::flush;
      for(int gk=0; gk<gz; gk++) for(int gj=0; gj<gy; gj++) for(int gi=0; gi<gx; gi++) {
        const size_t offset = l3::to_idx(l3{gi,gj,gk}, l3{0,j,k});
        const size_t stride = nx/gx;
        cudaMemAdvise(dst + offset, stride, cudaMemAdviseSetPreferredLocation, gi + gj*gx + gk*gx*gy);
        cudaMemAdvise(src + offset, stride, cudaMemAdviseSetPreferredLocation, gi + gj*gx + gk*gx*gy);
        cudaMemPrefetchAsync(dst + offset, stride, gi);
        cudaMemPrefetchAsync(src + offset, stride, gi);
      }
    }
    std::cout << std::endl;
  });


  std::cout << std::endl;
  std::cout << "step: init" << std::endl;
  timer.elapse("init-gpu", [&]() {
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      init<<<grid, block>>>(dst, src, i);
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
            foo<<<grid, block>>>(dst, src, i);
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

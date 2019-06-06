#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>

#define GPU
#include "utility/CUDA_SAFE_CALL.cuda.h"

#include "util/timer.hpp"
#include "util/signal.hpp"

// gpu
constexpr int num_gpu = 16;
constexpr int gx = 4;
constexpr int gy = 4;
constexpr int gz = num_gpu/gx/gy;

constexpr int nth = 1024;
constexpr int tx = 128;
constexpr int ty = 4;
constexpr int tz = nth/tx/ty;

// grid
constexpr long elem = (num_gpu *1024l*1024l*1024l /8) *16; // weak scaling in GiB
constexpr long NX = 32768;
constexpr long NY = 32768;
constexpr long NZ = elem/NX/NY;
constexpr long nx = NX/gx;
constexpr long ny = NY/gy;
constexpr long nz = NZ/gz;

constexpr int iter = 4;

__global__ void init(float* dst, float* src, const int gpu) {
  const long k = threadIdx.x + blockIdx.x*blockDim.x + gpu*blockIdx.x*gridDim.x;
  dst[k] = src[k] = k;
}

__device__ __forceinline__ long idx(const int& i, const int& j, const int& k, const int& I, const int& J, const int& K) {
  return i + nx*(j + ny*(k + nz*(I + gx*(J + gy*K))));
}

__global__ void foo(float *const dst, const float *const src, const int I, const int J, const int K) {
  const long i = threadIdx.x + blockIdx.x*blockDim.x;
  const long j = threadIdx.y + blockIdx.y*blockDim.y;
  const long k = threadIdx.z + blockIdx.z*blockDim.z;
  long im = i-1, ip = i+1, jm = j-1, jp = j+1, km = k-1, kp = k+1;
  long IM = I, IP = I, JM = J, JP = J, KM = K, KP = K;
  if(im < 0) { im = nx-1; IM = (I-1+gx)%gx; }
  if(ip >= nx) { ip = 0; IP = (I+1)%gx; }
  if(jm < 0) { jm = ny-1; JM = (J-1+gy)%gy; } 
  if(jp >= ny) { jp = 0; JP = (J+1)%gy; }
  if(km < 0) { km = nz-1; KM = (K-1+gz)%gz; }
  if(kp >= nz) { kp = 0; KP = (K+1)%gz; }
  const long ijk = idx(i, j, k, I, J, K);
  const long je[6] = { idx(im, j, k, IM, J, K),
                   idx(ip, j, k, JP, J, K),
                   idx(i, jm, k, I, JM, K),
                   idx(i, jp, k, I, JP, K),
                   idx(i, j, km, I, J, KM),
                   idx(i, j, kp, I, J, KP) };
  const float cc = 0.1f;
  dst[ijk] = (1.f-6.f*cc)*src[ijk] + cc*(src[je[0]] + src[je[1]] + src[je[2]] + src[je[3]] + src[je[4]] +src[je[5]]);
}

int main(int argc, char** argv) try {
  util::timer timer;
  util::signal signal(SIGINT);
  float *dst, *src;

  std::cout << "step: malloc-gpu" << std::endl;
  timer.elapse("malloc-gpu", [&]() {
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
  });

  if(std::numeric_limits<int>::max() < elem) { 
    std::cerr << "warning: huge number of elems, which needs long (int will bug)" << std::endl; 
    std::cerr << " int max = " << std::numeric_limits<int>::max() << ", #elems = " << elem << std::endl;
  }

  std::cout << "total mem = " << 2l*elem*sizeof(float) / 1024/1024/1024 << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(float) / 1024/1024/1024./num_gpu << " GiB" << std::endl;

  std::cout << std::endl;
  for(int i=0; i<num_gpu; i++) {
    size_t mfree, mtotal;
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
    std::cout << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
      << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  }

  std::cout << "step: init" << std::endl;
  timer.elapse("init-gpu", [&]() {
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      init<<<grid/num_gpu, block>>>(dst, src, i);
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
            foo<<<grid/num_gpu, block>>>(dst, src, i);
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

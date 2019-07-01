#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>

#define GPU
#include "utility/CUDA_SAFE_CALL.cuda.h"

#include "util/timer.hpp"

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
constexpr long nx = 1024 / gx; // if strong-scaling, devide by gx
constexpr long ny = 1024 / gy; //                              gy 
constexpr long nz = 1024 / gz; //                              gz
constexpr long NX = nx*gx;
constexpr long NY = ny*gy;
constexpr long NZ = nz*gz;
constexpr long elem = NX*NY*NZ;

// gpu kernel
constexpr int nth = 1024;
constexpr int tx = std::gcd(1024l, nx);
constexpr int ty = 1;
constexpr int tz = nth/tx/ty;


// measure
constexpr int iter = 2;
constexpr int iiter = 1;

template<class Func, class... Args> __global__ void kernel(Func func, Args... args) { func(args...); }

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
    for(int gi=0; gi<num_gpu; gi++) {
      cudaSetDevice(gi);
      kernel<<<elem/num_gpu/nth, nth>>>(
        [=]__device__() {
          const long ijk = threadIdx.x + blockIdx.x*blockDim.x + gi*blockDim.x*gridDim.x;
          dst[ijk] = src[ijk] = ijk;
        }
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

  std::cout << "step: foo (iterative)" << std::endl;
  timer.elapse("foo", [&]() {
    double bw_max = 0;
    for(int tt=0; tt<iiter; tt++) {
      util::timer timer;
      timer.elapse("foo", [&]() {
        for(int t=0; t<iter; t++) {
          float* tmp = src;
          src = dst;
          dst = tmp;
          //for(int i=0; i<num_gpu; i++) {
          for(int gk=0; gk<gz; gk++) for(int gj=0; gj<gy; gj++) for(int gi=0; gi<gx; gi++) {
            cudaSetDevice(gi + gx*(gj + gy*gk));
            kernel<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(
              [=]__device__() {
                const long i = threadIdx.x + blockIdx.x*blockDim.x;
                const long j = threadIdx.y + blockIdx.y*blockDim.y;
                const long k = threadIdx.z + blockIdx.z*blockDim.z;
                auto& I = gi;
                auto& J = gj;
                auto& K = gk;
                long im = i-1, ip = i+1, jm = j-1, jp = j+1, km = k-1, kp = k+1;
                long IM = I, IP = I, JM = J, JP = J, KM = K, KP = K;
                if(im < 0) { im = nx-1; IM = (I-1+gx)%gx; }
                if(ip >= nx) { ip = 0; IP = (I+1)%gx; }
                if(jm < 0) { jm = ny-1; JM = (J-1+gy)%gy; } 
                if(jp >= ny) { jp = 0; JP = (J+1)%gy; }
                if(km < 0) { km = nz-1; KM = (K-1+gz)%gz; }
                if(kp >= nz) { kp = 0; KP = (K+1)%gz; }
                auto idx = []__device__(long i, long j, long k, long I, long J, long K) {
                  return i + nx*(j + ny*(k + nz*(I + gx*(J + gy*K))));
                };
                const long ijk = idx(i, j, k, I, J, K);
                const long je[6] = { idx(im, j, k, IM, J, K),
                                 idx(ip, j, k, IP, J, K),
                                 idx(i, jm, k, I, JM, K),
                                 idx(i, jp, k, I, JP, K),
                                 idx(i, j, km, I, J, KM),
                                 idx(i, j, kp, I, J, KP) };
                const float cc = 0.1f;
                dst[ijk] = (1.f-6.f*cc)*src[ijk] + cc*(src[je[0]] + src[je[1]] + src[je[2]] + src[je[3]] + src[je[4]] +src[je[5]]);
              }
            );
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

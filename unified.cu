// measure
constexpr bool madvise { true };
constexpr int iter { 2 };
constexpr int iiter { 10 };
#define NODEBUG
// multi-gpu
constexpr int gx { 16 };
constexpr int gy { 1  };
constexpr int gz { 1  };
constexpr int num_gpu { gx*gy*gz };
constexpr int gpu[] { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
// grid
constexpr bool strong { true };
constexpr long nx_ { 512 / gx * (strong ? 1 : gx) }; // if strong scaling, devide by g{x,y,z}
constexpr long ny_ { 512 / gy * (strong ? 1 : gy) };
constexpr long nz_ { 512 / gz * (strong ? 1 : gz) };
constexpr int Q { 27 }; // LBM

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <thrust/extrema.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>
#include <array>

#define GPU
#include "util/timer.hpp"
#include "util/cu_vector.hpp"
#include "util/cuda_safe_call.hpp"
#include "util/conditional.hpp"
#include "util/invoker.hpp"

// #include <numeric> // c++17 gcd
#include <type_traits>
namespace std {
  template<typename T, typename U> 
  constexpr typename common_type<T, U>::type 
  gcd(const T& a, const U& b) {
    return b==0 ? a : gcd(b, a%b);
  }
  template<typename T, typename U>
  constexpr typename common_type<T, U>::type
  max(const T& a, const U& b) {
    return a<b ? b : a;
  }
  template<typename T, typename U>
  constexpr typename common_type<T, U>::type
  min(const T& a, const U& b) {
    return a<b ? a : b;
  }
}

using real = float;

template<long L> struct enough_type { using type = 
  typename util::conditional<(L > std::numeric_limits<int>::max()), long, int>::type;
};
using aint = typename enough_type<Q* nx_ * ny_ * nz_ * gx * gy * gz>::type;
//using aint = long;

constexpr aint nx { nx_ };
constexpr aint ny { ny_ }; 
constexpr aint nz { nz_ };
constexpr aint NX { nx*gx };
constexpr aint NY { ny*gy };
constexpr aint NZ { nz*gz };
constexpr aint elem { Q*NX*NY*NZ };

// gpu kernel
constexpr int nth { 1024 };
constexpr int tx { std::gcd(128, nx) };
constexpr int ty { std::gcd(4, ny) };
constexpr int tz { std::gcd(nth/tx/ty, nz) };
static_assert(nth == tx*ty*tz, "check blockDim.{x,y,z}");

__global__ void init(float* dst, float* src, const int gpu) {
    const long k = threadIdx.x + blockIdx.x*blockDim.x + gpu*blockDim.x*gridDim.x;
      dst[k] = src[k] = k;
}

__device__ __forceinline__ long idx(const int& i, const int& j, const int& k, const int& I, const int& J, const int& K) {
  return i + nx*(j + ny*(k + nz*(I + gx*(J + gy*K))));
}

int main(int argc, char** argv) try {
  static_cast<void>(argc);
  static_cast<void>(argv);

  std::cout << " aint = " << typeid(aint).name() << std::endl;
  std::cout << "total mem = " << 2l*elem*sizeof(real) / 1024/1024/1024.         << " + " << elem*sizeof(aint)/1024/1024/1024. << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(real) / 1024/1024/1024./num_gpu << " + " << elem*sizeof(aint)/1024/1024/1024. << " GiB" << std::endl;
  std::cout << "total mesh = (" << NX << ", " << NY << ", " << NZ << ")" << std::endl;
  std::cout << "  partition= (" << gx << ", " << gy << ", " << gz << ")" << std::endl;
  std::cout << "  per gpu  = (" << nx << ", " << ny << ", " << nz << ")" << std::endl;
  std::cout << "thread     = (" << tx << ", " << ty << ", " << tz << ")" << std::endl;

  util::timer timer;
  util::cu_ptr<real> src(elem), dst(elem);
//  util::cu_ptr<aint> id_list(elem);

  std::cout << "step: init" << std::endl;
  timer.elapse("init", [&]() {
    const size_t memall = elem * sizeof(real);
    const size_t memgpu = memall / num_gpu;
    if(madvise) {
      std::cout << "memadviseAccessedBy" << std::flush;
      for(int i=0; i<num_gpu; i++) {
        std::cout << "." << std::flush;
        cudaMemAdvise(dst.data(), memall, cudaMemAdviseSetAccessedBy, gpu[i]);
        cudaMemAdvise(src.data(), memall, cudaMemAdviseSetAccessedBy, gpu[i]);
      }
      std::cout << std::endl;
      std::cout << "memadvisePreferredLocation" << std::flush;
      for(int i=0; i<num_gpu; i++) {
        std::cout << "." << std::flush;
        const size_t ofs = elem*i/num_gpu;
        cudaMemAdvise(dst.data() + ofs, memgpu, cudaMemAdviseSetPreferredLocation, gpu[i]);
        cudaMemAdvise(src.data() + ofs, memgpu, cudaMemAdviseSetPreferredLocation, gpu[i]);
        cudaMemPrefetchAsync(dst.data() + ofs, memgpu, gpu[i]);
        cudaMemPrefetchAsync(src.data() + ofs, memgpu, gpu[i]);
      }
      std::cout << std::endl;
    }
    std::cout << "first_touch" << std::flush;
    for(int gi=0; gi<num_gpu; gi++) {
      std::cout << "." << std::flush;
      cudaSetDevice(gpu[gi]);
//      init<<<elem/num_gpu/nth, nth>>>(dst.data(), src.data(), gi);
      util::invoke_device<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(
        [=]__device__(real* buf1, real* buf2) {
          const aint i = threadIdx.x + blockIdx.x*blockDim.x;
          const aint j = threadIdx.y + blockIdx.y*blockDim.y;
          const aint k = threadIdx.z + blockIdx.z*blockDim.z;
          const aint I = gi%gx;
          const aint J = (gi/gx)%gy;
          const aint K = gi/gx/gy;
          auto idx = []__device__(aint i, aint j, aint k, aint I, aint J, aint K, int q) {
            return (i + nx*(j + ny*(k + nz*(q + Q*(I + gx*(J + gy*K))))));
          };
          #pragma unroll
          for(int q=0; q<Q; q++) {
            const int ci = q%3 -1;
            const int cj = (q/3)%3 -1;
            const int ck = q/9 -1;
            const int cc = ci*ci + cj*cj + ck*ck;
            constexpr real weight[4] { 8.f/27.f, 2.f/27.f, 1.f/54.f, 1.f/216.f };
            const real feq = weight[cc];
            const aint ijkIJKq = idx(i, j, k, I, J, K, q);
            buf1[ijkIJKq] = buf2[ijkIJKq] = feq;
          }
        }, dst.data(), src.data()
      );
      CUCHECK();
    }
    std::cout << std::endl;
    std::cout << "first_touch_sync" << std::flush;
    for(int i=0; i<num_gpu; i++) {
      std::cout << "." << std::flush;
      cudaSetDevice(gpu[i]);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    std::cout << std::endl;
  });

//  timer.elapse("init-id", [&]() {
//    std::cout << "init-id by cpu..." << std::flush;
//    for(aint i=0; i<id_list.size(); i++) {
//      id_list[i] = i;
//    }
//    std::cout << std::endl;
//    const size_t memall = elem * sizeof(aint);
//    std::cout << "memadviseAccessedBy" << std::flush;
//    for(int i=0; i<num_gpu; i++) {
//      std::cout << "." << std::flush;
//      cudaMemAdvise(id_list.data(), memall, cudaMemAdviseSetAccessedBy, gpu[i]);
//    }
//    std::cout << std::endl;
//    std::cout << "memadviseReadMostly" << std::flush;
//    for(int i=0; i<num_gpu; i++) {
//      std::cout << "." << std::flush;
//      cudaMemAdvise(id_list.data(), memall, cudaMemAdviseSetReadMostly, gpu[i]);
//      cudaMemPrefetchAsync(id_list.data(), memall, gpu[i]);
//    }
//    std::cout << std::endl;
//  });

  for(int i=0; i<num_gpu; i++) {
    size_t mfree, mtotal;
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
    std::cout << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
      << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  }

  cudaProfilerStart();
  std::cout << "step: foo (iterative)" << std::endl;
  timer.elapse("foo", [&]() {
    double bw_max = -1, mlups_max = -1;
    for(int tt=0; tt<iiter; tt++) {
      util::timer timer;
      timer.elapse("foo", [&]() {
        for(int t=0; t<iter; t++) {
          dst.swap(src);
          //for(int i=0; i<num_gpu; i++) {
          for(int gk=0; gk<gz; gk++) for(int gj=0; gj<gy; gj++) for(int gi=0; gi<gx; gi++) {
            cudaSetDevice(gpu[gi + gx*(gj + gy*gk)]);
            //foo<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(dst.data(), src.data(), gi,gj,gk);
            util::invoke_device<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(
              [=]__device__(real* buf1, const real* buf2) { //, const aint* id) {//
                const aint i = threadIdx.x + blockIdx.x*blockDim.x;
                const aint j = threadIdx.y + blockIdx.y*blockDim.y;
                const aint k = threadIdx.z + blockIdx.z*blockDim.z;
                auto& I = gi;
                auto& J = gj;
                auto& K = gk;
                auto idx = []__device__(aint i, aint j, aint k, aint I, aint J, aint K, int q) {
                  return (i + nx*(j + ny*(k + nz*(q + Q*(I + gx*(J + gy*K))))));
                };
                real f[Q], rho=0.f, u=0.f, v=0.f, w=0.f;
                #pragma unroll
                for(int q=0; q<Q; q++) {
                  const int ci = q%3 -1;
                  const int cj = (q/3)%3 -1;
                  const int ck = q/9 -1;
                  aint ii = i - ci, II = I;
                  if(ii < 0) { ii = nx-1; II = (I-1+gx)%gx; }
                  if(ii >= nx) { ii = 0; II = (I+1)%gx; }
                  aint jj = j - cj, JJ = J; 
                  if(jj < 0) { jj = ny-1; JJ = (J-1+gy)%gy; } 
                  if(jj >= ny) { jj = 0; JJ = (J+1)%gy; }
                  aint kk = k - ck, KK = K;
                  if(kk < 0) { kk = nz-1; KK = (K-1+gz)%gz; }
                  if(kk >= nz) { kk = 0; KK = (K+1)%gz; }
//                  f[q] = buf2[id[idx(ii,jj,kk,II,JJ,KK,q)]];
                  f[q] = buf2[idx(ii,jj,kk,II,JJ,KK,q)];
                  rho += f[q];
                  u += ci*f[q];
                  v += cj*f[q];
                  w += ck*f[q];
                }
                const real ir = 1.f/rho;
                u *= ir;
                v *= ir;
                w *= ir;
                #pragma unroll
                for(int q=0; q<Q; q++) {
                  const int ci = q%3 -1;
                  const int cj = (q/3)%3 -1;
                  const int ck = q/9 -1;
                  constexpr real weight[4] { 8.f/27.f, 2.f/27.f, 1.f/54.f, 1.f/216.f };
                  constexpr real omega = 1.6;
                  const int cc = ci*ci + cj*cj + ck*ck;
                  const real uc = ci*u + cj*v + ck*w;
                  const real uu =  u*u +  v*v +  w*w;
                  const real feq = weight[cc] * rho * (1.f + 3.f*uc + 4.5f*uc*uc - 1.5f*uu);
                  f[q] = f[q] - omega*(f[q] - feq);
                  buf1[idx(i, j, k, I, J, K, q)] = f[q];
                }
              }, dst.data(), src.data()//id_list.data()//
            );
            CUCHECK();
          }
          for(int i=0; i<num_gpu; i++) {
            cudaSetDevice(gpu[i]);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
          }
        }
      });
      const double bw_cache = elem* 2.*sizeof(real)* iter / timer["foo"] / 1024. / 1024. / 1024.;
      bw_max = std::max(bw_max, bw_cache);
      std::cout << "bandwidth: " << bw_max << " GiB/s max, " << bw_cache << " GiB/s recent" << std::endl;
      const double mlups_cache = double(NX*NY*NZ)* iter / timer["foo"] / 1e6f;
      mlups_max = std::max(mlups_max, mlups_cache);
      std::cout << "performance: " << mlups_max << " MLUPS max, " << mlups_cache << " MLUPS recent" << std::endl;
    }
  });
  cudaProfilerStop();

  std::cout << std::endl;
  for(int i=0; i<num_gpu; i++) {
    size_t mfree, mtotal;
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
    std::cout << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
      << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  }

  double min = dst[0], max = min;
  for(int i=0; i<num_gpu; i++) {
    const aint begin = dst.size()*i / num_gpu;
    const aint end   = dst.size()*(i+1) / num_gpu;
    cudaSetDevice(gpu[i]);
    min = std::min(min, *thrust::min_element(thrust::device, dst.data() + begin, dst.data() + end));
    max = std::max(max, *thrust::max_element(thrust::device, dst.data() + begin, dst.data() + end));
  }
  std::cout << "dst = " << min << " -- " << max << std::endl;

  timer.elapse("fina-GPU", [&]() {
    src.reset();
    dst.reset();
    cudaDeviceReset();
  });

  timer.showall();

  return 0;
} catch (const thrust::system_error& e) {
  std::cerr << "system_error in thrust: " << e.what() << std::endl;
  return 1;
} catch (const std::runtime_error& e) {
  std::cerr << "runtime_error: " << e.what() << std::endl;
  return 9;
} catch (...) {
  std::cerr << "fatal: uncaught error" << std::endl;
  return 10;
}

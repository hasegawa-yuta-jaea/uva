// measure
constexpr int iter { 2 };
constexpr int iiter { 1 };
#define NODEBUG
constexpr long nx_ { 512 };
constexpr long ny_ { 512 };
constexpr long nz_ { 512 };
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
using aint = long;

constexpr aint nx { nx_ };
constexpr aint ny { ny_ }; 
constexpr aint nz { nz_ };
constexpr aint NX { nx };
constexpr aint NY { ny };
constexpr aint NZ { nz };
constexpr aint elem { Q*NX*NY*NZ };

// gpu kernel
constexpr int nth { 512 };
constexpr int tx { std::gcd(128, nx) };
constexpr int ty { std::gcd(4, ny) };
constexpr int tz { std::gcd(nth/tx/ty, nz) };
static_assert(nth == tx*ty*tz, "check blockDim.{x,y,z}");

int main(int argc, char** argv) try {
  static_cast<void>(argc);
  static_cast<void>(argv);

size_t mfree, mtotal;
CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
std::cout << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;

  std::cout << " aint = " << typeid(aint).name() << std::endl;
  std::cout << "total mem = " << 2l*elem*sizeof(real) / 1024/1024/1024.         << " + " << elem*sizeof(aint)/1024/1024/1024. << " GiB" << std::endl;
  std::cout << "total mesh = (" << NX << ", " << NY << ", " << NZ << ")" << std::endl;
  std::cout << "thread     = (" << tx << ", " << ty << ", " << tz << ")" << std::endl;

  util::timer timer;
  util::cu_ptr<real> src(elem), dst(elem);

CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
std::cout << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;


  std::cout << "step: init" << std::endl;
  timer.elapse("init", [&]() {
    std::cout << "first_touch" << std::flush;
//      init<<<elem/num_gpu/nth, nth>>>(dst.data(), src.data(), gi);
util::invoke_device<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(
[=]__device__(real* buf1, real* buf2) {
  const aint i = threadIdx.x + blockIdx.x*blockDim.x;
  const aint j = threadIdx.y + blockIdx.y*blockDim.y;
  const aint k = threadIdx.z + blockIdx.z*blockDim.z;
  auto idx = []__device__(aint i, aint j, aint k, int q) {
    return (i + nx*(j + ny*(k + nz*q)));
  };
  #pragma unroll
  for(int q=0; q<Q; q++) {
    const int ci = q%3 -1;
    const int cj = (q/3)%3 -1;
    const int ck = q/9 -1;
    const int cc = ci*ci + cj*cj + ck*ck;
    constexpr real weight[4] { 8.f/27.f, 2.f/27.f, 1.f/54.f, 1.f/216.f };
    const real feq = weight[cc];
    const aint ijkIJKq = idx(i, j, k, q);
    buf1[ijkIJKq] = buf2[ijkIJKq] = feq;
  }
}, dst.data(), src.data()
);
CUCHECK();
    std::cout << std::endl;
    std::cout << "first_touch_sync" << std::flush;
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::cout << std::endl;
  });

CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
std::cout  << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;

  std::cout << "step: foo (iterative)" << std::endl;
  timer.elapse("foo", [&]() {
    double bw_max = -1, mlups_max = -1;
    for(int tt=0; tt<iiter; tt++) {
      util::timer timer;
      timer.elapse("foo", [&]() {
        for(int t=0; t<iter; t++) {
          dst.swap(src);
            //foo<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(dst.data(), src.data(), gi,gj,gk);
            util::invoke_device<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(
              [=]__device__(real* buf1, const real* buf2) { //, const aint* id) {//
                const aint i = threadIdx.x + blockIdx.x*blockDim.x;
                const aint j = threadIdx.y + blockIdx.y*blockDim.y;
                const aint k = threadIdx.z + blockIdx.z*blockDim.z;
                auto idx = []__device__(aint i, aint j, aint k, int q) {
                  return (i + nx*(j + ny*(k + nz*q)));
                };
                real f[Q], rho=0.f, u=0.f, v=0.f, w=0.f;
                #pragma unroll
                for(int q=0; q<Q; q++) {
                  const int ci = q%3 -1;
                  const int cj = (q/3)%3 -1;
                  const int ck = q/9 -1;
                  aint ii = i - ci;
                  if(ii < 0) { ii = nx-1; }
                  if(ii >= nx) { ii = 0; }
                  aint jj = j - cj; 
                  if(jj < 0) { jj = ny-1; } 
                  if(jj >= ny) { jj = 0; }
                  aint kk = k - ck;
                  if(kk < 0) { kk = nz-1; }
                  if(kk >= nz) { kk = 0; }
//                  f[q] = buf2[id[idx(ii,jj,kk,II,JJ,KK,q)]];
                  f[q] = buf2[idx(ii,jj,kk,q)];
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
                  buf1[idx(i, j, k, q)] = f[q];
                }
              }, dst.data(), src.data()//id_list.data()//
            );
            CUCHECK();
          }
CUDA_SAFE_CALL(cudaDeviceSynchronize());
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

  std::cout << std::endl;
CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
std::cout << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
  << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
}

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

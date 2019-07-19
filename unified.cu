#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>

#define GPU
#include "util/timer.hpp"
#include "util/cu_vector.hpp"
#include "util/cuda_safe_call.hpp"

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

// multi-gpu
constexpr int gx { 16 };
constexpr int gy { 1  };
constexpr int gz { 1  };
constexpr int num_gpu { gx*gy*gz };
constexpr int gpu[] { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
//constexpr int gpu[] { 0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15 };

// grid
constexpr bool strong { true };
constexpr long nx_ { 256 / gx * (strong ? 1 : gx) }; // if strong scaling, devide by g{x,y,z}
constexpr long ny_ { 256 / gy * (strong ? 1 : gy) };
constexpr long nz_ { 256 / gz * (strong ? 1 : gz) };
constexpr int Q { 27 }; // lattice Boltzmann
template<bool Cond, class Then, class Else> struct if_ { using type = Then; };
template<class Then, class Else> struct if_<false, Then, Else> { using type = Else; };
template<long L> struct enough_type { using type = 
  typename if_<(L > std::numeric_limits<int>::max()), long, int>::type;
};
using aint = typename enough_type<Q * nx_ * ny_ * nz_ * gx * gy * gz>::type;
//using aint = long;

constexpr aint bx { 4 };
constexpr aint by { 4 };
constexpr aint bz { 4 };
constexpr aint nx { nx_ / 4 };
constexpr aint ny { ny_ / 4 }; 
constexpr aint nz { nz_ / 4 };
constexpr aint NX { bx*nx*gx };
constexpr aint NY { by*ny*gy };
constexpr aint NZ { bz*nz*gz };
constexpr aint elem { Q*NX*NY*NZ };

// gpu kernel
constexpr int nth { 256 };
//constexpr int tx { std::gcd(256, nx) };
//constexpr int ty { nth/tx };
//constexpr int tz { nth/tx/ty };
//static_assert(nth == tx*ty*tz, "check blockDim.{x,y,z}");

// blocking gpu kernel
const dim3 blocking_block(bx, by, bz);
const dim3 blocking_grid(nx, ny, nz);

// measure
constexpr int iter { 2 };
constexpr int iiter { 10 };

template<class Func, class... Args> __global__ void kernel(Func func, Args... args) { func(args...); }

int main(int argc, char** argv) try {
  static_cast<void>(argc);
  static_cast<void>(argv);

  std::cout << " aint = " << typeid(aint).name() << sizeof(aint)*8 << std::endl;

  std::cout << "total mem = " << 2l*elem*sizeof(real) / 1024/1024/1024. << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(real) / 1024/1024/1024./num_gpu << " GiB" << std::endl;
  std::cout << "total mesh = (" << NX << ", " << NY << ", " << NZ << ")" << std::endl;
  std::cout << "       gpu = (" << gx << ", " << gy << ", " << gz << ")" << std::endl;
  std::cout << "mesh per g = (" << nx*bx << ", " << ny*by << ", " << nz*bz << ")" << std::endl;
  std::cout << "  blocking = (" << bx << ", " << by << ", " << bz << ")" << std::endl;
  std::cout << "blck per g = (" << nx << ", " << ny << ", " << nz << ")" << std::endl;

  util::timer timer;
  util::cu_vector<real> src(elem), dst(elem);
  //for(aint i=0; i<elem; i++) {
  //  src[i] = dst[i] = i;
  //}
  //std::cout << src.size() << std::endl;
  //util::cu_vector<real> src, dst;
  //for(aint i=0; i<elem; i++) {
  //  src.push_back(real(i));
  //  dst.push_back(real(2*i));
  //}

  std::cout << "step: init" << std::endl;
  timer.elapse("init", [&]() {
    const size_t memall = elem * sizeof(real);
    const size_t memgpu = memall / num_gpu;
    //cudaMallocManaged(&dst.data(), memall);
    //cudaMallocManaged(&src.data(), memall);
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
    std::cout << "first_touch" << std::flush;
    for(int gi=0; gi<num_gpu; gi++) {
      std::cout << "." << std::flush;
      cudaSetDevice(gpu[gi]);
      kernel<<<elem/num_gpu/nth, nth>>>(
        [=]__device__(real* buf1, real* buf2) {
          const aint ijk = threadIdx.x + blockIdx.x*blockDim.x + gi*blockDim.x*gridDim.x;
          buf1[ijk] = buf2[ijk] = ijk;
        }, dst.data(), src.data()
      );
    }
    std::cout << std::endl;
    std::cout << "first_touch::sync" << std::flush;
    for(int i=0; i<num_gpu; i++) {
      std::cout << "." << std::flush;
      cudaSetDevice(gpu[i]);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    std::cout << std::endl;
  });

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
    double bw_max = 0;
    for(int tt=0; tt<iiter; tt++) {
      util::timer timer;
      if(tt % (iiter/2) == 0) { // retouch
        for(aint i=0; i<elem; i++) {
          src[i] = dst[i] = i;
        }
      }
      timer.elapse("foo", [&]() {
        for(int t=0; t<iter; t++) {
          dst.swap(src);
          //for(int i=0; i<num_gpu; i++) {
          for(int gk=0; gk<gz; gk++) for(int gj=0; gj<gy; gj++) for(int gi=0; gi<gx; gi++) {
            cudaSetDevice(gpu[gi + gx*(gj + gy*gk)]);
            kernel<<<blocking_grid, blocking_block>>>(
              [=]__device__(real* buf1, const real* buf2) {
                struct tbl { 
                  aint t, b, L;
                  __device__ tbl(aint t, aint b, aint L): t(t), b(b), L(L) {}
                  __device__ void mod(const aint nt, const aint nb, const aint nl) {
                    if(t < 0) {
                      t = nt-1;
                      b -= 1;
                      if(b < 0) {
                        b = nb-1;
                        L -= 1;
                        if(L < 0) { L = nl-1; } // periodic boundary
                      }
                    } else if(t >= nt) {
                      t = 0;
                      b += 1;
                      if(b >= nb) {
                        b = 0;
                        L += 1;
                        if(L >= nl) { L = 0; } // periodic boundary
                      }
                    }
                  }
                };
                const tbl i ( threadIdx.x, blockIdx.x, gi );
                const tbl j ( threadIdx.y, blockIdx.y, gj );
                const tbl k ( threadIdx.z, blockIdx.z, gk );
                auto idx = []__device__(tbl i, tbl j, tbl k, int q) {
                  return (
                    i.t + bx*( j.t + by*( k.t + bz*( q + Q*(
                    i.b + nx*( j.b + ny*( k.b + nz*(
                    i.L + gx*( j.L + gy*( k.L
                  ))))))))));
                };
                #pragma unroll
                for(int q=0; q<Q; q++) {
                  const aint ci = q%3 -1;
                  const aint cj = (q/3)%3 -1;
                  const aint ck = q/9 -1;
                  tbl ii = i; ii.t -= ci; ii.mod(bx, nx, gx);
                  tbl jj = j; jj.t -= cj; jj.mod(by, ny, gy);
                  tbl kk = k; kk.t -= ck; kk.mod(bz, nz, gz);
                  buf1[idx(i,j,k,q)] = buf2[idx(ii,jj,kk,q)];
                }
              }, dst.data(), src.data()
            );
          }
          for(int i=0; i<num_gpu; i++) {
            cudaSetDevice(gpu[i]);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
          }
        }
      });
      const double bw_cache = 2.* elem* sizeof(real)* iter / timer["foo"] / 1024. / 1024. / 1024.;
      bw_max = std::max(bw_max, bw_cache);
      std::cout << "bandwidth: " << bw_max << " GiB/s max, " << bw_cache << " GiB/s recent\r" << std::endl;
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

  for(aint max=0, min=std::numeric_limits<aint>::max(), i=0; i<elem; i++) {
    max = std::max(max, dst[i]);
    min = std::min(min, dst[i]);
    if(i == elem-1) { std::cout << "dst = " << min << " -- " << max << std::endl; }
  }

  timer.elapse("fina-GPU", [&src, &dst]() {
    src.clear(); src.shrink_to_fit();
    dst.clear(); dst.shrink_to_fit();
    cudaProfilerStop();
    cudaDeviceReset();
  });

  timer.showall();

  return 0;
} catch (const std::runtime_error& e) {
  std::cerr << "fatal: " << e.what() << std::endl;
  return 1;
} catch (...) {
  std::cerr << "fatal: unknown error" << std::endl;
  return 2;
}

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
constexpr int gx { 16  };
constexpr int gy { 1  };
constexpr int gz { 1  };
constexpr int num_gpu { gx*gy*gz };
constexpr int gpu[] { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };

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

constexpr aint nx { nx_ };
constexpr aint ny { ny_ }; 
constexpr aint nz { nz_ };
constexpr aint NX { nx*gx };
constexpr aint NY { ny*gy };
constexpr aint NZ { nz*gz };
constexpr aint elem { Q*NX*NY*NZ };

// gpu kernel
constexpr int nth { 256 };
constexpr int tx { std::gcd(256, nx) };
constexpr int ty { nth/tx };
constexpr int tz { nth/tx/ty };
static_assert(nth == tx*ty*tz, "check blockDim.{x,y,z}");

//
constexpr int mem_invalid_shift = elem/num_gpu/8; // one block?

// measure
constexpr int iter { 2 };
constexpr int iiter { 10 };

template<class Func, class... Args> __global__ void kernel(Func func, Args... args) { func(args...); }

int main(int argc, char** argv) try {
  static_cast<void>(argc);
  static_cast<void>(argv);

  std::cout << " aint = " << typeid(aint).name() << std::endl;

  std::cout << "total mem = " << 2l*elem*sizeof(real) / 1024/1024/1024. << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(real) / 1024/1024/1024./num_gpu << " GiB" << std::endl;
  std::cout << "total mesh = (" << NX << ", " << NY << ", " << NZ << ")" << std::endl;
  std::cout << "  partition= (" << gx << ", " << gy << ", " << gz << ")" << std::endl;
  std::cout << "  per gpu  = (" << nx << ", " << ny << ", " << nz << ")" << std::endl;

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
      // invalid prefetch
      const size_t ofsi = (i != 0) ? ofs + mem_invalid_shift : ofs;
      const size_t memi = i == 0 ? memgpu + mem_invalid_shift*sizeof(float)
                          : (i == num_gpu-1 ? memgpu - mem_invalid_shift*sizeof(float)
                             : memgpu)
                        ;
      cudaMemAdvise(dst.data() + ofsi, memi, cudaMemAdviseSetPreferredLocation, gpu[i]);
      cudaMemAdvise(src.data() + ofsi, memi, cudaMemAdviseSetPreferredLocation, gpu[i]);
      cudaMemPrefetchAsync(dst.data() + ofs, memi, gpu[i]);
      cudaMemPrefetchAsync(src.data() + ofs, memi, gpu[i]);
    }
    std::cout << std::endl;
    for(int gi=0; gi<num_gpu; gi++) {
      std::cout << "." << std::flush;
      cudaSetDevice(gpu[gi]);
      const size_t ofs = elem*gi/num_gpu;
      // invalid prefetch
      const size_t ofsi = (gi != 0) ? ofs + mem_invalid_shift : ofs;
      const size_t memi = gi == 0 ? memgpu + mem_invalid_shift*sizeof(float)
                          : (gi == num_gpu-1 ? memgpu - mem_invalid_shift*sizeof(float)
                             : memgpu)
                        ;
      const long grid = (memi/sizeof(real) + nth -1)/nth;
      kernel<<<grid, nth>>>(
        [=]__device__(real* buf1, real* buf2) {
          const aint ijk = threadIdx.x + blockIdx.x*blockDim.x + ofsi;
          if(ijk >= memi/sizeof(real)) return;
          buf1[ijk] = buf2[ijk] = ijk;
        }, dst.data(), src.data()
      );
    }
    std::cout << std::endl;
    std::cout << "first_touch" << std::flush;
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
            kernel<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(
              [=]__device__(real* buf1, const real* buf2) {
                const aint i = threadIdx.x + blockIdx.x*blockDim.x;
                const aint j = threadIdx.y + blockIdx.y*blockDim.y;
                const aint k = threadIdx.z + blockIdx.z*blockDim.z;
                auto& I = gi;
                auto& J = gj;
                auto& K = gk;
                auto idx = []__device__(aint i, aint j, aint k, aint I, aint J, aint K, int q) {
                  return (i + nx*(j + ny*(k + nz*(q + Q*(I + gx*(J + gy*K))))));
                };
                #pragma unroll
                for(int q=0; q<Q; q++) {
                  const aint ci = q%3 -1;
                  const aint cj = (q/3)%3 -1;
                  const aint ck = q/9 -1;
                  aint ii = i - ci, II = I;
                  if(ii < 0) { ii = nx-1; II = (I-1+gx)%gx; }
                  if(ii >= nx) { ii = 0; II = (I+1)%gx; }
                  aint jj = j - cj, JJ = J; 
                  if(jj < 0) { jj = ny-1; JJ = (J-1+gy)%gy; } 
                  if(jj >= ny) { jj = 0; JJ = (J+1)%gy; }
                  aint kk = k - ck, KK = K;
                  if(kk < 0) { kk = nz-1; KK = (K-1+gz)%gz; }
                  if(kk >= nz) { kk = 0; KK = (K+1)%gz; }
                  buf1[idx(i,j,k,I,J,K,q)] = buf2[idx(ii,jj,kk,II,JJ,KK,q)];
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

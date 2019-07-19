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

// grid
constexpr bool strong { false };
constexpr long nx_ { 1024 / gx * (strong ? 1 : gx) }; // if strong scaling, devide by g{x,y,z}
constexpr long ny_ { 1024 / gy * (strong ? 1 : gy) };
constexpr long nz_ { 1024 / gz * (strong ? 1 : gz) };
template<bool Cond, class Then, class Else> struct if_ { using type = Then; };
template<class Then, class Else> struct if_<false, Then, Else> { using type = Else; };
template<long L> struct enough_type { using type = 
  typename if_<(L > std::numeric_limits<int>::max()), long, int>::type;
};
using aint = typename enough_type<nx_ * ny_ * nz_ * gx * gy * gz>::type;
//using aint = long;

constexpr aint nx { nx_ };
constexpr aint ny { ny_ }; 
constexpr aint nz { nz_ };
constexpr aint NX { nx*gx };
constexpr aint NY { ny*gy };
constexpr aint NZ { nz*gz };
constexpr aint elem { NX*NY*NZ };

// gpu kernel
constexpr int nth { 256 };
constexpr int tx { std::gcd(256, nx) };
constexpr int ty { nth/tx };
constexpr int tz { nth/tx/ty };
static_assert(nth == tx*ty*tz, "check blockDim.{x,y,z}");

// measure
constexpr int iter { 2 };
constexpr int iiter { 10 };

template<class Func, class... Args> __global__ void kernel(Func func, Args... args) { func(args...); }

int main(int argc, char** argv) try {
  static_cast<void>(argc);
  static_cast<void>(argv);
  util::timer timer;
  util::cu_ptr<real> src(elem), dst(elem);
  //for(aint i=0; i<elem; i++) {
  //  src[i] = dst[i] = i;
  //}
  //std::cout << src.size() << std::endl;
  //util::cu_vector<real> src, dst;
  //for(aint i=0; i<elem; i++) {
  //  src.push_back(real(i));
  //  dst.push_back(real(2*i));
  //}

  //// test task id
  //constexpr int lv_max = 3;
  //constexpr int num_color = 8;
  //util::cu_vec3d<int, lv_max, num_color> task_id;
  //for(int lv=0; lv<lv_max; lv++) {
  //  for(int color=0; color<num_color; color++) {
  //    for(int i=0; i<(1+lv*color); i++) {
  //      task_id[lv][color].push_back(2*color + 100.f*lv);
  //    }
  //  }
  //}
  //for(int lv=0; lv<lv_max; lv++) {
  //  for(int color=0; color<num_color; color++) {
  //    std::cout << "task_id[" << lv << "][" << color << "] = " << std::flush;
  //    for(const auto& id: task_id[lv][color]) {
  //      std::cout << id << " " << std::flush;
  //    }
  //    std::cout << std::endl;
  //  }
  //}

  std::cout << " aint = " << typeid(aint).name() << std::endl;

  std::cout << "total mem = " << 2l*elem*sizeof(real) / 1024/1024/1024. << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(real) / 1024/1024/1024./num_gpu << " GiB" << std::endl;
  std::cout << "total mesh = (" << NX << ", " << NY << ", " << NZ << ")" << std::endl;
  std::cout << "  partition= (" << gx << ", " << gy << ", " << gz << ")" << std::endl;
  std::cout << "  per gpu  = (" << nx << ", " << ny << ", " << nz << ")" << std::endl;

  std::cout << "step: init" << std::endl;
  timer.elapse("init", [&]() {
    const size_t memall = elem * sizeof(real);
    const size_t memgpu = memall / num_gpu;
    //cudaMallocManaged(&dst.data(), memall);
    //cudaMallocManaged(&src.data(), memall);
    for(int i=0; i<num_gpu; i++) {
      cudaMemAdvise(dst.data(), memall, cudaMemAdviseSetAccessedBy, gpu[i]);
      cudaMemAdvise(src.data(), memall, cudaMemAdviseSetAccessedBy, gpu[i]);
    }
    for(int i=0; i<num_gpu; i++) {
      const size_t ofs = elem*i/num_gpu;
      cudaMemAdvise(dst.data() + ofs, memgpu, cudaMemAdviseSetPreferredLocation, gpu[i]);
      cudaMemAdvise(src.data() + ofs, memgpu, cudaMemAdviseSetPreferredLocation, gpu[i]);
      cudaMemPrefetchAsync(dst.data() + ofs, memgpu, gpu[i]);
      cudaMemPrefetchAsync(src.data() + ofs, memgpu, gpu[i]);
    }
    for(int gi=0; gi<num_gpu; gi++) {
      cudaSetDevice(gpu[gi]);
      kernel<<<elem/num_gpu/nth, nth>>>(
        [=]__device__(real* buf1, real* buf2) {
          const aint ijk = threadIdx.x + blockIdx.x*blockDim.x + gi*blockDim.x*gridDim.x;
          buf1[ijk] = buf2[ijk] = ijk;
        }, dst.data(), src.data()
      );
    }
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(gpu[i]);
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
                aint im = i-1, ip = i+1, jm = j-1, jp = j+1, km = k-1, kp = k+1;
                aint IM = I, IP = I, JM = J, JP = J, KM = K, KP = K;
                if(im < 0) { im = nx-1; IM = (I-1+gx)%gx; }
                if(ip >= nx) { ip = 0; IP = (I+1)%gx; }
                if(jm < 0) { jm = ny-1; JM = (J-1+gy)%gy; } 
                if(jp >= ny) { jp = 0; JP = (J+1)%gy; }
                if(km < 0) { km = nz-1; KM = (K-1+gz)%gz; }
                if(kp >= nz) { kp = 0; KP = (K+1)%gz; }
                auto idx = []__device__(aint i, aint j, aint k, aint I, aint J, aint K) {
                  return i + nx*(j + ny*(k + nz*(I + gx*(J + gy*K))));
                };
                const aint ijk = idx(i, j, k, I, J, K);
                const aint je[6] = { idx(im, j, k, IM, J, K),
                                 idx(ip, j, k, IP, J, K),
                                 idx(i, jm, k, I, JM, K),
                                 idx(i, jp, k, I, JP, K),
                                 idx(i, j, km, I, J, KM),
                                 idx(i, j, kp, I, J, KP) };
                const real cc = 0.1f;
                buf1[ijk] = (1.f-6.f*cc)*buf2[ijk] + cc*(buf2[je[0]]);;// + buf2[je[1]] + buf2[je[2]] + buf2[je[3]] + buf2[je[4]] +buf2[je[5]]);
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

  timer.elapse("fina-GPU", []() {
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

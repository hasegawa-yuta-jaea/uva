// measure
constexpr int iter { 2 };
constexpr int iiter { 10 };
//#define NODEBUG
// multi-gpu
constexpr int gx { 1 };
constexpr int gy { 1  };
constexpr int gz { 16  };
constexpr int num_gpu { gx*gy*gz };
constexpr int gpu[] { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
// grid
constexpr bool strong { true };
constexpr long nx_ { 512 / gx * (strong ? 1 : gx) }; // if strong scaling, devide by g{x,y,z}
constexpr long ny_ { 512 / gy * (strong ? 1 : gy) };
constexpr long nz_ { 512 / gz * (strong ? 1 : gz) };
constexpr int Q { 27 }; // lbm
constexpr int Qmpi { 27 }; // lbm mpi send at surface (torima all, future work optimizes into 9)

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <array>

#define GPU
#include "util/timer.hpp"
#include "util/cu_vector.hpp"
#include "util/cuda_safe_call.hpp"
#include "util/conditional.hpp"
#include "util/invoker.hpp"
#include "util/mpi_safe_call.hpp"
#include "util/cu_device_ptr.hpp"

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
//using aint = typename enough_type<Q* (nx_ +2) * (ny_ +2) * (nz_ +2)>::type;
using aint = long;

constexpr aint hx { gx > 1 ? 1 : 0 }; // halo
constexpr aint hy { gy > 1 ? 1 : 0 }; // halo
constexpr aint hz { gz > 1 ? 1 : 0 }; // halo
constexpr aint nx { nx_ + 2*hx }; 
constexpr aint ny { ny_ + 2*hy }; 
constexpr aint nz { nz_ + 2*hz }; 
constexpr aint elem { Q*nx*ny*nz };

// gpu kernel
constexpr int nth { 1024 };
constexpr int tx { std::gcd(128, nx_) };
constexpr int ty { std::gcd(4, ny_) };
constexpr int tz { std::gcd(nth/tx/ty, nz_) };
static_assert(nth == tx*ty*tz, "check blockDim.{x,y,z}");

inline __device__ aint idx(aint i, aint j, aint k, aint q) { return(i+ nx*(j+ ny*(k+ nz*q))); }

int main(int argc, char** argv) try {
  CUDA_SAFE_CALL(cudaSetDevice(gpu[0]));
  MPI_SAFE_CALL(MPI_Init(&argc, &argv));
  const auto&& mpirank = []() { int rank_; MPI_Comm_rank(MPI_COMM_WORLD, &rank_); return rank_; }();
  const auto&& mpisize = []() { int size_; MPI_Comm_size(MPI_COMM_WORLD, &size_); return size_; }();
  RUNTIME_ASSERT(mpisize == num_gpu);
  const int gpui = gpu[mpirank];
  cudaSetDevice(gpui);

  std::ostream cout0(mpirank==0 ? std::cout.rdbuf() : (new std::ofstream("dev/null"))->rdbuf());
  std::cout << "this is rank " << mpirank << std::endl;
  MPI_SAFE_CALL(MPI_Barrier(MPI_COMM_WORLD));

  cout0 << " aint = " << typeid(aint).name() << std::endl;
  cout0 << "total mem = " << 2l*elem*sizeof(real) / 1024/1024/1024.*num_gpu << " + " << elem*sizeof(aint)/1024/1024/1024. << " GiB" << std::endl;
  cout0 << "  per gpu = " << 2l*elem*sizeof(real) / 1024/1024/1024.         << " + " << elem*sizeof(aint)/1024/1024/1024. << " GiB" << std::endl;
  cout0 << "total mesh = (" << nx_*gx << ", " << ny_*gy << ", " << nz_*gz << ")" << std::endl;
  cout0 << "  partition= (" << gx << ", " << gy << ", " << gz << ")" << std::endl;
  cout0 << "  per gpu  = (" << nx << ", " << ny << ", " << nz << ")" << std::endl;
  cout0 << "thread     = (" << tx << ", " << ty << ", " << tz << ")" << std::endl;

  util::timer timer;
  util::cu_device_ptr<real> src(nx*ny*nz*Q), dst(nx*ny*nz*Q);
  util::cu_device_ptr<real> 
    buf_recv_w(hx * ny * nz * Qmpi), buf_recv_e(hx * ny * nz * Qmpi),
    buf_recv_n(nx * hy * nz * Qmpi), buf_recv_s(nx * hy * nz * Qmpi),
    buf_recv_b(nx * ny * hz * Qmpi), buf_recv_t(nx * ny * hz * Qmpi),
    buf_send_w(hx * ny * nz * Qmpi), buf_send_e(hx * ny * nz * Qmpi),
    buf_send_n(nx * hy * nz * Qmpi), buf_send_s(nx * hy * nz * Qmpi),
    buf_send_b(nx * ny * hz * Qmpi), buf_send_t(nx * ny * hz * Qmpi);
//  util::cu_ptr<aint> id_list(elem);
  thrust::device_ptr<real> thrust_ptr_dst = thrust::device_pointer_cast(dst.data());

  cout0 << "step: init" << std::endl;
  timer.elapse("init", [&]() {
    cout0 << "first_touch" << std::flush;
    util::invoke_device<<<elem/Q/nth+1, nth>>>(
      []__device__(real* buf1, real* buf2) {
        const aint ijk = threadIdx.x + blockDim.x * blockIdx.x;
        if(ijk >= nx*ny*nz) return;
        #pragma unroll
        for(int q=0; q<Q; q++) {
          const int ci = q%3 -1;
          const int cj = (q/3)%3 -1;
          const int ck = q/9 -1;
          const int cc = ci*ci + cj*cj + ck*ck;
          constexpr real weight[4] { 8.f/27.f, 2.f/27.f, 1.f/54.f, 1.f/216.f };
          const real feq = weight[cc];
          const aint ijkq = ijk + nx*ny*nz*q;
          buf1[ijkq] = buf2[ijkq] = feq;
        }
      }, dst.data(), src.data()
    );
    CUCHECK();
    cout0 << std::endl;
    cout0 << "first_touch_sync" << std::flush;
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    MPI_SAFE_CALL(MPI_Barrier(MPI_COMM_WORLD));
    cout0 << std::endl;
    const double 
      min = *thrust::min_element(thrust::device, thrust_ptr_dst, thrust_ptr_dst + dst.size()),
      max = *thrust::max_element(thrust::device, thrust_ptr_dst, thrust_ptr_dst + dst.size());
    std::cout << "@rank " << mpirank << ": dst = " << min << " -- " << max << std::endl;
    MPI_SAFE_CALL(MPI_Barrier(MPI_COMM_WORLD));
  });

  size_t mfree, mtotal;
  CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
  cout0 << "gpu " << std::setw(2) << std::setfill(' ') << gpui << ": "
    << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  MPI_SAFE_CALL(MPI_Barrier(MPI_COMM_WORLD));

  cudaProfilerStart();
  cout0 << "step: foo (iterative)" << std::endl;
  timer.elapse("foo-iiter", [&]() {
    double bw_max = -1, mlups_max = -1;
    for(int tt=0; tt<iiter; tt++) {
      util::timer timer;
      timer.elapse("foo", [&]() {
        for(int t=0; t<iter; t++) {
          dst.swap(src);
          // mpi
          std::array<MPI_Request, 6> req_send, req_recv;
          MPI_Status status_null;
          auto&& pack = []__device__(real* buf, const dim3 size, const real* data, const dim3 ofs) {
            const aint i = threadIdx.x + blockIdx.x * blockDim.x;
            const aint j = threadIdx.y + blockIdx.y * blockDim.y;
            const aint k = threadIdx.z + blockIdx.z * blockDim.z;
            if(i >= size.x || j >= size.y || k >= size.z) return;
            #pragma unroll
            for(int q=0; q<Q; q++) {
              buf[i + size.x*(j + size.y*(k + size.z*q))] = data[idx(i + ofs.x, j + ofs.y, k + ofs.z, q)];
            }
          };
          auto&& unpack = []__device__(real* data, const dim3 size, const real* buf, const dim3 ofs) {
            const aint i = threadIdx.x + blockIdx.x * blockDim.x;
            const aint j = threadIdx.y + blockIdx.y * blockDim.y;
            const aint k = threadIdx.z + blockIdx.z * blockDim.z;
            if(i >= size.x || j >= size.y || k >= size.z) return;
            #pragma unroll
            for(int q=0; q<Q; q++) {
              data[idx(i + ofs.x, j + ofs.y, k + ofs.z, q)] = buf[i + size.x*(j + size.y*(k + size.z*q))];
            }
          };
          if(hz > 0) {
            // z
            // pack-sendrecv bottom
            constexpr aint size_z = nx * ny * hz * Qmpi * sizeof(real);
            static_assert(size_z <= std::numeric_limits<int>::max());
            CUCHECK({ util::invoke_device<<<dim3(nx/tx+1, ny/ty+1, 1), dim3(tx, ty, 1)>>> (
                pack, buf_send_b.data(), dim3(nx, ny, 1), src.data(), dim3(0, 0, 1)); 
            });
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const int rank_send_b = (mpirank - gx*gy + mpisize) % mpisize;
            MPI_SAFE_CALL(MPI_Isend(buf_send_b.data(), size_z, MPI_BYTE, rank_send_b, 0, MPI_COMM_WORLD, &req_send[0]));
            const int rank_recv_b = (mpirank + gx*gy)           % mpisize;
            MPI_SAFE_CALL(MPI_Irecv(buf_recv_b.data(), size_z, MPI_BYTE, rank_recv_b, 0, MPI_COMM_WORLD, &req_recv[0]));
            // pack-sendrecv top
            CUCHECK({ util::invoke_device<<<dim3(nx/tx+1, ny/ty+1, 1), dim3(tx, ty, 1)>>> (
                pack, buf_send_t.data(), dim3(nx, ny, 1), src.data(), dim3(0, 0, nz-2));
            });
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const int rank_send_t = rank_recv_b;
            MPI_SAFE_CALL(MPI_Isend(buf_send_t.data(), size_z, MPI_BYTE, rank_send_t, 1,  MPI_COMM_WORLD, &req_send[1]));
            const int rank_recv_t = rank_send_b;
            MPI_SAFE_CALL(MPI_Irecv(buf_recv_t.data(), size_z, MPI_BYTE, rank_recv_t, 1,  MPI_COMM_WORLD, &req_recv[1]));
            // unpack
            MPI_SAFE_CALL(MPI_Wait(&req_send[0], &status_null)); MPI_SAFE_CALL(MPI_Wait(&req_recv[0], &status_null));
            CUCHECK({ util::invoke_device<<<dim3(nx/tx+1, ny/ty+1, 1), dim3(tx, ty, 1)>>> (
                unpack, src.data(), dim3(nx, ny, 1), buf_recv_b.data(), dim3(0, 0, nz-1)); 
            });
            MPI_SAFE_CALL(MPI_Wait(&req_send[1], &status_null)); MPI_SAFE_CALL(MPI_Wait(&req_recv[1], &status_null));
            CUCHECK({ util::invoke_device<<<dim3(nx/tx+1, ny/ty+1, 1), dim3(tx, ty, 1)>>> (
                unpack, src.data(), dim3(nx, ny, 1), buf_recv_t.data(), dim3(0, 0, 0)); 
            });
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
          }
          if(hy > 0) {
            // y
            // pack-sendrecv south
            constexpr aint size_y = nx * nz * Qmpi * sizeof(real);
            CUCHECK({ util::invoke_device<<<dim3(nx/tx+1, 1, nz/ty+1), dim3(tx, 1, ty)>>> (
                pack, buf_send_s.data(), dim3(nx, 1, nz), src.data(), dim3(0, 1, 0)); 
            });
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const int rank_send_s = (mpirank - gx + mpisize) % mpisize;
            MPI_SAFE_CALL(MPI_Isend(buf_send_s.data(), size_y, MPI_BYTE, rank_send_s, 0, MPI_COMM_WORLD, &req_send[0]));
            const int rank_recv_s = (mpirank + gx) % mpisize;
            MPI_SAFE_CALL(MPI_Irecv(buf_recv_s.data(), size_y, MPI_BYTE, rank_recv_s, 0, MPI_COMM_WORLD, &req_recv[0]));
            // pack-sendrecv north
            CUCHECK({ util::invoke_device<<<dim3(nx/tx+1, 1, nz/ty+1), dim3(tx, 1, ty)>>> (
                pack, buf_send_n.data(), dim3(nx, 1, nz), src.data(), dim3(0, ny-2, 0));
            });
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const int rank_send_n = rank_recv_s;
            MPI_SAFE_CALL(MPI_Isend(buf_send_n.data(), size_y, MPI_BYTE, rank_send_n, 1, MPI_COMM_WORLD, &req_send[1]));
            const int rank_recv_n = rank_send_s;
            MPI_SAFE_CALL(MPI_Irecv(buf_recv_t.data(), size_y, MPI_BYTE, rank_recv_n, 1, MPI_COMM_WORLD, &req_recv[1]));
            // unpack
            MPI_SAFE_CALL(MPI_Wait(&req_send[0], &status_null)); MPI_SAFE_CALL(MPI_Wait(&req_recv[0], &status_null));
            CUCHECK({ util::invoke_device<<<dim3(nx/tx+1, 1, nz/ty+1), dim3(tx, 1, ty)>>> (
                unpack, src.data(), dim3(nx_, 1, nz), buf_recv_s.data(), dim3(0, ny-1, 0)); 
            });
            MPI_SAFE_CALL(MPI_Wait(&req_send[1], &status_null)); MPI_SAFE_CALL(MPI_Wait(&req_recv[1], &status_null));
            CUCHECK({ util::invoke_device<<<dim3(nx/tx+1, 1, nz/ty+1), dim3(tx, 1, ty)>>> (
                unpack, src.data(), dim3(nx, 1, nz), buf_recv_n.data(), dim3(0, 0, 0)); 
            });
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
          }
          if(hx > 0) {
            // x
            // pack-sendrecv west
            constexpr aint size_x = ny * nz * Qmpi * sizeof(real);
            CUCHECK({ util::invoke_device<<<dim3(1, ny/tx+1, nz/ty+1), dim3(1, tx, ty)>>> (
                pack, buf_send_w.data(), dim3(1, ny, nz), src.data(), dim3(0, 0, 0)); 
            });
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const int rank_send_w = (mpirank - 1 + mpisize) % mpisize;
            MPI_SAFE_CALL(MPI_Isend(buf_send_w.data(), size_x, MPI_BYTE, rank_send_w, 0, MPI_COMM_WORLD, &req_send[0]));
            const int rank_recv_w = (mpirank + 1) % mpisize;
            MPI_SAFE_CALL(MPI_Irecv(buf_recv_w.data(), size_x, MPI_BYTE, rank_recv_w, 0, MPI_COMM_WORLD, &req_recv[0]));
            // pack-sendrecv east
            CUCHECK({ util::invoke_device<<<dim3(1, ny/tx+1, nz/ty+1), dim3(1, tx, ty)>>> (
                pack, buf_send_e.data(), dim3(1, ny, nz), src.data(), dim3(nx-2, 0, 0));
            });
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            const int rank_send_e = rank_recv_w;
            MPI_SAFE_CALL(MPI_Isend(buf_send_e.data(), size_x, MPI_BYTE, rank_send_e, 1, MPI_COMM_WORLD, &req_send[1]));
            const int rank_recv_e = rank_send_w;
            MPI_SAFE_CALL(MPI_Irecv(buf_recv_e.data(), size_x, MPI_BYTE, rank_recv_e, 1, MPI_COMM_WORLD, &req_recv[1]));
            // unpack
            MPI_SAFE_CALL(MPI_Wait(&req_send[0], &status_null)); MPI_SAFE_CALL(MPI_Wait(&req_recv[0], &status_null));
            CUCHECK({ util::invoke_device<<<dim3(1, ny/tx+1, nz/ty+1), dim3(1, tx, ty)>>> (
                unpack, src.data(), dim3(1, ny, nz), buf_recv_w.data(), dim3(nx-1, 0, 0)); 
            });
            MPI_SAFE_CALL(MPI_Wait(&req_send[1], &status_null)); MPI_SAFE_CALL(MPI_Wait(&req_recv[1], &status_null));
            CUCHECK({ util::invoke_device<<<dim3(1, ny/tx+1, nz/ty+1), dim3(1, tx, ty)>>> (
                unpack, src.data(), dim3(1, ny, nz), buf_recv_e.data(), dim3(0, 0, 0)); 
            });
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
          }
          // lbm
          CUCHECK({
            util::invoke_device<<<dim3(nx_/tx, ny_/ty, nz_/tz), dim3(tx, ty, tz)>>>(
              [=]__device__(real* buf1, const real* buf2) { //, const aint* id) {//
                const aint i = threadIdx.x + blockIdx.x*blockDim.x + hx;
                const aint j = threadIdx.y + blockIdx.y*blockDim.y + hy;
                const aint k = threadIdx.z + blockIdx.z*blockDim.z + hz;
                if(i >= nx-hx || j >= ny-hy || k >= nz-hz) return;
                real f[Q], rho=0.f, u=0.f, v=0.f, w=0.f;
                #pragma unroll
                for(int q=0; q<Q; q++) {
                  const int ci = q%3 -1;
                  const int cj = (q/3)%3 -1;
                  const int ck = q/9 -1;
                  const int ii = (i-ci+nx)%nx;
                  const int jj = (j-cj+ny)%ny;
                  const int kk = (k-ck+nz)%nz;
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
          });
          CUDA_SAFE_CALL(cudaDeviceSynchronize());
        }
      });
      const double bw_cache = nx_*ny_*nz_*Q*num_gpu* 2.*sizeof(real)* iter / timer["foo"] / 1024. / 1024. / 1024.;
      bw_max = std::max(bw_max, bw_cache);
      cout0 << "bandwidth: " << bw_max << " GiB/s max, " << bw_cache << " GiB/s recent" << std::endl;
      const double mlups_cache = double(num_gpu*nx_*ny_*nz_)* iter / timer["foo"] / 1e6f;
      mlups_max = std::max(mlups_max, mlups_cache);
      cout0 << "performance: " << mlups_max << " MLUPS max, " << mlups_cache << " MLUPS recent" << std::endl;
    }
  });
  cudaProfilerStop();

  cout0 << std::endl;
  for(int i=0; i<num_gpu; i++) {
    size_t mfree, mtotal;
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
    cout0 << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
      << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  }

  const double 
    min = *thrust::min_element(thrust::device, thrust_ptr_dst, thrust_ptr_dst + dst.size()),
    max = *thrust::max_element(thrust::device, thrust_ptr_dst, thrust_ptr_dst + dst.size());
  std::cout << "@rank " << mpirank << ": dst = " << min << " -- " << max << std::endl;

  timer.showall();

  src.reset();
  dst.reset();
  buf_send_w.reset(); buf_recv_w.reset();
  buf_send_e.reset(); buf_recv_e.reset();
  buf_send_s.reset(); buf_recv_s.reset();
  buf_send_n.reset(); buf_recv_n.reset();
  buf_send_b.reset(); buf_recv_b.reset();
  buf_send_t.reset(); buf_recv_t.reset();
  MPI_SAFE_CALL(MPI_Finalize());

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

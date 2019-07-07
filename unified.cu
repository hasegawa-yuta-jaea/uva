#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sys/time.h>

#define GPU
//#include "utility/CUDA_SAFE_CALL.cuda.h"

#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
			     fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
			     exit(-1);}} while(0)
#define CUDA_SAFE_CALL(X) ERR_NE((X),cudaSuccess)

//#include "util/timer.hpp"

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
//constexpr int nth = 1024;
constexpr int nth = 256;
constexpr int tx = std::gcd(dim(256l), nx);
constexpr int ty = nth/tx;
constexpr int tz = 1;


// measure
constexpr int iter = 100;

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
      src[im + nx*(j  + ny*k)]
    + src[ip + nx*(j  + ny*k)]
    + src[i  + nx*(jm + ny*k)]
    + src[i  + nx*(jp + ny*k)]
    + src[ijkm]
    + src[ijkp] );
}

int main(int argc, char** argv) {
  float *dst, *src;

  std::cout << "total mem = " << 2l*elem*sizeof(float) / 1024/1024/1024. << " GiB" << std::endl;
  std::cout << "  per gpu = " << 2l*elem*sizeof(float) / 1024/1024/1024./num_gpu << " GiB" << std::endl;
  std::cout << "total mesh = (" << NX << ", " << NY << ", " << NZ << ")" << std::endl;
  std::cout << "  partition= (" << gx << ", " << gy << ", " << gz << ")" << std::endl;
  std::cout << "  per gpu  = (" << nx << ", " << ny << ", " << nz << ")" << std::endl;

  struct timeval tv_start, tv_end;
  double elapsed;
  
  std::cout << "step: malloc&init" << std::endl;
  {
    gettimeofday( &tv_start, NULL );
    const size_t memall = elem * sizeof(float);
    const size_t memgpu = memall / num_gpu;
    CUDA_SAFE_CALL(cudaMallocManaged(&dst, memall));
    CUDA_SAFE_CALL(cudaMallocManaged(&src, memall));
    for(int i=0; i<num_gpu; i++) {
	CUDA_SAFE_CALL(cudaMemAdvise(dst, memall, cudaMemAdviseSetAccessedBy, i));
	CUDA_SAFE_CALL(cudaMemAdvise(src, memall, cudaMemAdviseSetAccessedBy, i));
    }
    for(int i=0; i<num_gpu; i++) {
      const size_t ofs = elem*i/num_gpu;
      CUDA_SAFE_CALL(cudaMemAdvise(dst + ofs, memgpu, cudaMemAdviseSetPreferredLocation, i));
      CUDA_SAFE_CALL(cudaMemAdvise(src + ofs, memgpu, cudaMemAdviseSetPreferredLocation, i));
      CUDA_SAFE_CALL(cudaMemPrefetchAsync(dst + ofs, memgpu, i));
      CUDA_SAFE_CALL(cudaMemPrefetchAsync(src + ofs, memgpu, i));
    }
    for(int i=0; i<num_gpu; i++) {
	CUDA_SAFE_CALL(cudaSetDevice(i));
	init<<<elem/num_gpu/nth, nth>>>(dst, src, i);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    for(int i=0; i<num_gpu; i++) {
	CUDA_SAFE_CALL(cudaSetDevice(i));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    gettimeofday( &tv_end, NULL );
    elapsed = (double)(tv_end.tv_sec - tv_start.tv_sec) + (double)(tv_end.tv_usec - tv_start.tv_usec) * 1e-6;
    std::cout << "real (sec) : " << elapsed << std::endl; 
  }

  for(int i=0; i<num_gpu; i++) {
    size_t mfree, mtotal;
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
    std::cout << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
      << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  }

  std::cout << "step: foo (first call)" << std::endl;
  {
    gettimeofday( &tv_start, NULL );
    for(int k=0; k<gz; k++) for(int j=0; j<gy; j++) for(int i=0; i<gx; i++) {
      cudaSetDevice(i + gx*(j + gy*k));
      foo<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(dst, src, i,j,k);
    }
    for(int i=0; i<num_gpu; i++) {
      cudaSetDevice(i);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    gettimeofday( &tv_end, NULL );
    elapsed = (double)(tv_end.tv_sec - tv_start.tv_sec) + (double)(tv_end.tv_usec - tv_start.tv_usec) * 1e-6;
    std::cout << "real (sec) : " << elapsed << std::endl; 
  }

  std::cout << "step: foo (iterative)" << std::endl;
  {
    gettimeofday( &tv_start, NULL );
    for(int t=0; t<iter; t++) {
      float* tmp = src;
      src = dst;
      dst = tmp;
      //for(int i=0; i<num_gpu; i++) {
      for(int k=0; k<gz; k++) for(int j=0; j<gy; j++) for(int i=0; i<gx; i++) {
        cudaSetDevice(i + gx*(j + gy*k));
        foo<<<dim3(nx/tx, ny/ty, nz/tz), dim3(tx, ty, tz)>>>(dst, src, i,j,k);
      }
      for(int i=0; i<num_gpu; i++) {
        cudaSetDevice(i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }
    }
    gettimeofday( &tv_end, NULL );
    elapsed = (double)(tv_end.tv_sec - tv_start.tv_sec) + (double)(tv_end.tv_usec - tv_start.tv_usec) * 1e-6;
    std::cout << "real (sec) : " << elapsed << std::endl; 
  }
  // const double bw_cache = 2.* elem* sizeof(float)* iter / timer["foo"] / 1024. / 1024. / 1024.;
  // std::cout << "bandwidth: " << bw_cache << " GiB/s\r" << std::endl;

  std::cout << std::endl;
  for(int i=0; i<num_gpu; i++) {
    size_t mfree, mtotal;
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMemGetInfo(&mfree, &mtotal));
    std::cout << "gpu " << std::setw(2) << std::setfill(' ') << i << ": "
      << std::setw(4) << double(mtotal - mfree) /1024./1024./1024. << " GiB used" << std::endl;
  }

  // timer.elapse("fina-GPU", cudaDeviceReset);

  // timer.showall();

  return 0;
//} catch (const Utility::Exception& e) {
//  std::cerr << "fatal: " << e.ToString() << std::endl;
//  return 1;
//} catch (...) {
//  std::cerr << "fatal: unknown error" << std::endl;
//  return 2;
}

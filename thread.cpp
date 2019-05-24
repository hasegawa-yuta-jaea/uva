//#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>
#include "parallel.hpp"
#include "timer.hpp"

int main(int argc, char** argv) try {
  util::parallel parallel(4);
  util::timer timer;
  timer.elapse("test", [&]() { 
    parallel.work([] (int i) {
      std::printf("%d\n", i);
    });
  });
} catch (...) {
  std::cerr << "fatal: unknown error" << std::endl;
}

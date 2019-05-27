all: 
	nvcc -O2 -arch sm_70 --compiler-bindir=g++ --compiler-options="-O2 -std=c++11 -pthread" unified.cu -o run
	./run

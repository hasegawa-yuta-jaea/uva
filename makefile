PROG:=run

.PHONY: 

all: $(TARGET)

$(PROG): unified.cu
	nvcc -O2 -std=c++11 -arch sm_70 --compiler-bindir=g++ --compiler-options="-O2 -std=c++11" unified.cu -o run


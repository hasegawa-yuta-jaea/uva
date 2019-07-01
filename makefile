PROG:=run

.PHONY: all clean

NVFLAGS := -O2 -std=c++11 -arch=sm_70 --expt-extended-lambda -Xptxas=-v

all: $(PROG)

$(PROG): unified.cu
	nvcc unified.cu -o run $(NVFLAGS)

clean:
	-rm -f $(PROG)

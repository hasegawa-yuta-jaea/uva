PROG:=run

.PHONY: all clean

NVFLAGS += --generate-code arch=compute_70,code=sm_70
NVFLAGS += --generate-code arch=compute_60,code=sm_60
NVFLAGS += -O2
NVFLAGS += -std=c++11 --expt-extended-lambda
NVFLAGS += -lineinfo
NVFLAGS += -Xptxas=-v
NVFLAGS += -Xcompiler="-Wall -Wextra"

all: $(PROG)

$(PROG): unified.cu
	nvcc unified.cu -o run $(NVFLAGS)

clean:
	-rm -f $(PROG)

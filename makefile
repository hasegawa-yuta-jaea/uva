PROG:=run

.PHONY: all clean do

NVFLAGS += --generate-code arch=compute_70,code=sm_70
NVFLAGS += --generate-code arch=compute_60,code=sm_60
NVFLAGS += -O2
NVFLAGS += -std=c++11 --expt-extended-lambda
NVFLAGS += -lineinfo
NVFLAGS += -Xptxas=-v
NVFLAGS += -Xcompiler="-Wall -Wextra"

all: clean $(PROG) do

$(PROG): unified.cu
	nvcc unified.cu -o $(PROG) $(NVFLAGS)

do:
	./$(PROG)

clean:
	-rm -f $(PROG)

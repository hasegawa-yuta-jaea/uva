PROG :=run
N ?=2

.PHONY: all clean do

NVFLAGS += --generate-code arch=compute_70,code=sm_70
NVFLAGS += --generate-code arch=compute_60,code=sm_60
NVFLAGS += -ccbin=mpic++
NVFLAGS += -O2
NVFLAGS += -std=c++11 --expt-extended-lambda
NVFLAGS += -lineinfo
NVFLAGS += -maxrregcount=64
#NVFLAGS += -Xptxas=-v
NVFLAGS += -Xcompiler="-Wall -Wextra"
NVFLAGS += -Xptxas="-warn-double-usage -warn-lmem-usage -warn-spills"

all: clean $(PROG) do

$(PROG): unified.cu
	nvcc unified.cu -o $(PROG) $(NVFLAGS)

do: $(PROG)
	mpirun \
		-n $(N) \
		-x UCX_MEMTYPE_CACHE=n \
		--mca btl_vader_single_copy_mechanism none \
		./run

clean:
	-rm -f $(PROG)

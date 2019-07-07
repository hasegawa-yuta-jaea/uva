NVCC = nvcc

NVCCFLAGS =
NVCCFLAGS+= --std c++11
NVCCFLAGS+= -lineinfo
NVCCFLAGS+= --generate-code arch=compute_70,code=sm_70

NVLDFLAGS =

test: unified.cu
	${NVCC} ${NVCCFLAGS} -o $@ $^ ${NVLDFLAGS}

clean:
	rm -f test *~

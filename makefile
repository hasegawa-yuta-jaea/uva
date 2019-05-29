ifeq ("$(shell which nvidia-smi 2>/dev/null)", "")
	TARGET:= r
else
	TARGET:= e
endif

all: $(TARGET)

e:
	nvcc -O2 -arch sm_70 --compiler-bindir=g++ --compiler-options="-O2 -std=c++11 -pthread" unified.cu -o run
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 ./run
r:
	./remote jd

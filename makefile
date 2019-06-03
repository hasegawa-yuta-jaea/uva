ifeq ("$(shell which nvidia-smi 2>/dev/null)", "")
	TARGET:= r
else
	TARGET:= e
endif

all: $(TARGET)

e:
	nvcc -O2 -arch sm_70 --compiler-bindir=g++ --compiler-options="-O2 -std=c++11" unified.cu -o run
	./run
r:
	./remote jd

PROG:=run

ifeq ("$(shell which nvidia-smi 2>/dev/null)", "")
  TARGET:= r
else
ifeq ("$(shell hostname)", "jdgx2")
  TARGET:= j
else
  TARGET:= $(PROG)
endif

PHONY: all r j

all: $(TARGET)

$(PROG): unifiled.cu
	nvcc -O2 -arch sm_70 --compiler-bindir=g++ --compiler-options="-O2 -std=c++11" unified.cu -o run

j: $(PROG)
	qsub jd.qsub
	tail -F result.txt

r:
	./remote jd

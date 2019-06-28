PROG:=run

.PHONY: all clean

all: $(PROG)

$(PROG): unified.cu
	nvcc -O2 -std=c++11 -arch sm_70 --compiler-bindir=g++ --compiler-options="-O2 -std=c++11" unified.cu -o run

clean:
	-rm -f $(PROG)

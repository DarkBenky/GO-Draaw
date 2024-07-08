NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -Xcompiler -fPIC
LDFLAGS = -shared
ARCH = sm_75  # Replace with your target GPU's compute capability

all: libmaxmul.so

libmaxmul.so: maxmul.o
	$(NVCC) $(LDFLAGS) -o $@ $^

maxmul.o: maxmul.cu
	$(NVCC) $(CFLAGS) -arch=$(ARCH) -c -o $@ maxmul.cu  # Specify architecture here

clean:
	rm -f *.o *.so
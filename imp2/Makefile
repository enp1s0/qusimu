NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I./src/cutf --expt-extended-lambda -rdc=true -O2
TARGET=qusimu

$(TARGET):src/main.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)

# Makefile for rocBLAS C++ benchmark

CXX = hipcc
CXXFLAGS = -O3 -std=c++17 --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode
ROCM_PATH = /opt/rocm
INCLUDES = -I$(ROCM_PATH)/include
LIBS = -L$(ROCM_PATH)/lib -lrocblas -lamdhip64

TARGET = rocblas-bench
SOURCE = rocblas-bench.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

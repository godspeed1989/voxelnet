CUDA_ROOT = /opt/cuda

INCLUDE += -I ${TF_INC}
INCLUDE += -I ${TF_INC}/external/nsync/public
INCLUDE += -I /opt/cuda/include
LIB = -lcudart -ltensorflow_framework -L ${CUDA_ROOT}/lib64 -L ${TF_LIB}
FLAGS = -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1
CUDA_FLAGS = -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr --Wno-deprecated-gpu-targets

all: tf_grouping_so.so

tf_grouping_g.cu.o:
	nvcc -std=c++11 -c -o $@ tf_grouping_g.cu ${INCLUDE} ${CUDA_FLAGS}

tf_grouping_so.so: tf_grouping.cpp tf_grouping_g.cu.o
	g++ -fPIC -shared ${FLAGS} $+ -o $@  ${INCLUDE} ${LIB}

clean:
	rm -rf *.o *.so

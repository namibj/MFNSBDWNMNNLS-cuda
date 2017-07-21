all: processor
main.cu: main.m4 m4_redefines.m4 m4_changeword_evil_utf8.m4
	m4 -I m4_redefines.m4 main.m4 > main.cu
processor: main.cu
	nvcc main.cu -L /usr/lib/x86_64-linux-gnu/ -lcufft_static -lculibos `pkg-config --cflags --libs opencv` -o processor -gencode arch=compute_30,\"code=sm_30\" -gencode arch=compute_35,\"code=sm_35\"  -gencode arch=compute_50,\"code=sm_50\" -gencode arch=compute_52,\"code=sm_52\"
#-gencode arch=compute_60,\"code=sm_60\" -gencode arch=compute_60,\"code=compute_60\"

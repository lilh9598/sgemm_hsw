gcc -O3 -march=armv8.2-a  -c sgemm_kernel_aarch64.c
gcc -O3 -c main.c
gcc -O3 -pthread -o sgemm_l1d main.o sgemm_kernel_aarch64.o

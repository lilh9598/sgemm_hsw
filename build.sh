gcc -march=armv8.2-a+fp16 -O3 -pthread -o sgemm_l1d main.c sgemm_kernel_aarch64.c kernel_12x8.S
./sgemm_l1d 8 512
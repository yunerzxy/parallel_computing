# Load CUDA using the following command
# module load cuda
#
CC = nvcc
CFLAGS = -O3 -arch=compute_37 -code=sm_37
NVCCFLAGS = -O3 -arch=compute_37 -code=sm_37
LIBS =

TARGETS = serial gpu autograder naive-gpu

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
gpu: gpu.o common.o
	$(CC) -o $@ $(NVCCLIBS) gpu.o common.o
naive-gpu: naive-gpu.o common.o
	$(CC) -o $@ $(NVCCLIBS) naive-gpu.o common.o
gpu-queue: gpu-queue.o common.o
	$(CC) -o $@ $(NVCCLIBS) gpu-queue.o common.o
autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o

serial.o: serial.cu common.h
	$(CC) -c $(CFLAGS) serial.cu
autograder.o: autograder.cu common.h
	$(CC) -c $(CFLAGS) autograder.cu
gpu.o: gpu.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu.cu
common.o: common.cu common.h
	$(CC) -c $(CFLAGS) common.cu
naive-gpu.o: naive-gpu.cu common.h
	$(CC) -c $(NVCCFLAGS) naive-gpu.cu
gpu-queue.o: gpu-queue.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu-queue.cu

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt

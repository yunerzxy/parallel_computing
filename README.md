# Parallel Computing - HPC, Frameworks, ALgorithms, and Techniques
Repo for Parallel Computing using supercomputer Cori. A C repo, not a Ruby one ;)

What's NERSC
National Energy Research Scientific Computing Center,(part of Lawrence Berkeley National Laboratory in Berkeley, CA) a high performance scientific computing center. NERSC provides High Performance Computing and Storage facilities and has about 6,000 active user accounts from across the U.S. and internationally. 

Major Computing Resources
Cori
A Cray XC40 with 76,416 compute cores of Intel Xeon and 658,784 compute cores of Intel Xeon Phi. The Xeon nodes have a total of 307 TB of memory, and the Xeon Phi nodes have a total of nearly 1.1 PB of memory.

Content
1. Matrix Multiplication
Use different approaches to optimize single-thread matrix multiplication. Methods include matrix transpose, block multiplication, prefetching, vectorization using AVX intrinsics, register blocking, and data alignment.

2. Mass Particles Simulations
This project simulate the repulsion among mass particles based on a slightly simplified Velocity Verlet integration, which conserves energy better than explicit Euler method. Implemented 3 different approaches to parallel the simulation.

2.1 Serial Implementation and OpenMP
Implemented serial code to reduce runtime from O(n^2) to O(n)
Implemented multi-threaded shared-memory OpenMP model to further optimize runtime to O(n/p), where p is the number of threads

2.2 MPI
Implemented MPI, the multiprocessing model with distributed memory based on serial binning teniques
Achieved O(n) runtime and O(n/p) with scaling, where p is the number of processors.
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include "common.h"

// stores particles in each bin
template <typename T, std::size_t capacity>
struct bin_t {
  int size = 0;
  T data[capacity]; // how many particles can be in the same bin

  __device__ bool insert(const T& value) {
    int index = atomicAdd(&size, 1);
    if (index >= capacity) {
      return false;
    } else {
      data[index] = value;
      return true;
    }
  }

  __device__ void clear() {size = 0;}
}

#define NUM_THREADS 256
const std::size_t capacity = 4; // experimental
const int x_offset[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
const int y_offset[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
extern double size;
//
//  benchmarking program
//

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__device__ void binning(particle_t * particles, int n,
                        bin_t<int, capacity>* bins, int num_bins_side) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;

  int bin_x = particles[tid].x / cutoff,
      bin_y = particles[tid].y / cutoff;
  bool isAssigned = bins[bin_x + bin_y * num_bins_side].insert(tid);
  if (!isAssigned) {
    std::cout << "Overflowing bin! Increase capacity." << std::endl;
  }
}

__device__ void clear_bins(particle_t* particles, bin_t<int, capacity>* bins,
                                                          int num_bins_side) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;

  int bin_x = particles[tid].x / cutoff,
      bin_y = particles[tid].y / cutoff;
  for (bin_t<int, capacity>& bin : bins) {
    bins[bin_x + bin_y * num_bins_side].clear();
  }
}

__global__ void compute_forces_gpu(particle_t * particles, int n)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particles[tid].ax = particles[tid].ay = 0;
  for(int j = 0 ; j < n ; j++)
    apply_force_gpu(particles[tid], particles[j]);

}

__global__ void compute_forces_gpu_bin (particle_t* particles, int n,
                              bin_t<int, capacity>* bins, int num_bins_side) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;

  particles[tid].ax = particles[tid].ay = 0;
  int bin_x = particles[tid].x / cutoff,
      bin_y = particles[tid].y / cutoff;
  for (int i = 0; i < 8; i++) { // find neighbor bins
    int nei_bin_x = bin_x + x_offset[i];
    int nei_bin_y = bin_y + y_offset[i];
    if (nei_bin_x < 0 || nei_bin_y < 0
      || nei_bin_x >= num_bins_side || nei_bin_y >= num_bins_side) continue;
    for (int& p_id : bins[nei_bin_x + nei_bin_y * num_bins_side].data) {
      apply_force_gpu(particles[tid], particles[p_id]);
    }
  }
}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}


int main( int argc, char **argv )
{
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );
    init_particles( n, particles );

    // GPU bins data structure
    int num_bins_side = ceil(sqrt(density * n) / cutoff);
    bin_t<int, capacity> * d_bins;
    cudaMalloc((void **) &d_bins, sizeof(bin_t<int, capacity>) *
                                        num_bins_side * num_bins_side);

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
	int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
        //
        //  binning
        //
  binning <<< blks, NUM_THREADS >>> (d_particles, n, d_bins, num_bins_side);
        //
        //  compute forces
        //
	compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n);

  clear_bins <<< blks, NUM_THREADS >>> ();
        //
        //  move particles
        //
	move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
	}
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );

    return 0;
}

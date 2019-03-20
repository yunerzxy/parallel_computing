#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"
// collision-free queue struct
template <typename T, std::size_t capacity>
struct queue {
 int size = 0;
 T data[capacity];
 __device__ bool insert(const T& value) {
  int result = atomicAdd(&size, 1);
  if (result >= capacity) {
   // Queue is overflowing. Do nothing.
   //printf("Queue overflowing\n");
   return false;
  } else {
   data[result] = value;
   return true;
  }
 }
 __device__ void clear() {
  size = 0;
  return;
 }
};
#define NUM_THREADS 256
const std::size_t capacity = 10; //* cutoff * cutoff * density;
extern double size;
//
// benchmarking program
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
 // very simple short-range repulsive force
 //
 double coef = ( 1 - cutoff / r ) / r2 / mass;
 particle.ax += coef * dx;
 particle.ay += coef * dy;
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
__global__ void populate_cells (particle_t * particles, int n, queue<int, capacity>* cells, int max_size)
{
 // Get thread (particle) ID
 int tid = threadIdx.x + blockIdx.x * blockDim.x;
 if(tid >= n) return;
 int cell_x = (int) floor(particles[tid].x / cutoff);
 int cell_y = (int) floor(particles[tid].y / cutoff);
 cells[cell_x+cell_y*max_size].insert(tid);
}
__global__ void compute_forces_gpu_fast(particle_t * particles, int n, queue<int, capacity>* cells, int max_size)
{
 // Get thread (particle) ID
 int tid = threadIdx.x + blockIdx.x * blockDim.x;
 if(tid >= n) return;
 particles[tid].ax = particles[tid].ay = 0;
 int cell_x = (int) floor(particles[tid].x / cutoff);
 int cell_y = (int) floor(particles[tid].y / cutoff);
 for (int offset_x = -1; offset_x < 2; offset_x++) {
  for (int offset_y = -1; offset_y < 2; offset_y++) {
   int _cell_x = (int) min(max(cell_x + offset_x, 0), max_size-1);
   int _cell_y = (int) min(max(cell_y + offset_y, 0), max_size-1);
   for (int idx = 0; idx < cells[_cell_x+_cell_y*max_size].size; idx++) {
    int p_idx = cells[_cell_x+_cell_y*max_size].data[idx];
    apply_force_gpu(particles[tid], particles[p_idx]);
   }
  }
 }
}
__global__ void clear_cells (particle_t * particles, int n, queue<int, capacity>* cells, int max_size)
{
 // Get thread (particle) ID
 int tid = threadIdx.x + blockIdx.x * blockDim.x;
 if(tid >= n) return;
 int cell_x = (int) floor(particles[tid].x / cutoff);
 int cell_y = (int) floor(particles[tid].y / cutoff);
 cells[cell_x+cell_y*max_size].clear();
}
__global__ void move_gpu (particle_t * particles, int n, double size)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;
  particle_t * p = &particles[tid];
  //
  // slightly simplified Velocity Verlet integration
  // conserves energy better than explicit Euler method
  //
  p->vx += p->ax * dt;
  p->vy += p->ay * dt;
  p->x += p->vx * dt;
  p->y += p->vy * dt;
  //
  // bounce from walls
  //
  while( p->x < 0 || p->x > size )
  {
    p->x = p->x < 0 ? -(p->x) : 2*size-p->x;
    p->vx = -(p->vx);
  }
  while( p->y < 0 || p->y > size )
  {
    p->y = p->y < 0 ? -(p->y) : 2*size-p->y;
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
  cudaThreadSynchronize();
  double copy_time = read_timer( );
  int max_size = ceil(sqrt( density * n ) / cutoff);
  queue<int, capacity>* d_cells;
  cudaMalloc(&d_cells, sizeof(queue<int, capacity>) * max_size * max_size);
  // std::vector< std::vector< std::vector< int > > > cells(max_size, std::vector< std::set< int > > (max_size, std::vector< int >()));
  // Copy the particles to the GPU
  cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);
  cudaThreadSynchronize();
  copy_time = read_timer( ) - copy_time;
  //
  // simulate a number of time steps
  //
  cudaThreadSynchronize();
  double simulation_time = read_timer( );
  for( int step = 0; step < NSTEPS; step++ )
  {
 // equivalent to int blks = ceil(n/NUM_THREADS)
  int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    //
    // populate cells
    //
 populate_cells <<< blks, NUM_THREADS >>> (d_particles, n, d_cells, max_size);
    //
    // compute forces
    //
  // compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n);
 compute_forces_gpu_fast <<< blks, NUM_THREADS >>> (d_particles, n, d_cells, max_size);
    //
    // clear cells
    //
 clear_cells <<< blks, NUM_THREADS >>> (d_particles, n, d_cells, max_size);
    //
    // move particles
    //
  move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
    //
    // save if necessary
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

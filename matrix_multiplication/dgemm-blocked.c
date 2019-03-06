const char* dgemm_desc = "Transpose";

#include <stdlib.h>
#if !defined(BLOCK_SIZE)
  #define BLOCK_SIZE 40
#endif
#define BLOCK_SIZE_R 40
#define BLOCK_SIZE_TR 64

#define min(a,b) (((a) < (b))? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
    //For each column j of B
    for (int j = 0; j < N; ++j) {
      // For each row i of A
      for (int i = 0; i < M; ++i) {
        // Compute C(i,j)
        int cpos = i+j*lda;
        double cij = C[cpos];
        for (int k = 0; k < K; ++k) {
          cij = cij + A[i+k*lda] * B[k+j*lda];
        }
        C[cpos] = cij;
    }
  }
}
static void do_transpose_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  int cpos;
  double cij;
  //For each column j of B
  for (int j = 0; j < N; ++j) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
      // Compute C(i,j)
      cpos = i+j*lda;
      cij = C[cpos];
      //__builtin_prefetch(A+i*lda+lda);
      for (int k = 0; k < K; ++k) {
        cij = cij + A[k+i*lda] * B[k+j*lda];
      }
      C[cpos] = cij;
    }
  }
}

void transpose(const int lda, double* restrict A, double* restrict AT) {
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < lda; ++j) {
      AT[j + i*lda] = A[i + j*lda];
    }
  }
}

void transpose_block(const int lda, double* restrict A, double* restrict AT) {
  int i;
  int j;
  int idL;
  int jdL;
  for (int jb = 0; jb < lda; jb += BLOCK_SIZE_TR) {
    jdL = min(BLOCK_SIZE_TR, lda - jb);
    for (int ib = 0; ib < lda; ib += BLOCK_SIZE_TR) {
      idL = min(BLOCK_SIZE_TR, lda - ib);
      for (int jd = 0; jd < jdL; ++jd) {
        j = jb + jd;
        for (int id = 0; id < idL; ++id) {
          i = ib + id;
          AT[j + i*lda] = A[i + j*lda];
        }
      }
    }
  }
}
void self_transpose(const int lda, double* A) {
  int p1;
  int p2;
  double tmp;
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < i; ++j) {
      p1 = j + i*lda;
      p2 = i + j*lda;
      tmp = A[p1];
      A[p1] = A[p2];
      A[p2] = tmp;
    }
  }
}
void self_transpose_block(const int lda, double* A) {
  int i;
  int j;
  double tmp;
  for (int jb = 0; jb < lda; jb += BLOCK_SIZE_TR) {
    for (int ib = jb; ib < lda; ib += BLOCK_SIZE_TR) {
      for (int jd = 0; jd < BLOCK_SIZE_TR; ++jd) {
        j = jb + jd;
        for (int id = 0; id < BLOCK_SIZE_TR; ++id) {
          i = ib + id;
          if (j < i && i < lda) {
            tmp = A[j + i*lda];
            A[j + i*lda] = A[i + j*lda];
            A[i + j*lda] = tmp;
          }
        }
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm_blocked (int lda, double* A, double* B, double* C)
{
    // For each block-column of B
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
      // Accumulate block dgemms into block of C
      for (int k = 0; k < lda; k += BLOCK_SIZE) {
        // For each block-row of A
        for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // Correct block dimensions if block "goes off edge of" the matrix
        int M = min (BLOCK_SIZE, lda-i);
        int N = min (BLOCK_SIZE, lda-j);
        int K = min (BLOCK_SIZE, lda-k);
        // Perform individual block dgemm
        do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}
void square_dgemm_transpose_blocked (int lda, double* A, double* B, double* C)
{
  self_transpose(lda, A);
  // For each block-column of B
  for (int j = 0; j < lda; j += BLOCK_SIZE) {
    // Accumulate block dgemms into block of C
    for (int k = 0; k < lda; k += BLOCK_SIZE) {
      // For each block-row of A
      for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // Correct block dimensions if block "goes off edge of" the matrix
        int M = min (BLOCK_SIZE, lda-i);
        int N = min (BLOCK_SIZE, lda-j);
        int K = min (BLOCK_SIZE, lda-k);
        // Perform individual block dgemm
        do_transpose_block(lda, M, N, K, A + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
  self_transpose(lda, A);
}
void square_dgemm_transpose (const int lda, double* A, double const* const B, double* restrict C)
{
  //double* AT = malloc(sizeof(double) * lda * lda);
  self_transpose(lda, A);
  int bpos;
  int apos;
  int cpos;
  double cij;
  //self_transpose_block(lda, A);
  /* For each block-column of B */
  for (int j = 0; j < lda; ++j) {
    /* For each block-row of A */
    bpos = j*lda;
    //__builtin_prefetch(B+bpos);
    //This causes a minor slowdown
    for (int i = 0; i < lda; ++i) {
      cpos = i + j*lda;
      cij = C[cpos];
      apos = i*lda;
      for (int k = 0; k < lda; ++k) {
        cij += A[k + apos] * B[k + bpos];
      }
      C[cpos] = cij;
    }
  }
  self_transpose_block(lda, A);
  //free(AT);
}

//Vectorization is slower on Cori. I assume it does automatic
//vectorization more efficiently.

/*double* z = calloc(VECTOR_PACK, sizeof(double));
__m256d aggr = _mm256_loadu_pd(z);
for (int k = 0; k < lda/VECTOR_PACK; ++k) {
  __m256d a_vec = _mm256_loadu_pd(A);
  __m256d b_vec = _mm256_loadu_pd(B);
  aggr = _mm256_add_pd(aggr,
          _mm256_mul_pd(a_vec, b_vec));
  A += VECTOR_PACK;
  B += VECTOR_PACK;
}
_mm256_storeu_pd(z, aggr);
for (int ii = 0; ii < VECTOR_PACK; ++ii) {
  cij += z[ii];
}
for (int k = lda/VECTOR_PACK*VECTOR_PACK; k < lda; ++k) {
  cij += *(A++) * *(B++);
}*/

int max(int a, int b, int c) {
  if (b > a) {
    if (c > b) {
      return c;
    }
    return b;
  } else if (c > a) {
    return c;
  }
  return a;
}

//A is NxM, B is MxP, C is NxP.
void square_dgemm_recursive (int lda, int n, int m, int p, double* A, double* B, double* C)
{
  int w = max(n, m, p);
  if (w <= BLOCK_SIZE_R) {
    do_block(lda, n, p, m, A, B, C);
  } else {
    if (w == n) {
      // Split A horizontally
      int n1 = n/2;
      int n2 = n - n/2;
      square_dgemm_recursive(lda, n1, m, p, A, B, C);
      square_dgemm_recursive(lda, n2, m, p, A+n1, B, C+n1);
    } else if (w == p) {
      //Split B vertically
      int p1 = p/2;
      int p2 = p - p/2;
      square_dgemm_recursive(lda, n, m, p1, A, B, C);
      square_dgemm_recursive(lda, n, m, p2, A, B+p1*lda, C+p1*lda);
    } else {
      //Split both
      int m1 = m/2;
      int m2 = m - m/2;
      square_dgemm_recursive(lda, n, m1, p, A, B, C);
      square_dgemm_recursive(lda, n, m2, p, A+m1*lda, B+m1, C);
    }
  }
}
void square_dgemm_recursive_root(int lda, double* A, double* B, double* C) {
  square_dgemm_recursive(lda, lda, lda, lda, A, B, C);
}


//A is NxM, B is MxP, C is NxP.
void square_dgemm_recursive_transpose (int lda, int n, int m, int p, double* A, double* B, double* C)
{
  int w = max(n, m, p);
  if (w <= BLOCK_SIZE_R) {
    do_transpose_block(lda, n, p, m, A, B, C);
  } else {
    if (w == n) {
      // Split A horizontally
      int n1 = n/2;
      int n2 = n - n/2;
      square_dgemm_recursive_transpose(lda, n1, m, p, A, B, C);
      square_dgemm_recursive_transpose(lda, n2, m, p, A+n1*lda, B, C+n1);
    } else if (w == p) {
      //Split B vertically
      int p1 = p/2;
      int p2 = p - p/2;
      square_dgemm_recursive_transpose(lda, n, m, p1, A, B, C);
      square_dgemm_recursive_transpose(lda, n, m, p2, A, B+p1*lda, C+p1*lda);
    } else {
      //Split both
      int m1 = m/2;
      int m2 = m - m/2;
      square_dgemm_recursive_transpose(lda, n, m1, p, A, B, C);
      square_dgemm_recursive_transpose(lda, n, m2, p, A+m1, B+m1, C);
    }
  }
}


typedef struct {
    int n;
    int m;
    int p;
    double* A;
    double* B;
    double* C;
} mult;

typedef struct {
    int size;
    int index;
    mult* data;
} stack;

stack* new_stack(int size) {
  stack* s = malloc(sizeof(stack));
  s->size = size;
  s->index = 0;
  s->data = malloc(sizeof(mult) * size);
  return s;
}
void stack_add(stack* s, mult m) {
  if (s->index >= s->size) {
    s->size *= 2;
    s->data = realloc(s->data, sizeof(mult) * s->size);
  }
  s->data[(s->index)++] = m;
}
mult stack_pop(stack* s) {
  s->index = s->index - 1;
  return s->data[s->index];
}
//A is NxM, B is MxP, C is NxP.
void square_dgemm_queue_transpose (int lda, double* A, double* B, double* C)
{
  self_transpose_block(lda, A);
  stack* s = new_stack(32);
  mult deflt = {.n=lda, .m=lda, .p=lda, .A=A, .B=B, .C=C};
  stack_add(s, deflt);
  while (s->index > 0) {
    mult m = stack_pop(s);
    int w = max(m.n, m.m, m.p);
    if (w <= BLOCK_SIZE_R) {
      do_transpose_block(lda, m.n, m.p, m.m, m.A, m.B, m.C);
    } else {
      if (w == m.n) {
        // Split A horizontally
        int n1 = m.n/2;
        int n2 = m.n - m.n/2;
        mult newm1 = {.n=n1, .m=m.m, .p=m.p, .A=m.A, .B=m.B, .C=m.C};
        stack_add(s, newm1);
        mult newm2 = {.n=n2, .m=m.m, .p=m.p, .A=m.A+n1*lda, .B=m.B, .C=m.C+n1};
        stack_add(s, newm2);
      } else if (w == m.p) {
        //Split B vertically
        int p1 = m.p/2;
        int p2 = m.p - m.p/2;
        mult newm1 = {.n=m.n, .m=m.m, .p=p1, .A=m.A, .B=m.B, .C=m.C};
        stack_add(s, newm1);
        mult newm2 = {.n=m.n, .m=m.m, .p=p2, .A=m.A, .B=m.B+p1*lda, .C=m.C+p1*lda};
        stack_add(s, newm2);
      } else {
        //Split both
        int m1 = m.m/2;
        int m2 = m.m - m.m/2;
        mult newm1 = {.n=m.n, .m=m1, .p=m.p, .A=m.A, .B=m.B, .C=m.C};
        stack_add(s, newm1);
        mult newm2 = {.n=m.n, .m=m2, .p=m.p, .A=m.A+m1, .B=m.B+m1, .C=m.C};
        stack_add(s, newm2);
      }
    }
  }
  self_transpose_block(lda, A);
}
void square_dgemm_recursive_root_transpose (int lda, double* A, double* B, double* C)
{
  self_transpose(lda, A);
  square_dgemm_recursive_transpose(lda, lda, lda, lda, A, B, C);
  self_transpose(lda, A);
}

void square_dgemm (int lda, double* A, double* B, double* C)
{
  square_dgemm_transpose(lda, A, B, C);
}

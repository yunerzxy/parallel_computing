

void transpose_irr(const int height, const int width, const int lda, double* A, double* AT) {
  for (int j = 0; j < width; ++j) {
    for (int i = 0; i < height; ++i) {
      AT[j + i*width] = A[i + j*lda];
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
  for (int i = 0; i < lda; ++i) {
    for (int j = 0; j < i; ++j) {
      double tmp = A[j + i*lda];
      A[j + i*lda] = A[i + j*lda];
      A[i + j*lda] = tmp;
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

void square_dgemm_transpose (const int lda, double* A, double const* const B, double* restrict C)
{
  double* AT = malloc(sizeof(double) * lda * lda);
  transpose_block(lda, A, AT);
  //self_transpose_block(lda, A);
  /* For each block-column of B */
  for (int j = 0; j < lda; ++j) {
    /* For each block-row of A */
    for (int i = 0; i < lda; ++i) {
      double cij = C[i+j*lda];
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
      for (int k = 0; k < lda; ++k) {
        cij += AT[k + i*lda] * B[k + j*lda];
      }
      C[i+j*lda] = cij;
    }
  }
  //Should free AT here
}

/*
 * par_gauss.c
 *
 * CS 470 Project 2 (OpenMP)
 * OpenMP parallelized version
 *
 * @Authors Ryan Gaffney, Jacob Gottschalk
 *
 * Compile with --std=c99
 */

#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// custom timing macros
#include "timer.h"
#ifdef _OPENMP
#include <omp.h>
#endif
// uncomment this line to enable the alternative back substitution method
#define USE_COLUMN_BACKSUB

// use 64-bit IEEE arithmetic (change to "float" to use 32-bit arithmetic)
#define REAL double

// linear system: Ax = b    (A is n x n matrix; b and x are n x 1 vectors)
int n;
unsigned int used_threads = 1;
REAL *A;
REAL *x;
REAL *b;

// enable/disable debugging output (don't enable for large matrix sizes!)
bool debug_mode = false;

// enable/disable triangular mode (to skip the Gaussian elimination phase)
bool triangular_mode = false;

// NOTE: Currently gives a few extra seconds, may be able to make faster.
// I think other medthods may be taking up the majority of the time.
/*
 * Generate a random linear system of size n.
 *
 * In order to achieve parallelization I made a few additions and changes to the
 * function For starters this function uses the threads id for the seed as
 * stated in the spec. This in turn provides some more randomness when creating
 * matricies. However in order to get the thread ID OpenMP required its
 * functinos to be inside of a parallel section. Doing this also has the added
 * benefit of only needing to spawn our threads once. Lastly what was added was
 * two "for" directives. Since there were two seprate nested loops this seemed
 * to be the easiest and most efficient way. The code follows the same logic as
 * the serial version but adds those two "for" directives to improve our overall
 * performance In terms of scheduling I did not notice much of a performance
 * difference when adding that clause to the pragma. As a result I chose to
 * leave scheduling out.
 *
 * In short: Code follows same logic as serial version but we use our thread ID
 * for the seed and added for directives to improve performance and fully
 * utilize our spawned threads.
 */
void rand_system() {
  // allocate space for matrices
  A = (REAL *)calloc(n * n, sizeof(REAL));
  b = (REAL *)calloc(n, sizeof(REAL));
  x = (REAL *)calloc(n, sizeof(REAL));

  // verify that memory allocation succeeded
  if (A == NULL || b == NULL || x == NULL) {
    printf("Unable to allocate memory for linear system\n");
    exit(EXIT_FAILURE);
  }

  // initialize pseudorandom number generator
  // (see https://en.wikipedia.org/wiki/Linear_congruential_generator)
  // Seeding the seed with the thread id
  // int row, col;
#pragma omp parallel default(none) \
    shared(A, b, n, triangular_mode, used_threads)
  {
// First make sure openMP is enabled
#ifdef _OPENMP
    // Set the seed to our thread ID
    unsigned long seed = omp_get_thread_num();
    // This tells use how many threads were used
    used_threads = omp_get_num_threads();
#else
    unsigned long seed = 0;
#endif

// generate random matrix entries
// Utilizing the for directive to split up the work and utilize our threads
#pragma omp for
    for (int row = 0; row < n; row++) {
      int col = triangular_mode ? row : 0;
      for (; col < n; col++) {
        if (row != col) {
          seed = (1103515245 * seed + 12345) % (1 << 31);
          A[row * n + col] = (REAL)seed / (REAL)ULONG_MAX;
        } else {
          A[row * n + col] = n / 10.0;
        }
      }
    }

// generate right-hand side such that the solution matrix is all 1s
// Same idea as before. Utilizing the for directive to split up the work and use
// our threads
#pragma omp for
    for (int row = 0; row < n; row++) {
      b[row] = 0.0;
      for (int col = 0; col < n; col++) {
        b[row] += A[row * n + col] * 1.0;
      }
    }
  }
}

/*
 * Reads a linear system of equations from a file in the form of an augmented
 * matrix [A][b].
 *
 * Don't Parallelize
 */
void read_system(const char *fn) {
  // open file and read matrix dimensions
  FILE *fin = fopen(fn, "r");
  if (fin == NULL) {
    printf("Unable to open file \"%s\"\n", fn);
    exit(EXIT_FAILURE);
  }
  if (fscanf(fin, "%d\n", &n) != 1) {
    printf("Invalid matrix file format\n");
    exit(EXIT_FAILURE);
  }

  // allocate space for matrices
  A = (REAL *)malloc(sizeof(REAL) * n * n);
  b = (REAL *)malloc(sizeof(REAL) * n);
  x = (REAL *)malloc(sizeof(REAL) * n);

  // verify that memory allocation succeeded
  if (A == NULL || b == NULL || x == NULL) {
    printf("Unable to allocate memory for linear system\n");
    exit(EXIT_FAILURE);
  }

  // read all values
  for (int row = 0; row < n; row++) {
    for (int col = 0; col < n; col++) {
      if (fscanf(fin, "%lf", &A[row * n + col]) != 1) {
        printf("Invalid matrix file format\n");
        exit(EXIT_FAILURE);
      }
    }
    if (fscanf(fin, "%lf", &b[row]) != 1) {
      printf("Invalid matrix file format\n");
      exit(EXIT_FAILURE);
    }
    x[row] = 0.0;  // initialize x while we're reading A and b
  }
  fclose(fin);
}

/*
 * Performs Gaussian elimination on the linear system.
 * Assumes the matrix is singular and doesn't require any pivoting.
 *
 * Parallelize
 */
void gaussian_elimination() {
  for (int pivot = 0; pivot < n; pivot++) {
#pragma omp parallel for default(none) shared(A, b, n, pivot)
    for (int row = pivot + 1; row < n; row++) {
      REAL coeff = A[row * n + pivot] / A[pivot * n + pivot];
      A[row * n + pivot] =
          0.0;  // everything in the column below pivot cell (no race)
      for (int col = pivot + 1; col < n; col++) {
        A[row * n + col] -= A[pivot * n + col] * coeff;
      }
      b[row] -= b[pivot] * coeff;
    }
  }
}

/*
 * Performs backwards substitution on the linear system.
 * (row-oriented version)
 *
 * Parallelize
 *
 * To Parallelize this function we needed to put a little more effort into
 * thinking this through but this is what we were able to come up with in order
 * to parallelize. The following has weak scaling on smaller inputs (<10000) but
 * tends to improve later on. Like in the rand_system method threads are only
 * spawned once. This would provide us with the best speedup Otherwise you may
 * need to spawn your threads inside of the for loop which results in some more
 * wasted time since the threads need to spawn and then join. To alleviate this
 * a single call to parallel was used in addition to a barrier, single and a for
 * directive with the reduction clause. Combining these three directives gave us
 * the correct output when spawning the threads at the top. The first barrier is
 * used to ensure no threads enter our for loop before tmp is set. This helpst
 * becouse our temp is a data dependency. Next a reduction is used on tmp with
 * the + opperator since we are summing. Then lastly the single directive is
 * used. We use single here since we only want our x[row] to be set once by a
 * thread And the implicit barrier that comes with the single directive stops
 * our threads from changing tmp before it gets reset.
 */
void back_substitution_row() {
  REAL tmp = 0;
#pragma omp parallel default(none) shared(A, b, x, n, tmp)
  for (int row = n - 1; row >= 0; row--) {
    tmp = b[row];

#pragma omp barrier
#pragma omp for reduction(+ : tmp)
    for (int col = row + 1; col < n; col++) {
      tmp += -A[row * n + col] * x[col];
    }

#pragma omp single
    x[row] = tmp / A[row * n + row];
  }
}

/*
 * Performs backwards substitution on the linear system.
 * (column-oriented version)
 *
 * Like some of the previous functions we only spawn our threads once.
 * Like before we use a single parallel directive and multiple for directives
 * this is because we have multiple seperate for loops so this was needed to
 * improve performance. lastly the main value that had a data dependence was our
 * x[col] right before the second for directive this was used to ensure we get
 * the correct value in our x matrix. I was unable to successuflly parallelize
 * that foor loop right before the "single" directive without getting the
 * correct output. Which is likely do to the fact calculations on the x matrix
 * provides a loop carried dependency. Because the same matrix is shared among
 * our threads.
 */
void back_substitution_column() {
#pragma omp parallel default(none) shared(x, n, b, A)
  {
#pragma omp for
    for (int row = 0; row < n; row++) {
      x[row] = b[row];
    }

    for (int col = n - 1; col >= 0; col--) {
#pragma omp single
      x[col] /= A[col * n + col];
#pragma omp for
      for (int row = 0; row < col; row++) {
        x[row] += -A[row * n + col] * x[col];
      }
    }
  }
}

/*
 * Find the maximum error in the solution (only works for randomly-generated
 * matrices).
 */
REAL find_max_error() {
  REAL error = 0.0, tmp;
  for (int row = 0; row < n; row++) {
    tmp = fabs(x[row] - 1.0);
    if (tmp > error) {
      error = tmp;
    }
  }
  return error;
}

/*
 * Prints a matrix to standard output in a fixed-width format.
 */
void print_matrix(REAL *mat, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      printf("%8.1e ", mat[row * cols + col]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  // check and parse command line options
  int c;
  while ((c = getopt(argc, argv, "dt")) != -1) {
    switch (c) {
      case 'd':
        debug_mode = true;
        break;
      case 't':
        triangular_mode = true;
        break;
      default:
        printf("Usage: %s [-dt] <file|size>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }
  if (optind != argc - 1) {
    printf("Usage: %s [-dt] <file|size>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // read or generate linear system
  long int size = strtol(argv[optind], NULL, 10);
  START_TIMER(init)
  if (size == 0) {
    read_system(argv[optind]);
  } else {
    n = (int)size;
    rand_system();
  }
  STOP_TIMER(init)

  if (debug_mode) {
    printf("Original A = \n");
    print_matrix(A, n, n);
    printf("Original b = \n");
    print_matrix(b, n, 1);
  }

  // perform gaussian elimination
  START_TIMER(gaus)
  if (!triangular_mode) {
    gaussian_elimination();
  }
  STOP_TIMER(gaus)

  // perform backwards substitution
  START_TIMER(bsub)
#ifndef USE_COLUMN_BACKSUB
  back_substitution_row();
#else
  back_substitution_column();
#endif
  STOP_TIMER(bsub)

  if (debug_mode) {
    printf("Triangular A = \n");
    print_matrix(A, n, n);
    printf("Updated b = \n");
    print_matrix(b, n, 1);
    printf("Solution x = \n");
    print_matrix(x, n, 1);
  }

  // print results
  printf("Nthreads=%2d  ERR=%8.1e  INIT: %8.4fs  GAUS: %8.4fs  BSUB: %8.4fs\n",
         used_threads, find_max_error(), GET_TIMER(init), GET_TIMER(gaus),
         GET_TIMER(bsub));

  // clean up and exit
  free(A);
  free(b);
  free(x);
  return EXIT_SUCCESS;
}

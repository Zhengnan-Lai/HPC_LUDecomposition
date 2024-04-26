#include <mpi.h>
#include <stdio.h>
#include <algorithm>

const char* lu_desc = "Parallel LU decomposition.";

// Perform a LU decomposition on (m x m) blocks of A, L, U
void do_block(int n, double* A, double* L, double* U, int m){
    for(int i = 0; i < m; i++) for(int j = i; j < m; j++){
        U[i * n + j] = A[i * n + j];
    }
    for(int k = 0; k < m; k++){
        L[k * n + k] = 1;
        for(int i = k + 1; i < m; i++){
            L[i * n + k] = A[i * n + k] / A[k * n + k];
        }
        for(int j = k + 1; j < m; j++) for(int i = k + 1; i < m; i++){
            U[i * n + j] = U[i * n + j] - L[i * n + k] * U[k * n + j];
        }
    }
}

/*
 * This routine performs a LU decomposition
 *  A = L * U
 * where A, L, and U are lda-by-lda matrices stored in row-major format.
 * On exit, A maintain its input values.
 */
void lu_decomposition(int n, double* A, double* L, double* U, int rank, int num_procs) {
	int colsPerProcessor = (n + num_procs - 1) / num_procs;
	int startCol = std::min(rank * colsPerProcessor, n);
	int endCol = std::min((rank + 1) * colsPerProcessor, n);

    for(int i = 0; i < n; i++) for(int j = 0; j < n; j++){
        U[i * n + j] = A[i * n + j];
    }

	for (int k = 0; k < n-1; k++) {
		if (startCol <= k && endCol > k) {
			for (int i = k + 1; i < n; i++) {
				L[i * n + k] = U[i * n + k] / U[k * n + k];
			}
		}

		// broadcast L[i][k] for k < i <= n
		/*
		for (int i = k + 1; i < n; i++) {
			MPI_Bcast(&L[i * n + k], 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
		}
		*/

		for (int j = std::max(k+1, startCol); j < endCol; j++) {
			for (int i = k + 1; i < n; i++) {
				U[i * n + j] = U[i * n + j] - L[i * n + k] * U[k * n + j];
			}
		}
	}

    for(int i = 0; i < n; i++) for(int j = 0; j < i; j++){
        U[i * n + j] = 0;
    }
}

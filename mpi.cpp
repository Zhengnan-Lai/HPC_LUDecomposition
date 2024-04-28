#include <mpi.h>
#include <stdio.h>
#include <algorithm>

const char* lu_desc = "Parallel LU decomposition with MPI.";

/*
 * This routine performs a LU decomposition
 *  A = L * U
 * where A, L, and U are lda-by-lda matrices stored in row-major format.
 * On exit, A is overwritten by U.
 */
void lu_decomposition(int n, double* A, double* L, int rank, int num_procs) {
	// int colsPerProcessor = (n + num_procs - 1) / num_procs;
	// int startCol = std::min(rank * colsPerProcessor, n);
	// int endCol = std::min((rank + 1) * colsPerProcessor, n);

	// for (int k = 0; k < n; k++) {
	// 	if (startCol <= k && endCol > k) {
	// 		L[k * n + k] = 1;
	// 		for (int i = k + 1; i < n; i++) {
	// 			L[i * n + k] = A[i * n + k] / A[k * n + k];
	// 		}
	// 	}
	// 	// broadcast L[i][k] for k < i <= n
	// 	for (int i = k + 1; i < n; i++) {
	// 		MPI_Bcast(&L[i * n + k], 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
	// 	}
	// 	for (int j = std::max(k+1, startCol); j < endCol; j++) {
	// 		for (int i = k + 1; i < n; i++) {
	// 			A[i * n + j] = A[i * n + j] - L[i * n + k] * A[k * n + j];
	// 		}
	// 		A[j * n + k] = 0;
	// 	}
	// }
	int rowsPerProcessor = (n + num_procs - 1) / num_procs;
	int startRow = std::min(rank * rowsPerProcessor, n);
	int endRow = std::min((rank + 1) * rowsPerProcessor, n);
	for (int k = 0; k < n; k++) {
		// broadcast A[k][j] for k <= j < n
		MPI_Bcast(&A[k * n + k], n - k, MPI_DOUBLE, k / rowsPerProcessor, MPI_COMM_WORLD);
		L[k * n + k] = 1;
		for (int i = std::max(k+1, startRow); i < endRow; i++) {
			L[i * n + k] = A[i * n + k] / A[k * n + k];
		}
		for (int j = k + 1; j < n; j++) {
			for (int i = std::max(k+1, startRow); i < endRow; i++) {
				A[i * n + j] = A[i * n + j] - L[i * n + k] * A[k * n + j];
			}
			A[j * n + k] = 0;
		}
	}
	if (rank == 0) {
		// Root process: prepare to receive data from all processes, including itself
		MPI_Gather(MPI_IN_PLACE, (endRow-startRow)*n, MPI_DOUBLE, L, (endRow-startRow)*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(MPI_IN_PLACE, (endRow-startRow)*n, MPI_DOUBLE, A, (endRow-startRow)*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	} else {
		// Non-root processes: send data to root
		MPI_Gather(&L[startRow*n], (endRow-startRow)*n, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(&A[startRow*n], (endRow-startRow)*n, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
}

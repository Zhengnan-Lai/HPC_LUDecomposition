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
		MPI_Gather(MPI_IN_PLACE, (endRow-startRow)*n, MPI_DOUBLE, L, (endRow-startRow)*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	} else {
		MPI_Gather(&L[startRow*n], (endRow-startRow)*n, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
}

struct{
	double value;
	int index;
} local_p, global_p;

// Find the max element in the k-th column and [j, l)-th row
void find_column_max(int n, double* A, int k, int j, int l, double* value, int* index){
	*value = std::abs(A[j * n + k]); *index = j;
    for(int i = j + 1; i < l; i++){
        if(std::abs(A[i * n + k]) > *value){
            *value = std::abs(A[i * n + k]);
            *index = i;
        }
    }
}


/*
 * This routine performs a pivoted LU decomposition
 *  P * A = L * U
 * where A, L, and U are n-by-n matrices stored in row-major format, P is stored in one dimensional array
 * On exit, A is overwritten by U.
 */
void pivoted_lu_decomposition(int n, double* A, double* L, int* P, int rank, int num_procs){
	int rowsPerProcessor = (n + num_procs - 1) / num_procs;
	int startRow = std::min(rank * rowsPerProcessor, n);
	int endRow = std::min((rank + 1) * rowsPerProcessor, n);
	for (int k = 0; k < n; k++) {
		// find the local max element in parallel
		find_column_max(n, A, k, std::max(k, startRow), std::max(k, endRow), &local_p.value, &local_p.index);
		// communicate to find the largest element
		MPI_Allreduce(&local_p, &global_p, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
		P[k] = global_p.index;
		// swap row if necessary
		if(P[k] != k){
			if(k / rowsPerProcessor == P[k] / rowsPerProcessor){
				// if in the same processor, just swap
				if(rank == k / rowsPerProcessor){
					std::swap_ranges(&A[k * n], &A[(k + 1) * n], &A[P[k] * n]);
					std::swap_ranges(&L[k * n], &L[k * n + k], &L[P[k] * n]);
				}
			}
			else{
				// if not in the same processor, communicate
				if(rank == k / rowsPerProcessor){
					MPI_Send(&A[k * n], n, MPI_DOUBLE, P[k] / rowsPerProcessor, 0, MPI_COMM_WORLD);
					MPI_Send(&L[k * n], k, MPI_DOUBLE, P[k] / rowsPerProcessor, 1, MPI_COMM_WORLD);
				}
				if(rank == P[k] / rowsPerProcessor){
					MPI_Send(&A[P[k] * n], n, MPI_DOUBLE, k / rowsPerProcessor, 2, MPI_COMM_WORLD);
					MPI_Send(&L[P[k] * n], k, MPI_DOUBLE, k / rowsPerProcessor, 3, MPI_COMM_WORLD);
				}
				if(rank == k / rowsPerProcessor){
					MPI_Recv(&A[k * n], n, MPI_DOUBLE, P[k] / rowsPerProcessor, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(&L[k * n], k, MPI_DOUBLE, P[k] / rowsPerProcessor, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if(rank == P[k] / rowsPerProcessor){
					MPI_Recv(&A[P[k] * n], n, MPI_DOUBLE, k / rowsPerProcessor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(&L[P[k] * n], k, MPI_DOUBLE, k / rowsPerProcessor, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
		}
		// broadcast A[k][j] for k <= j < n
		MPI_Bcast(&A[k * n + k], n - k, MPI_DOUBLE, k / rowsPerProcessor, MPI_COMM_WORLD);
		// Gaussian elimination
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
		MPI_Gather(MPI_IN_PLACE, (endRow-startRow)*n, MPI_DOUBLE, L, (endRow-startRow)*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	} else {
		MPI_Gather(&L[startRow*n], (endRow-startRow)*n, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
}
const char* lu_desc = "Naive LU decomposition.";

/*
 * This routine performs a LU decomposition
 *  A = L * U
 * where A, L, and U are lda-by-lda matrices stored in row-major format.
 * On exit, A maintain its input values.
 */
void lu_decomposition(int n, double* A, double* L, double* U, int rank, int num_procs) {
    for(int i = 0; i < n; i++) for(int j = 0; j < n; j++){
        U[i * n + j] = A[i * n + j];
    }
    for(int k = 0; k < n; k++){
        L[k * n + k] = 1;
        for(int i = k + 1; i < n; i++){
            L[i * n + k] = U[i * n + k] / U[k * n + k];
        }
        for(int j = k + 1; j < n; j++) for(int i = k + 1; i < n; i++){
            U[i * n + j] = U[i * n + j] - L[i * n + k] * U[k * n + j];
        }
    }
	
    for(int i = 0; i < n; i++) for(int j = 0; j < i; j++){
        U[i * n + j] = 0;
    }
}

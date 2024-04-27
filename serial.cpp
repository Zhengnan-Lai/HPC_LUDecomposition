const char* lu_desc = "Naive LU decomposition.";

/*
 * This routine performs a LU decomposition
 *  A = L * U
 * where A, L, and U are lda-by-lda matrices stored in row-major format.
 * On exit, A is overwritten by U.
 */
void lu_decomposition(int n, double* A, double* L, int rank, int num_procs) {
    for(int k = 0; k < n; k++){
        L[k * n + k] = 1;
        for(int i = k + 1; i < n; i++){
            L[i * n + k] = A[i * n + k] / A[k * n + k];
        }
        for(int j = k + 1; j < n; j++){
            for(int i = k + 1; i < n; i++){
                A[i * n + j] = A[i * n + j] - L[i * n + k] * A[k * n + j];
            }
            A[j * n + k] = 0;
        }
    }
}

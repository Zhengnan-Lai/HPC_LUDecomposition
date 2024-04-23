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
void lu_decomposition(int n, double* A, double* L, double* U, int m) {
    // Decompose A_11 = L_11 * U_11
    do_block(n, A, L, U, n/2);
    // Solve A_12 = L_11 * U_12 for U_12

    // Solve A_21 = L_21 * U_11 for L_21

    // Recurse on A_22 - L_21 * U_12 = L_22 * U_22
    lu_decomposition(n, A, L, U, m);
}
